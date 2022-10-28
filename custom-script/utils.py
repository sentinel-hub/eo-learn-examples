import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from eolearn.core import EOPatch, FeatureType
from sentinelhub import DataCollection, MimeType, SentinelHubRequest, SHConfig

PRECISION_SCORES = 4
PRECISION_THRESHOLD = None

BANDS = ["B02", "B03", "B04", "B08", "B11"]
BANDS_STR = ",".join(BANDS)
MODEL_INPUTS = ["B02", "B03", "B04", "NDWI", "NDMI"]
MODEL_INPUTS_STR = ", ".join(MODEL_INPUTS)


def prepare_data_helper(
    array: List[EOPatch],
    feature: Tuple[FeatureType, str],
) -> np.ndarray:
    """
    Function that reshapes data into a np.array of correct dimensions
    """
    tmp_data = np.array([eopatch[feature] for eopatch in array])
    if not feature[0].is_timeless():
        p, t, w, h, f = tmp_data.shape
        return np.moveaxis(tmp_data, 1, 3).reshape(p * w * h, t * f)

    return tmp_data.flatten()


def prepare_data(
    train_eopatches: List[EOPatch],
    test_eopatches: List[EOPatch],
    features: Tuple[FeatureType, str],
    is_data: Tuple[FeatureType, str],
    labels: Tuple[FeatureType, str],
) -> Dict[str, np.ndarray]:
    """
    Function prepare data from eopatches to correct np.array
    """
    mask_train = prepare_data_helper(train_eopatches, is_data)
    mask_test = prepare_data_helper(test_eopatches, is_data)

    return {
        "features train": prepare_data_helper(train_eopatches, features)[mask_train],
        "labels train": prepare_data_helper(train_eopatches, labels)[mask_train],
        "features test": prepare_data_helper(test_eopatches, features)[mask_test],
        "labels test": prepare_data_helper(test_eopatches, labels)[mask_test],
    }


def parse_subtree(node: dict, brackets: bool = True) -> str:
    if "leaf_index" in node:
        score = float(node["leaf_value"])
        if PRECISION_SCORES is not None:
            score = round(score, PRECISION_SCORES)
        return f"{score}"

    feature = MODEL_INPUTS[int(node["split_feature"])]

    threshold = float(node["threshold"])
    if PRECISION_THRESHOLD is not None:
        threshold = round(threshold, PRECISION_THRESHOLD)

    condition = f'{feature}{node["decision_type"]}{threshold}'

    left = parse_subtree(node["left_child"])
    right = parse_subtree(node["right_child"])

    result = f"({condition})?{left}:{right}"
    if brackets:
        return f"({result})"
    return result


def parse_one_tree(root: dict, index: int) -> str:
    """
    Parse one tree.
    """
    return f"""
function pt{index}({MODEL_INPUTS_STR}) {{
   return {parse_subtree(root, brackets=False)};
}}
"""


def parse_trees(trees: List) -> str:
    """
    Parse trees into eval script.
    """
    tree_functions = "\n".join(parse_one_tree(tree["tree_structure"], idx) for idx, tree in enumerate(trees))
    function_sum = "+".join(f"pt{i}({MODEL_INPUTS_STR})" for i in range(len(trees)))

    return f"""
//VERSION=3

function setup() {{
    return {{
        input: [{{
            bands: [{','.join(f'"{band}"' for band in BANDS)}],
            units: "reflectance"
        }}],
        output: {{
            id:"default",
            bands: 1,
            sampleType: "FLOAT32"
        }}
    }}
}}

function evaluatePixel(sample) {{
    let NDWI = index(sample.B03, sample.B08);
    let NDMI = index(sample.B08, sample.B11);

    return [predict(sample.B02, sample.B03, sample.B04, NDWI, NDMI)]
}}

{tree_functions}

function predict({MODEL_INPUTS_STR}) {{
    return [1/(1+Math.exp(-1*({function_sum})))];
}}
"""


def parse_model(model: Any, js_output_filename: Optional[str] = None) -> str:
    """
    Parse model for eval script.
    """
    model_json = model.booster_.dump_model()
    model_javascript = parse_trees(model_json["tree_info"])

    if js_output_filename:
        with open(js_output_filename, "w") as f:
            f.write(model_javascript)

    return model_javascript


def get_model_predictions(patch: EOPatch, model: Any) -> np.ndarray:
    """
    Applies model to the features of the EOPatch.
    """
    features = patch.data["FEATURES"][0]
    height, width, nfeats = features.shape
    reshaped_features = features.reshape(height * width, nfeats)
    return model.predict_proba(reshaped_features)[..., -1].reshape(height, width)


def get_sh_predictions(patch: EOPatch, model: Any, config: SHConfig) -> np.ndarray:
    """
    Gets model results from sh-py
    """
    model_script = parse_model(model, None)

    request = SentinelHubRequest(
        evalscript=model_script,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(
                    patch.timestamp[0] - dt.timedelta(minutes=5),
                    patch.timestamp[0] + dt.timedelta(minutes=5),
                ),
                maxcc=1,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=patch.bbox,
        size=(64, 64),
        config=config,
    )
    return request.get_data()[0]


def print_results(
    f1_scores: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    predict_labels_test: np.ndarray,
    labels_test: np.ndarray,
) -> None:
    """
    Prints the statistics of model performance.
    """
    class_names = ["non-water", "water"]
    print(f"Classification accuracy {100 * metrics.accuracy_score(labels_test, predict_labels_test):.1f}%")
    print(
        "Classification F1-score"
        f" {100 * metrics.f1_score(labels_test, predict_labels_test, average='weighted'):.1f}% \n"
    )
    print("    Class    =  F1  | Recall | Precision")
    print("----------------------------------------")
    for idx, classname in enumerate(class_names):
        print(
            f"• {classname:10s} = {f1_scores[idx] * 100:.1f} |  {recall[idx] * 100:.1f}  | {precision[idx] * 100:.1f}"
        )


def plot_comparison(
    patch: EOPatch,
    sh_prediction: np.ndarray,
    model_prediction: np.ndarray,
    threshold: float = 0.5,
    factor: float = 3.15,
) -> None:
    """
    Plots a comparison of using the model locally and using it in an evalscript.
    """

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), sharex=True, sharey=True)

    for axx in ax.flatten():
        axx.set_xticks([])
        axx.set_yticks([])

    rgb_image = factor * patch.data["BANDS-S2-L1C"][0][..., [3, 2, 1]]

    ax[0][0].imshow(rgb_image)
    ax[0][0].set_title("RGB image", fontsize=14)

    ax[0][1].imshow(sh_prediction, vmin=0, vmax=1)
    ax[0][1].set_title("[a] water prediction probabilities with evalscript on SH", fontsize=14)

    ax[0][2].imshow(model_prediction, vmin=0, vmax=1)
    ax[0][2].set_title("[b] water prediction probabilities with model, locally", fontsize=14)

    ax[0][3].imshow(sh_prediction - model_prediction, vmin=-0.2, vmax=0.2, cmap="RdBu")
    ax[0][3].set_title("differences between [a] and [b]", fontsize=14)

    sh_thr = np.where(sh_prediction > threshold, 1, 0)

    ax[1][0].imshow(rgb_image)
    ax[1][0].set_title("RGB image", fontsize=14)

    ax[1][1].imshow(sh_thr, vmin=0, vmax=1)
    ax[1][1].set_title("[c] water prediction with evalscript on SH", fontsize=14)

    model_thr = np.where(model_prediction > threshold, 1, 0)
    ax[1][2].imshow(model_thr, vmin=0, vmax=1)
    ax[1][2].set_title("[d] water prediction with model, locally", fontsize=14)

    ax[1][3].imshow(sh_thr - model_thr, vmin=-1, vmax=1, cmap="RdBu")
    ax[1][3].set_title("differences between [c] and [d]", fontsize=14)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)
