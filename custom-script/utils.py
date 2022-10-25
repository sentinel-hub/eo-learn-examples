import datetime as dt
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from eolearn.core import FeatureType
from eolearn.core.eodata import EOPatch
from sentinelhub import BBox, DataCollection, MimeType, SentinelHubRequest, SHConfig

PRECISION_SCORES = 4
PRECISION_THRESHOLD = None

BANDS = ["B02", "B03", "B04", "B08", "B11"]
BANDS_STR = ",".join(BANDS)
MODEL_INPUTS = ["B02", "B03", "B04", "NDWI", "NDMI"]
MODEL_INPUTS_STR = ", ".join(MODEL_INPUTS)

FEATURES_SAMPLED = FeatureType.DATA, "FEATURES_SAMPLED"
IS_DATA_SAMPLED = FeatureType.MASK, "IS_DATA_SAMPLED"
LABELS_SAMPLED = FeatureType.MASK_TIMELESS, "water_label_SAMPLED"


def prepare_data(eopatches: np.ndarray, train_eopatches: np.ndarray):
    # Set the features and the labels for train and test sets
    features_train = np.array([eopatch[FEATURES_SAMPLED] for eopatch in eopatches[train_eopatches]])
    labels_train = np.array([eopatch[LABELS_SAMPLED] for eopatch in eopatches[train_eopatches]])
    mask_train = np.array([eopatch[IS_DATA_SAMPLED] for eopatch in eopatches[train_eopatches]])

    features_test = np.array([eopatch[FEATURES_SAMPLED] for eopatch in eopatches[~train_eopatches]])
    labels_test = np.array([eopatch[LABELS_SAMPLED] for eopatch in eopatches[~train_eopatches]])
    mask_test = np.array([eopatch[IS_DATA_SAMPLED] for eopatch in eopatches[~train_eopatches]])

    # get shape
    p1, t, w, h, f = features_train.shape
    p2, t, w, h, f = features_test.shape

    # reshape to n x m
    features_train = np.moveaxis(features_train, 1, 3).reshape(p1 * w * h, t * f)
    labels_train = np.moveaxis(labels_train, 1, 2).reshape(p1 * w * h, 1).squeeze()
    mask_train = np.moveaxis(mask_train, 1, 2).reshape(p1 * w * h, 1).squeeze()

    features_test = np.moveaxis(features_test, 1, 3).reshape(p2 * w * h, t * f)
    labels_test = np.moveaxis(labels_test, 1, 2).reshape(p2 * w * h, 1).squeeze()
    mask_test = np.moveaxis(mask_test, 1, 2).reshape(p2 * w * h, 1).squeeze()

    # remove points with no valid data
    return features_train[mask_train], labels_train[mask_train], features_test[mask_test], labels_test[mask_test]


def parse_subtree(node: Dict, brackets: bool = True):
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


def parse_one_tree(root: Dict, index: int):
    return f"""
function pt{index}({MODEL_INPUTS_STR}) {{
   return {parse_subtree(root, brackets=False)};
}}
"""


def parse_trees(trees: List) -> str:
    tree_functions = "\n".join([parse_one_tree(tree["tree_structure"], idx) for idx, tree in enumerate(trees)])
    function_sum = "+".join([f"pt{i}({MODEL_INPUTS_STR})" for i in range(len(trees))])

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


def parse_model(model: Any, js_output_filename: str = None) -> str:
    model_json = model.booster_.dump_model()
    model_javascript = parse_trees(model_json["tree_info"])

    if js_output_filename:
        with open(js_output_filename, "w") as f:
            f.write(model_javascript)

    return model_javascript


def predict_on_sh(model_script: str, bbox: BBox, size: Tuple[int], timestamp: list, config: SHConfig) -> np.ndarray:
    request = SentinelHubRequest(
        evalscript=model_script,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(timestamp - dt.timedelta(minutes=5), timestamp + dt.timedelta(minutes=5)),
                maxcc=1,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
    )
    return request.get_data()[0]


def get_model_predictions(patch: EOPatch, model: Any) -> np.ndarray:
    features = patch.data["FEATURES"][0]
    height, width, nfeats = features.shape
    reshaped_features = features.reshape(height * width, nfeats)
    return model.predict_proba(reshaped_features)[..., -1].reshape(height, width)


def get_sh_predictions(patch: EOPatch, model: Any, config: SHConfig) -> np.ndarray:
    model_script = parse_model(model, None)
    return predict_on_sh(model_script, patch.bbox, (64, 64), patch.timestamp[0], config)


def print_results(
    f1_scores: np.ndarray,
    recall: np.ndarray,
    precision: np.ndarray,
    predict_labels_test: np.ndarray,
    labels_test: np.ndarray,
):
    class_names = ["non-water", "water"]

    print("Classification accuracy {:.1f}%".format(100 * metrics.accuracy_score(labels_test, predict_labels_test)))
    print(
        "Classification F1-score {:.1f}% \n".format(
            100 * metrics.f1_score(labels_test, predict_labels_test, average="weighted")
        )
    )
    print("             Class              =  F1  | Recall | Precision")
    print("         --------------------------------------------------")
    for idx, classname in enumerate(class_names):
        print(
            "         * {0:20s} = {1:2.1f} |  {2:2.1f}  | {3:2.1f}".format(
                classname, f1_scores[idx] * 100, recall[idx] * 100, precision[idx] * 100
            )
        )


def plot_comparison(
    patch: EOPatch,
    sh_prediction: np.ndarray,
    model_prediction: np.ndarray,
    threshold: float = 0.5,
    factor: float = 3.15,
):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 12), sharex=True, sharey=True)

    for axx in ax.flatten():
        axx.set_xticks([])
        axx.set_yticks([])

    rgb_image = factor * patch.data["BANDS-S2-L1C"][0][..., [3, 2, 1]].squeeze()

    ax[0][0].imshow(rgb_image)
    ax[0][0].set_title("RGB image")

    ax[0][1].imshow(sh_prediction, vmin=0, vmax=1)
    ax[0][1].set_title("[a] water prediction probabilities with evalscript on SH", fontsize=14)

    ax[0][2].imshow(model_prediction, vmin=0, vmax=1)
    ax[0][2].set_title("[b] water prediction probabilities with model, locally", fontsize=14)

    ax[0][3].imshow(sh_prediction - model_prediction, vmin=-0.2, vmax=0.2, cmap="RdBu")
    ax[0][3].set_title("differences between [a] and [b]", fontsize=14)

    sh_thr = np.where(sh_prediction > threshold, 1, 0)

    ax[1][0].imshow(rgb_image)
    ax[1][0].set_title("RGB image")

    ax[1][1].imshow(sh_thr, vmin=0, vmax=1)
    ax[1][1].set_title("[c] water prediction with evalscript on SH", fontsize=14)

    model_thr = np.where(model_prediction > threshold, 1, 0)
    ax[1][2].imshow(model_thr, vmin=0, vmax=1)
    ax[1][2].set_title("[d] water prediction with model, locally", fontsize=14)

    ax[1][3].imshow(sh_thr - model_thr, vmin=-1, vmax=1, cmap="RdBu")
    ax[1][3].set_title("differences between [c] and [d]", fontsize=14)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=2.0)


def plot_eopatch(patch: EOPatch, factor: float = 3.15):
    fig, ax = plt.subplots(ncols=3, figsize=(22, 7))
    ax[0].imshow(factor * patch.data["BANDS-S2-L1C"][0][..., [3, 2, 1]].squeeze())
    ax[0].set_title(f"True color, {patch.timestamp[0]}")

    ax[1].imshow(patch.data_timeless["DEM"].squeeze())
    ax[1].set_title("DEM")

    ax[2].imshow(patch.mask_timeless["water_label"].squeeze(), vmin=0, vmax=1)
    ax[2].set_title("water mask")


def plot_miss_prediction(patch: EOPatch, model_prediction: np.ndarray, threshold: float = 0.5):
    fig, ax = plt.subplots(ncols=3, figsize=(22, 7))
    mask = patch.mask_timeless["water_label"].squeeze()
    ax[0].imshow(mask, vmin=0, vmax=1)
    ax[0].set_title("water mask")

    model_thr = np.where(model_prediction > threshold, 1, 0)
    ax[1].imshow(model_thr, vmin=0, vmax=1)
    ax[1].set_title("model")

    ax[2].imshow(mask - model_thr, vmin=-1, vmax=1, cmap="RdBu")
    ax[2].set_title("miss prediction")
