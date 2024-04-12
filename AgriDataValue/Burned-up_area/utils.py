import datetime
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eolearn.core import EOPatch, EOTask, FeatureType
from sentinelhub import BBox

Feature = Tuple[FeatureType, str]

ID_X_COLUMN, ID_Y_COLUMN = "ID_X", "ID_Y"
ID_COLUMNS = [ID_X_COLUMN, ID_Y_COLUMN]
P_ID_COLUMN = "P_ID"
COORD_X_COLUMN, COORD_Y_COLUMN = "COORD_X", "COORD_Y"
TIMESTAMP_COLUMN = "TIMESTAMP"
CRS_COLUMN = "CRS"


class PatchToDataframeTask(EOTask):
    """Task for converting an EOPatch to a pd.DataFrame

    It transfers all the data from patch features to a dataframe, which is saved as a META_INFO feature.
    """

    def __init__(
        self,
        *,
        append_coords: bool = False,
        append_p_id: bool = False,
        output_feature: Feature,
        shape_info_feature: Feature,
        mask_feature: Optional[Feature],
    ) -> None:
        self.append_coords = append_coords
        self.append_p_id = append_p_id
        self.output_feature = self.parse_feature(output_feature, allowed_feature_types=[FeatureType.META_INFO])
        self.shape_info_feature = self.parse_feature(shape_info_feature, allowed_feature_types=[FeatureType.META_INFO])

        self.mask_feature = None
        if mask_feature is not None:
            self.mask_feature = self.parse_feature(mask_feature, allowed_feature_types=[FeatureType.MASK_TIMELESS])

        self.relevant_feature_types = [
            FeatureType.DATA,
            FeatureType.DATA_TIMELESS,
            FeatureType.MASK,
            FeatureType.MASK_TIMELESS,
        ]

    def get_basic_info(self, eopatch: EOPatch) -> Tuple[BBox, List[datetime.date], Tuple[int, int]]:
        """Extracts the BBox, timestamps, and spatial size of EOPatch."""
        bbox = eopatch.bbox
        timestamps = eopatch.timestamps or []

        some_features = self.parse_features(self.relevant_feature_types, eopatch=eopatch)

        if not some_features:
            ValueError("EOPatch has no relevant features.")
        height, width = eopatch.get_spatial_dimension(*some_features[0])
        return bbox, timestamps, (height, width)

    def prepare_content(
        self,
        bbox: BBox,
        timestamp: List[datetime.date],
        spatial_shape: Tuple[int, int],
        eopatch_id: int,
        mask: np.ndarray,
    ) -> pd.DataFrame:
        """Prepares dataframe with columns containing basic information."""
        height, width = spatial_shape
        time_dim = len(timestamp)

        id_x = np.full(spatial_shape, np.arange(width), dtype=np.uint16)[mask].ravel()
        id_y = np.full(spatial_shape, np.expand_dims(np.arange(height), axis=-1), dtype=np.uint16)[mask].ravel()

        if time_dim == 0:
            dataframe = pd.DataFrame({ID_X_COLUMN: id_x, ID_Y_COLUMN: id_y})
        else:
            ts_array = np.full((time_dim, height, width), np.expand_dims(timestamp, axis=(-2, -1)))[..., mask].ravel()
            dataframe = pd.DataFrame(
                {
                    TIMESTAMP_COLUMN: pd.to_datetime(ts_array),
                    ID_X_COLUMN: np.full((time_dim, len(id_x)), id_x).ravel(),
                    ID_Y_COLUMN: np.full((time_dim, len(id_y)), id_y).ravel(),
                }
            )

        if self.append_p_id:
            dataframe[P_ID_COLUMN] = (
                eopatch_id * height * width + dataframe[ID_Y_COLUMN] * width + dataframe[ID_X_COLUMN]
            ).astype(np.uint64)

        if self.append_coords:
            x1, y1, x2, y2 = bbox.geometry.bounds
            res_x, res_y = (x2 - x1) / width, (y2 - y1) / height

            dataframe[COORD_X_COLUMN] = (
                np.linspace(x1, x2, width, endpoint=False, dtype=np.float32)[dataframe[ID_X_COLUMN]] + res_x / 2
            )
            dataframe[COORD_Y_COLUMN] = (
                np.linspace(y1, y2, height, endpoint=False, dtype=np.float32)[dataframe[ID_Y_COLUMN]] + res_y / 2
            )
            dataframe[CRS_COLUMN] = bbox.crs.epsg
            dataframe[CRS_COLUMN] = dataframe[CRS_COLUMN].astype("category")

        dataframe.reset_index(drop=True, inplace=True)
        return dataframe

    def transfer_features(
        self,
        dataframe: pd.DataFrame,
        features: Iterable[Feature],
        eopatch: EOPatch,
        time_dim: int,
        spatial_shape: Tuple[int, int],
        mask: np.ndarray,
    ) -> None:
        """Transfers features from eopatch to dataframe. Mutates existing structures."""
        for f_type, f_name in features:
            data_shape = eopatch.get_spatial_dimension(f_type, f_name)
            if data_shape != spatial_shape:
                raise ValueError(
                    f"Features have different spatial shapes, {(f_type, f_name)} has {data_shape} but"
                    f" {spatial_shape} was expected."
                )

            data = eopatch[f_type, f_name]
            del eopatch[f_type, f_name]

            if data.shape[-1] != 1:
                raise ValueError(f"Features should have depth of 1, {(f_type, f_name)} has {data.shape[-1]}.")

            data = data[..., mask, 0]

            if time_dim > 0 and f_type.is_timeless():
                data = np.full((time_dim, np.count_nonzero(mask)), data)

            dataframe[f_name] = data.ravel()
            del data

    def execute(self, eopatch: EOPatch, eopatch_global_id: int = 0) -> EOPatch:
        bbox, timestamp, spatial_shape = self.get_basic_info(eopatch)

        mask = np.ones(spatial_shape, dtype=bool)
        features = set(self.parse_features(self.relevant_feature_types, eopatch=eopatch))

        if self.mask_feature is not None:
            mask = eopatch[self.mask_feature].squeeze(axis=-1).astype(bool)
            features -= {self.mask_feature}

        dataframe = self.prepare_content(bbox, timestamp, spatial_shape, eopatch_global_id, mask)
        self.transfer_features(dataframe, features, eopatch, len(timestamp), spatial_shape, mask)

        new_eopatch = EOPatch(bbox=bbox)
        new_eopatch[self.output_feature] = dataframe
        new_eopatch[self.shape_info_feature] = spatial_shape
        return new_eopatch


def plot_data(
    data: np.ndarray,
    timestamp_index: int,
    brightness_factor: float = 1,
    clip=None,
    title="",
    ax=None,
    **kwargs,
):
    image = data[timestamp_index] * brightness_factor

    if clip is not None:
        image = np.clip(image, *clip)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(image, **kwargs)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_mask(mask: np.ndarray, title="", ax=None):
    prepared_mask = mask.squeeze(-1)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(prepared_mask, vmin=0, vmax=1)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def plot_combo(
    data: np.ndarray,
    mask: np.ndarray,
    brightness_factor: float = 1,
    clip=None,
    **kwargs,
):
    _, axs = plt.subplots(ncols=3, figsize=(15, 5))

    plot_data(
        data,
        brightness_factor=brightness_factor,
        timestamp_index=0,
        clip=clip,
        title="Before",
        ax=axs[0],
        **kwargs,
    )
    plot_data(
        data,
        brightness_factor=brightness_factor,
        timestamp_index=1,
        clip=clip,
        title="After",
        ax=axs[1],
        **kwargs,
    )
    plot_mask(mask, title="Burned area mask", ax=axs[2])

    # plot mask
    axs[2].imshow(mask, vmin=0, vmax=1)

    plt.tight_layout()


def plot_band_hist(df, band, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(15, 5))

    ax.hist(df.query("BURN_AREA == 0")[band], **kwargs, label="Not burned", alpha=0.5)
    ax.hist(df.query("BURN_AREA == 1")[band], **kwargs, label="Burned", alpha=0.5)

    ax.set_xlabel(f"Band: {band}")
    ax.grid("on")
    ax.legend()


def convert_eop_to_df(eopatch, timestamp_index=None):
    patch2df_task = PatchToDataframeTask(
        output_feature=(FeatureType.META_INFO, "DF"),
        shape_info_feature=(FeatureType.META_INFO, "SHAPE"),
        mask_feature=None,
    )
    df = patch2df_task(eopatch.copy(deep=True)).meta_info["DF"]
    df.loc[(df.BURN_AREA == 1) & (df.TIMESTAMP == df.TIMESTAMP.min()), "BURN_AREA"] = 0

    if timestamp_index is None:
        return df

    ts_array = pd.to_datetime(eopatch.timestamps)
    df.query(f'TIMESTAMP == "{ts_array[timestamp_index]}"', inplace=True)
    return df


def apply_function_to_eopatch(eopatch, timestamp_index, function, output_mask_name):
    df = convert_eop_to_df(eopatch, timestamp_index)
    mask_timeless_shape = eopatch.mask_timeless["BURN_AREA"].shape
    values = df.apply(function, axis=1).values
    eopatch.mask_timeless[output_mask_name] = values.reshape(mask_timeless_shape)
    return eopatch


def apply_decision_trees_to_eopatch(eopatch, timestamp_index, classifier, output_mask_name, features):
    df = convert_eop_to_df(eopatch, timestamp_index)
    mask_timeless_shape = eopatch.mask_timeless["BURN_AREA"].shape
    values = classifier.predict(df[features].values)
    eopatch.mask_timeless[output_mask_name] = values.reshape(mask_timeless_shape)
    return eopatch


def apply_lgbm_to_eopatch(eopatch, timestamp_index, classifier, output_mask_name):
    df = convert_eop_to_df(eopatch, timestamp_index)
    mask_timeless_shape = eopatch.mask_timeless["BURN_AREA"].shape
    mask = classifier.predict(df[classifier.feature_name_].values)
    proba = classifier.predict_proba(df[classifier.feature_name_].values)[:, -1]
    eopatch.mask_timeless[output_mask_name] = mask.reshape(mask_timeless_shape)
    eopatch.data_timeless[f"{output_mask_name}_PROBA"] = proba.reshape(mask_timeless_shape)
    return eopatch
