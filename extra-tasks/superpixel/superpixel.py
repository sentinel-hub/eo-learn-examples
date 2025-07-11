"""
Module for super-pixel segmentation

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import numpy as np
import skimage.segmentation

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.exceptions import EODeprecationWarning, EORuntimeWarning
from eolearn.core.types import Feature
from sentinelhub.exceptions import deprecated_class

LOGGER = logging.getLogger(__name__)


class SuperpixelSegmentationTask(EOTask):
    """Super-pixel segmentation task

    Given a raster feature it will segment data into super-pixels. Representation of super-pixels will be returned as
    a mask timeless feature where all pixels with the same value belong to one super-pixel

    Examples of `segmentation_object` values:
    - `skimage.segmentation.felzenszwalb` (the defalt)
    - `skimage.segmentation.slic`
    """

    def __init__(
        self,
        feature: Feature,
        superpixel_feature: Feature,
        *,
        segmentation_object: Callable = skimage.segmentation.felzenszwalb,
        **segmentation_params: Any,
    ):
        """
        :param feature: Raster feature which will be used in segmentation
        :param superpixel_feature: A new mask timeless feature to hold super-pixel mask
        :param segmentation_object: A function (object) which performs superpixel segmentation, by default that is
            `skimage.segmentation.felzenszwalb`
        :param segmentation_params: Additional parameters which will be passed to segmentation_object function
        """
        self.feature = self.parse_feature(feature, allowed_feature_types=lambda fty: fty.is_spatial())
        self.superpixel_feature = self.parse_feature(
            superpixel_feature, allowed_feature_types={FeatureType.MASK_TIMELESS}
        )
        self.segmentation_object = segmentation_object
        self.segmentation_params = segmentation_params

    def _create_superpixel_mask(self, data: np.ndarray) -> np.ndarray:
        """Method which performs the segmentation"""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return self.segmentation_object(data, **self.segmentation_params)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Main execute method"""
        data = eopatch[self.feature]

        if np.isnan(data).any():
            warnings.warn(  # noqa: B028
                "There are NaN values in given data, super-pixel segmentation might produce bad results",
                EORuntimeWarning,
            )

        if self.feature[0].is_temporal():
            data = np.moveaxis(data, 0, 2)
            data = data.reshape((data.shape[0], data.shape[1], data.shape[2] * data.shape[3]))

        superpixel_mask = np.atleast_3d(self._create_superpixel_mask(data))

        eopatch[self.superpixel_feature] = superpixel_mask

        return eopatch


@deprecated_class(
    EODeprecationWarning,
    "Use `SuperpixelSegmentationTask` with `segmentation_object=skimage.segmentation.felzenszwalb`.",
)
class FelzenszwalbSegmentationTask(SuperpixelSegmentationTask):
    """Super-pixel segmentation which uses Felzenszwalb's method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb
    """

    def __init__(self, feature: Feature, superpixel_feature: Feature, **kwargs: Any):
        """Arguments are passed to `SuperpixelSegmentationTask` task"""
        super().__init__(feature, superpixel_feature, segmentation_object=skimage.segmentation.felzenszwalb, **kwargs)


@deprecated_class(
    EODeprecationWarning,
    "Use `SuperpixelSegmentationTask` with `segmentation_object=skimage.segmentation.slic` and `start_label=0`.",
)
class SlicSegmentationTask(SuperpixelSegmentationTask):
    """Super-pixel segmentation which uses SLIC method of segmentation

    Uses segmentation function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.slic
    """

    def __init__(self, feature: Feature, superpixel_feature: Feature, **kwargs: Any):
        """Arguments are passed to `SuperpixelSegmentationTask` task"""
        super().__init__(
            feature, superpixel_feature, segmentation_object=skimage.segmentation.slic, start_label=0, **kwargs
        )

    def _create_superpixel_mask(self, data: np.ndarray) -> np.ndarray:
        """Method which performs the segmentation"""
        if np.issubdtype(data.dtype, np.floating) and data.dtype != np.float64:
            data = data.astype(np.float64)
        return super()._create_superpixel_mask(data)


class MarkSegmentationBoundariesTask(EOTask):
    """Takes super-pixel segmentation mask and creates a new mask where boundaries of super-pixels are marked

    The result is a binary mask with values 0 and 1 and dtype `numpy.uint8`

    Uses `mark_boundaries` function documented at:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.mark_boundaries
    """

    def __init__(self, feature: Feature, new_feature: Feature, **params: Any):
        """
        :param feature: Input feature - super-pixel mask
        :param new_feature: Output feature - a new feature where new mask with boundaries will be put
        :param params: Additional parameters which will be passed to `mark_boundaries`. Supported parameters are `mode`
            and `background_label`
        """
        self.feature = self.parse_feature(feature, allowed_feature_types={FeatureType.MASK_TIMELESS})
        self.new_feature = self.parse_feature(new_feature, allowed_feature_types={FeatureType.MASK_TIMELESS})

        self.params = params

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute method"""
        segmentation_mask = eopatch[self.feature][..., 0]

        bounds_mask = skimage.segmentation.mark_boundaries(
            np.zeros(segmentation_mask.shape[:2], dtype=np.uint8), segmentation_mask, **self.params
        )

        bounds_mask = bounds_mask[..., :1].astype(np.uint8)
        eopatch[self.new_feature[0]][self.new_feature[1]] = bounds_mask
        return eopatch
