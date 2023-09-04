"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np

from eolearn.core import MapFeatureTask
from eolearn.core.types import FeaturesSpecification


class LinearFunctionTask(MapFeatureTask):
    """Applies a linear function to the values of input features.

    Each value in the feature is modified as `x -> x * slope + intercept`. The `dtype` of the result can be customized.
    """

    def __init__(
        self,
        input_features: FeaturesSpecification,
        output_features: FeaturesSpecification | None = None,
        slope: float = 1,
        intercept: float = 0,
        dtype: str | type | np.dtype | None = None,
    ):
        """
        :param input_features: Feature or features on which the function is used.
        :param output_features: Feature or features for saving the result. If not provided the input_features are
            overwritten.
        :param slope: Slope of the function i.e. the multiplication factor.
        :param intercept: Intercept of the function i.e. the value added.
        :param dtype: Numpy dtype of the output feature. If not provided the dtype is determined by Numpy, so it is
            recommended to set manually.
        """
        if output_features is None:
            output_features = input_features
        self.dtype = dtype if dtype is None else np.dtype(dtype)

        super().__init__(input_features, output_features, slope=slope, intercept=intercept)

    def map_method(self, feature: np.ndarray, slope: float, intercept: float) -> np.ndarray:  # type:ignore[override]
        """A method where feature is multiplied by a slope"""
        rescaled_feature = feature * slope + intercept
        return rescaled_feature if self.dtype is None else rescaled_feature.astype(self.dtype)
