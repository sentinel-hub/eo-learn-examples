"""
A collection of bands extraction EOTasks

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np

from eolearn.core import MapFeatureTask
from eolearn.core.types import Feature


class EuclideanNormTask(MapFeatureTask):
    """The task calculates the Euclidean Norm:

        :math:`Norm = \\sqrt{\\sum_{i} B_i^2}`

    where :math:`B_i` are the individual bands within a user-specified feature array.
    """

    def __init__(
        self,
        input_feature: Feature,
        output_feature: Feature,
        bands: list[int] | None = None,
    ):
        """
        :param input_feature: A source feature from which to take the subset of bands.
        :param output_feature: An output feature to which to write the euclidean norm.
        :param bands: A list of bands from which to extract the euclidean norm. If None, all bands are taken.
        """
        super().__init__(input_feature, output_feature)
        self.bands = bands

    def map_method(self, feature: np.ndarray) -> np.ndarray:
        """
        :param feature: An eopatch on which to calculate the euclidean norm.
        """
        array = feature if not self.bands else feature[..., self.bands]
        return np.sqrt(np.sum(array**2, axis=-1))[..., np.newaxis]
