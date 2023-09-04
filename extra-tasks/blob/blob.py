"""
Module for computing blobs in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import itertools as it
from math import sqrt
from typing import Any, Callable

import numpy as np

from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.types import SingleFeatureSpec


class BlobTask(EOTask):
    """
    Task to compute blobs

    A blob is a region of an image in which some properties are constant or approximately constant; all the points in a
    blob can be considered in some sense to be similar to each other.

    3 methods are implemented: The Laplacian of Gaussian (LoG), the difference of Gaussian approach (DoG) and the
    determinant of the Hessian (DoH).

    The output is a `FeatureType.DATA` where the radius of each blob is stored in his center.
    ie : If blob[date, i, j, 0] = 5 then a blob of radius 5 is present at the coordinate (i, j)

    The task uses `skimage.feature.blob_log`, `skimage.feature.blob_dog` or `skimage.feature.blob_doh` for extraction.

    The input image must be in [-1,1] range.

    :param feature: A feature that will be used and a new feature name where data will be saved, e.g.
        `(FeatureType.DATA, 'bands', 'blob')`.
    :param blob_object: Callable that calculates the blob
    :param blob_parameters: Parameters to be passed to the blob function. Consult documentation of `blob_object`
        for available parameters.
    """

    def __init__(self, feature: SingleFeatureSpec, blob_object: Callable, **blob_parameters: Any):
        self.feature_parser = self.get_feature_parser(feature, allowed_feature_types=[FeatureType.DATA])

        self.blob_object = blob_object
        self.blob_parameters = blob_parameters

    def _compute_blob(self, data: np.ndarray) -> np.ndarray:
        result = np.zeros(data.shape, dtype=np.float32)
        num_time, _, _, num_bands = data.shape
        for time_idx, band_idx in it.product(range(num_time), range(num_bands)):
            image = data[time_idx, :, :, band_idx]
            blob = self.blob_object(image, **self.blob_parameters)
            x_coord = blob[:, 0].astype(int)
            y_coord = blob[:, 1].astype(int)
            radius = blob[:, 2] * sqrt(2)
            result[time_idx, x_coord, y_coord, band_idx] = radius
        return result

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """Execute computation of blobs on input eopatch

        :param eopatch: Input eopatch
        :return: EOPatch instance with new key holding the blob image.
        """
        for ftype, fname, new_fname in self.feature_parser.get_renamed_features(eopatch):
            eopatch[ftype, new_fname] = self._compute_blob(eopatch[ftype, fname].astype(np.float64))

        return eopatch
