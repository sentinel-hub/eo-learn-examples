"""
A collection of bands extraction EOTasks

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import numpy as np
from euclidean_norm import EuclideanNormTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

INPUT_FEATURE = (FeatureType.DATA, "TEST")


def test_euclidean_norm():
    eopatch = EOPatch(bbox=BBox((0, 0, 1, 1), CRS(3857)), timestamps=["1234-05-06"] * 5)

    data = np.zeros(5 * 10 * 10 * 7).reshape(5, 10, 10, 7)
    bands = [0, 1, 2, 4, 6]
    data[..., bands] = 1

    eopatch[INPUT_FEATURE] = data

    eopatch = EuclideanNormTask(INPUT_FEATURE, (FeatureType.DATA, "NORM"), bands)(eopatch)
    assert (eopatch.data["NORM"] == np.sqrt(len(bands))).all()
