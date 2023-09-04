"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from doubly_logistic_approximation import DoublyLogisticApproximationTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "_test_eopatch")
    return EOPatch.load(path, lazy_loading=True)


def test_double_logistic_approximation(example_eopatch):
    data = example_eopatch.data["NDVI"]
    timestamps = example_eopatch.timestamps
    mask = example_eopatch.mask["IS_VALID"]
    indices = list(np.nonzero([t.year == 2016 for t in timestamps])[0])
    start, stop = indices[0], indices[-1] + 2

    eopatch = EOPatch(bbox=BBox((0, 0, 1, 1), CRS(3857)))
    eopatch.timestamps = timestamps[start:stop]
    eopatch.data["TEST"] = np.reshape(data[start:stop, 0, 0, 0], (-1, 1, 1, 1))
    eopatch.mask["IS_VALID"] = np.reshape(mask[start:stop, 0, 0, 0], (-1, 1, 1, 1))
    eopatch = DoublyLogisticApproximationTask(
        feature=(FeatureType.DATA, "TEST"),
        valid_mask=(FeatureType.MASK, "IS_VALID"),
        new_feature=(FeatureType.DATA_TIMELESS, "TEST_OUT"),
    ).execute(eopatch)

    names = "c1", "c2", "a1", "a2", "a3", "a4", "a5"
    values = eopatch.data_timeless["TEST_OUT"].squeeze()
    expected_values = 0.207, 0.464, 0.686, 0.222, 1.204, 0.406, 15.701
    delta = 0.1

    for name, value, expected_value in zip(names, values, expected_values):
        assert value == pytest.approx(expected_value, abs=delta), f"Missmatch in value of {name}"
