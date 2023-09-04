"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import pytest

from eolearn.core import EOPatch, FeatureType
from eolearn.features import LocalBinaryPatternTask
from sentinelhub.testing_utils import assert_statistics_match

LBP_FEATURE = (FeatureType.DATA, "NDVI", "lbp")
OUTPUT_FEATURE = (FeatureType.DATA, "lbp")


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "_test_eopatch")
    example_eopatch = EOPatch.load(path, lazy_loading=True)

    ndvi = example_eopatch.data["NDVI"][:, :20, :20]
    ndvi[np.isnan(ndvi)] = 0
    example_eopatch.data["NDVI"] = ndvi
    example_eopatch.consolidate_timestamps(example_eopatch.get_timestamps()[:10])
    return example_eopatch


@pytest.mark.parametrize(
    ("task", "expected_statistics"),
    [
        (
            LocalBinaryPatternTask(LBP_FEATURE, nb_points=24, radius=3),
            {"exp_min": 0.0, "exp_max": 25.0, "exp_mean": 15.8313, "exp_median": 21.0},
        ),
    ],
)
@pytest.mark.filterwarnings("ignore::eolearn.core.exceptions.TemporalDimensionWarning", "ignore::UserWarning")
def test_local_binary_pattern(example_eopatch, task, expected_statistics):
    eopatch = copy.deepcopy(example_eopatch)
    task.execute(eopatch)

    assert_statistics_match(eopatch[OUTPUT_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, abs_delta=1e-4)

    del eopatch[OUTPUT_FEATURE]
    assert example_eopatch == eopatch, "Other features of the EOPatch were affected."
