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
from haralick import HaralickTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub.testing_utils import assert_statistics_match

FEATURE = (FeatureType.DATA, "NDVI", "haralick")
OUTPUT_FEATURE = (FeatureType.DATA, "haralick")


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
            HaralickTask(FEATURE, texture_feature="contrast", angle=0, levels=255, window_size=3),
            {"exp_min": 3.5, "exp_max": 9079.0, "exp_mean": 965.8295, "exp_median": 628.5833},
        ),
        (
            HaralickTask(FEATURE, texture_feature="sum_of_square_variance", angle=np.pi / 2, levels=8, window_size=5),
            {"exp_min": 0.96899, "exp_max": 48.7815, "exp_mean": 23.0229, "exp_median": 23.8987},
        ),
        (
            HaralickTask(FEATURE, texture_feature="sum_entropy", angle=-np.pi / 2, levels=8, window_size=7),
            {"exp_min": 0, "exp_max": 1.7463, "exp_mean": 0.5657, "exp_median": 0.50558},
        ),
        (
            HaralickTask(FEATURE, texture_feature="difference_variance", angle=-np.pi / 2, levels=8, window_size=7),
            {"exp_min": 42, "exp_max": 110.6122, "exp_mean": 53.857082, "exp_median": 50},
        ),
    ],
)
def test_haralick(example_eopatch, task, expected_statistics):
    eopatch = copy.deepcopy(example_eopatch)
    task.execute(eopatch)

    assert_statistics_match(eopatch[OUTPUT_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, abs_delta=1e-4)

    del eopatch[OUTPUT_FEATURE]
    assert example_eopatch == eopatch, "Other features of the EOPatch were affected."
