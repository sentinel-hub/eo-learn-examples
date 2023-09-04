"""
Module for computing blobs in EOPatch

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from blob import BlobTask
from skimage.feature import blob_dog, blob_doh, blob_log

from eolearn.core import EOPatch, FeatureType
from sentinelhub.testing_utils import assert_statistics_match

FEATURE = (FeatureType.DATA, "NDVI", "blob")
BLOB_FEATURE = (FeatureType.DATA, "blob")


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "_test_eopatch")
    example_eopatch = EOPatch.load(path, lazy_loading=True)

    ndvi = example_eopatch.data["NDVI"][:, :20, :20]
    ndvi[np.isnan(ndvi)] = 0
    example_eopatch.data["NDVI"] = ndvi
    example_eopatch.consolidate_timestamps(example_eopatch.get_timestamps()[:10])
    return example_eopatch


BLOB_TESTS = [
    (
        BlobTask(FEATURE, blob_dog, threshold=0, max_sigma=30),
        {"exp_min": 0.0, "exp_max": 37.9625, "exp_mean": 0.08545, "exp_median": 0.0},
    ),
    (
        BlobTask(FEATURE, blob_doh, num_sigma=5, threshold=0),
        {"exp_min": 0.0, "exp_max": 21.9203, "exp_mean": 0.05807, "exp_median": 0.0},
    ),
    (
        BlobTask(FEATURE, blob_log, log_scale=True, threshold=0, max_sigma=30),
        {"exp_min": 0, "exp_max": 42.4264, "exp_mean": 0.09767, "exp_median": 0.0},
    ),
]


@pytest.mark.parametrize(("task", "expected_statistics"), BLOB_TESTS)
def test_blob_task(example_eopatch, task, expected_statistics):
    task.execute(example_eopatch)

    assert_statistics_match(
        example_eopatch[BLOB_FEATURE], exp_shape=(10, 20, 20, 1), **expected_statistics, abs_delta=1e-4
    )
