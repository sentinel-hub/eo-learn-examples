"""
Copyright (c) 2017- Sinergise and contributors.
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import skimage.segmentation
from superpixel import SuperpixelSegmentationTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub.testing_utils import assert_statistics_match

SUPERPIXEL_FEATURE = FeatureType.MASK_TIMELESS, "SP_FEATURE"


@pytest.fixture(name="test_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "_test_eopatch")
    return EOPatch.load(path, lazy_loading=True)


@pytest.mark.parametrize(
    ("task", "expected_statistics"),
    [
        (
            SuperpixelSegmentationTask(
                (FeatureType.DATA, "BANDS-S2-L1C"), SUPERPIXEL_FEATURE, scale=100, sigma=0.5, min_size=100
            ),
            {"exp_dtype": np.int64, "exp_min": 0, "exp_max": 25, "exp_mean": 10.6809, "exp_median": 11},
        ),
        (
            SuperpixelSegmentationTask(
                (FeatureType.DATA_TIMELESS, "MAX_NDVI"), SUPERPIXEL_FEATURE, scale=21, sigma=1.0, min_size=52
            ),
            {"exp_dtype": np.int64, "exp_min": 0, "exp_max": 22, "exp_mean": 8.5302, "exp_median": 7},
        ),
        (
            SuperpixelSegmentationTask((FeatureType.MASK, "CLM"), SUPERPIXEL_FEATURE, scale=1, sigma=0, min_size=15),
            {"exp_dtype": np.int64, "exp_min": 0, "exp_max": 171, "exp_mean": 86.46267, "exp_median": 90},
        ),
        (
            SuperpixelSegmentationTask(
                (FeatureType.DATA, "CLP"),
                SUPERPIXEL_FEATURE,
                segmentation_object=skimage.segmentation.slic,
                start_label=0,
                n_segments=55,
                compactness=25.0,
                max_num_iter=20,
                sigma=0.8,
            ),
            {"exp_dtype": np.int64, "exp_min": 0, "exp_max": 48, "exp_mean": 24.6072, "exp_median": 25},
        ),
        (
            SuperpixelSegmentationTask(
                (FeatureType.MASK_TIMELESS, "RANDOM_UINT8"),
                SUPERPIXEL_FEATURE,
                segmentation_object=skimage.segmentation.slic,
                start_label=0,
                n_segments=231,
                compactness=15.0,
                max_num_iter=7,
                sigma=0.2,
            ),
            {"exp_dtype": np.int64, "exp_min": 0, "exp_max": 195, "exp_mean": 100.1844, "exp_median": 101},
        ),
    ],
)
def test_superpixel(test_eopatch, task, expected_statistics):
    task.execute(test_eopatch)
    result = test_eopatch[SUPERPIXEL_FEATURE]

    assert_statistics_match(result, **expected_statistics, abs_delta=1e-4)
