"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from hog import HOGTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub.testing_utils import assert_statistics_match


@pytest.fixture(name="example_eopatch")
def example_eopatch_fixture():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "_test_eopatch")
    example_eopatch = EOPatch.load(path, lazy_loading=True)

    ndvi = example_eopatch.data["NDVI"][:, :20, :20]
    ndvi[np.isnan(ndvi)] = 0
    example_eopatch.data["NDVI"] = ndvi
    example_eopatch.consolidate_timestamps(example_eopatch.get_timestamps()[:10])
    return example_eopatch


def test_hog(example_eopatch):
    task = HOGTask(
        (FeatureType.DATA, "NDVI", "hog"),
        orientations=9,
        pixels_per_cell=(2, 2),
        cells_per_block=(2, 2),
        visualize=True,
        visualize_feature_name="hog_visu",
    )

    eopatch = copy.deepcopy(example_eopatch)
    task.execute(eopatch)

    for feature_name, expected_statistics in [
        ("hog", {"exp_min": 0.0, "exp_max": 0.5567, "exp_mean": 0.09309, "exp_median": 0.0}),
        ("hog_visu", {"exp_min": 0.0, "exp_max": 0.3241, "exp_mean": 0.010537, "exp_median": 0.0}),
    ]:
        assert_statistics_match(eopatch.data[feature_name], **expected_statistics, abs_delta=1e-4)

    del eopatch[(FeatureType.DATA, "hog")]
    del eopatch[(FeatureType.DATA, "hog_visu")]
    assert example_eopatch == eopatch, "Other features of the EOPatch were affected."
