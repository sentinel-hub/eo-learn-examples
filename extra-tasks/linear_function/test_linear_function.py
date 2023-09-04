"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
from linear_function import LinearFunctionTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS(3857))


def test_linear_function_task():
    eopatch = EOPatch(bbox=DUMMY_BBOX, timestamps=["1994-02-01"] * 8)

    data_feature = (FeatureType.DATA, "DATA_TEST")
    data_result_feature = (FeatureType.DATA, "DATA_TRANSFORMED")
    data_shape = (8, 10, 10, 5)
    eopatch[data_feature] = np.arange(np.prod(data_shape)).reshape(data_shape).astype(np.float32)

    mask_timeless_feature = (FeatureType.MASK_TIMELESS, "MASK_TIMELESS_TEST")
    mask_timeless_result_feature = (FeatureType.MASK_TIMELESS, "MASK_TIMELESS_TRANSFORMED")
    mask_shape = (10, 10, 1)
    eopatch[mask_timeless_feature] = np.ones(mask_shape, dtype=np.uint32)

    task_default = LinearFunctionTask(data_feature, data_result_feature)
    task_default(eopatch)
    assert np.array_equal(eopatch[data_feature], eopatch[data_result_feature])

    task_double_minus_five = LinearFunctionTask(
        [data_feature, mask_timeless_feature],
        [data_result_feature, mask_timeless_result_feature],
        slope=2,
        intercept=-5,
    )
    task_double_minus_five(eopatch)
    expected_result = np.arange(np.prod(data_shape)).reshape(data_shape).astype(float) * 2 - 5
    assert np.array_equal(eopatch[data_result_feature], expected_result)
    assert np.array_equal(eopatch[mask_timeless_result_feature], np.ones(mask_shape) * 2 - 5)
    assert eopatch[data_result_feature].dtype == np.float32
    # The value of the mask timeless changes here

    task_change_dtype = LinearFunctionTask(
        mask_timeless_feature, mask_timeless_result_feature, slope=256, dtype=np.uint8
    )
    task_change_dtype(eopatch)
    assert np.array_equal(eopatch[mask_timeless_result_feature], np.zeros(mask_shape))
    assert eopatch[mask_timeless_result_feature].dtype == np.uint8

    task_override = LinearFunctionTask(
        [data_feature, mask_timeless_feature],
        slope=5,
    )
    task_override(eopatch)
    assert np.array_equal(eopatch[mask_timeless_feature], np.ones(mask_shape) * 5)
