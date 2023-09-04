"""
Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see https://github.com/sentinel-hub/eo-learn/blob/master/CREDITS.md.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from value_fillout import ValueFilloutTask

from eolearn.core import EOPatch, FeatureType
from sentinelhub import CRS, BBox

DUMMY_BBOX = BBox((0, 0, 1, 1), CRS(3857))


def test_fill():
    array = np.array(
        [
            [np.NaN] * 4 + [1.0, 2.0, 3.0, 4.0] + [np.NaN] * 3,
            [1.0, np.NaN, np.NaN, 2.0, 3.0, np.NaN, np.NaN, np.NaN, 4.0, np.NaN, 5.0],
        ]
    )

    array_ffill = ValueFilloutTask.fill(array, operation="f")
    array_bfill = ValueFilloutTask.fill(array, operation="b")
    array_fbfill = ValueFilloutTask.fill(array_ffill, operation="b")

    assert_allclose(
        array_ffill,
        np.array(
            [
                [np.NaN] * 4 + [1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            ]
        ),
        equal_nan=True,
    )

    assert_allclose(
        array_bfill,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0] + [np.NaN] * 3,
                [1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0],
            ]
        ),
        equal_nan=True,
    )

    assert_allclose(
        array_fbfill,
        np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0],
            ]
        ),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("input_feature", "operation"),
    [((FeatureType.DATA, "TEST"), "x"), ((FeatureType.DATA, "TEST"), 4), (None, "f"), (np.zeros((4, 5)), "x")],
)
def test_bad_input(input_feature, operation):
    with pytest.raises(ValueError):
        ValueFilloutTask(input_feature, operations=operation)


def test_value_fillout():
    feature = (FeatureType.DATA, "TEST")
    shape = (8, 10, 10, 5)
    data = np.random.randint(0, 100, size=shape).astype(float)
    eopatch = EOPatch(bbox=DUMMY_BBOX, timestamps=["2002-11-11"] * 8, data={"TEST": data})

    # nothing to be filled, return the same eopatch object immediately
    eopatch_new = ValueFilloutTask(feature, operations="fb", axis=0)(eopatch)
    assert eopatch == eopatch_new

    eopatch[feature][0, 0, 0, :] = np.nan

    def execute_fillout(eopatch, feature, **kwargs):
        input_array = eopatch[feature]
        eopatch = ValueFilloutTask(feature, **kwargs)(eopatch)
        output_array = eopatch[feature]
        return eopatch, input_array, output_array

    # filling forward temporally should not fill nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="f", axis=0)
    compare_mask = ~np.isnan(input_array)
    assert np.isnan(output_array[0, 0, 0, :]).all()
    assert_array_equal(input_array[compare_mask], output_array[compare_mask])
    assert id(input_array) != id(output_array)

    # filling in any direction along axis=-1 should also not fill nans since all neighbors are nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="fb", axis=-1)
    assert np.isnan(output_array[0, 0, 0, :]).all()
    assert id(input_array) != id(output_array)

    # filling nans backwards temporally should fill nans
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="b", axis=0)
    assert not np.isnan(output_array).any()
    assert id(input_array) != id(output_array)

    # try filling something else than nan (e.g.: -1)
    eopatch[feature][0, :, 0, 0] = -1
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="b", value=-1, axis=-1)
    assert not (output_array == -1).any()
    assert id(input_array) != id(output_array)

    # [nan, 1, nan, 2, ... ]  ---('fb')---> [1, 1, 1, 2, ... ]
    eopatch[feature][0, 0, 0, 0:4] = [np.nan, 1, np.nan, 2]
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="fb", axis=-1)
    assert_array_equal(output_array[0, 0, 0, 0:4], [1, 1, 1, 2])
    assert id(input_array) != id(output_array)

    # [nan, 1, nan, 2, ... ]  ---('bf')---> [1, 1, 2, 2, ... ]
    eopatch[feature][0, 0, 0, 0:4] = [np.nan, 1, np.nan, 2]
    eopatch, input_array, output_array = execute_fillout(eopatch, feature, operations="bf", axis=-1)
    assert_array_equal(output_array[0, 0, 0, 0:4], [1, 1, 2, 2])
    assert id(input_array) != id(output_array)
