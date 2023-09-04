"""
Module for basic feature manipulations, i.e. removing a feature from EOPatch, or removing a slice (time-frame) from
the time-dependent features.

Copyright (c) 2017- Sinergise and contributors
For the full list of contributors, see the CREDITS file in the root directory of this source tree.

This source code is licensed under the MIT license, see the LICENSE file in the root directory of this source tree.
"""
from __future__ import annotations

from typing import Literal, cast

import numpy as np

from eolearn.core import EOPatch, EOTask
from eolearn.core.types import Feature


class ValueFilloutTask(EOTask):
    """Overwrites occurrences of a desired value with their neighbor values in either forward, backward direction or
    both, along an axis.

    Possible fillout operations are 'f' (forward), 'b' (backward) or both, 'fb' or 'bf':

        'f': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> nan, nan, nan, 8, 5, 5, 1, 0, 0, 0

        'b': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, nan, nan

        'fb': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 5, 1, 0, 0, 0

        'bf': nan, nan, nan, 8, 5, nan, 1, 0, nan, nan -> 8, 8, 8, 8, 5, 1, 1, 0, 0, 0
    """

    def __init__(
        self,
        feature: Feature,
        operations: Literal["f", "b", "fb", "bf"] = "fb",
        value: float = np.nan,
        axis: int = 0,
    ):
        """
        :param feature: A feature that must be value-filled.
        :param operations: Fill directions, which should be one of ['f', 'b', 'fb', 'bf'].
        :param value: Which value to fill by its neighbors.
        :param axis: An axis along which to fill values.
        """
        if operations not in ["f", "b", "fb", "bf"]:
            raise ValueError("'operations' parameter should be one of the following options: f, b, fb, bf.")

        self.feature = self.parse_feature(feature)
        self.operations = operations
        self.value = value
        self.axis = axis

    @staticmethod
    def fill(data: np.ndarray, value: float = np.nan, operation: Literal["f", "b"] = "f") -> np.ndarray:
        """Fills occurrences of a desired value in a 2d array with their neighbors in either forward or backward
        direction.

        :param data: A 2d numpy array.
        :param value: Which value to fill by its neighbors.
        :param operation: Fill directions, which should be either 'f' or 'b'.
        :return: Value-filled numpy array.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Wrong data input")

        if operation not in ["f", "b"]:
            raise ValueError("'operation' parameter should either be 'f' (forward) or 'b' (backward)!")

        n_rows, n_frames = data.shape

        value_mask = np.isnan(data) if np.isnan(value) else (data == value)
        init_index = 0 if operation == "f" else (n_frames - 1)

        idx = np.where(value_mask, init_index, np.arange(n_frames))

        if operation == "f":
            idx = np.maximum.accumulate(idx, axis=1)
        else:
            idx = idx[:, ::-1]
            idx = np.minimum.accumulate(idx, axis=1)
            idx = idx[:, ::-1]

        return data[np.arange(n_rows)[:, np.newaxis], idx]

    def execute(self, eopatch: EOPatch) -> EOPatch:
        """
        :param eopatch: Source EOPatch from which to read the feature data.
        :return: An eopatch with the value-filled feature.
        """
        data = eopatch[self.feature]

        value_mask = np.isnan(data) if np.isnan(self.value) else (data == self.value)

        if not value_mask.any():
            return eopatch

        data = np.swapaxes(data, self.axis, -1)
        original_shape = data.shape
        data = data.reshape(np.prod(original_shape[:-1]), original_shape[-1])

        for operation in self.operations:  # iterates over string that represents the operation
            operation = cast(Literal["f", "b"], operation)
            data = self.fill(data, value=self.value, operation=operation)

        data = data.reshape(*original_shape)
        data = np.swapaxes(data, self.axis, -1)

        eopatch[self.feature] = data

        return eopatch
