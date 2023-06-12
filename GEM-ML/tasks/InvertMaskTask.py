from typing import Tuple, Optional

import numpy as np
from eolearn.core import EOTask, FeatureType, EOPatch


class InvertMaskTask(EOTask):
    def __init__(self, feature_in: Tuple[FeatureType, str], feature_out: Optional[Tuple[FeatureType, str]] = None):
        """
        Task to invert a mask feature.
        :param feature_in: Mask feature to invert
        :param feature_out: Output mask feature
        """
        feature_out = feature_out or feature_in
        if isinstance(feature_in, list):
            self.feature_in = feature_in
            self.feature_out = feature_out
        else:
            self.feature_in = [feature_in]
            self.feature_out = [feature_out]

    def execute(self, eopatch: EOPatch) -> EOPatch:
        for feature_in_, feature_out_ in zip(self.feature_in, self.feature_out):
            eopatch[feature_out_] = (eopatch[feature_in_] < 1).astype(np.uint8)
        return eopatch
