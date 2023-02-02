import numpy as np
from eolearn.core import FeatureType, EOTask


class NDVITask(EOTask):
    def __init__(self, feature_in, bands, feature_out=(FeatureType.DATA, "NDVI"), remove_feature_in=False):
        """
        Computes the NDVI from a data feature.
        :param feature_in: Feature to compute the NDVI from
        :param bands: array of strings specifying the bands in feature_in
        :param feature_out: Feature where to store computed NDVI
        :param remove_feature_in: Only keep the computed NDVI instead of the whole data or keep input data
        """
        self.feature_in = feature_in
        self.feature_out = feature_out

        self.remove_feature_in = remove_feature_in

        self.band2index = dict(zip(bands, range(len(bands))))

    def execute(self, eopatch):
        feature_in = eopatch[self.feature_in]
        if self.remove_feature_in:
            del eopatch[self.feature_in]
        elif self.feature_in == self.feature_out:
            raise ValueError("feature_in == feature_out, only allowed with remove_feature_in=True")

        b8 = feature_in[:, :, :, self.band2index["B08"]]
        b4 = feature_in[:, :, :, self.band2index["B04"]]
        ndvi = (b8 - b4) / (b8 + b4)
        eopatch[self.feature_out] = ndvi[:, :, :, np.newaxis]

        return eopatch
