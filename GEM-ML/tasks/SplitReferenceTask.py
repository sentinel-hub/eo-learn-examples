from eolearn.core import EOTask, FeatureType


class SplitReferenceTask(EOTask):
    def __init__(self, feature_in, feature_out=None, remove_feature_in=False):
        """
        Splits a time series into two series: an observational time series and a corresponding reference series.
        :param feature_in: Feature containing the whole series
        :param feature_out: Feature to store the split reference series in
        :param remove_feature_in: Whether to keep the observational data or remove it to save space
        """
        feature_out = feature_out or feature_in
        if isinstance(feature_in, list):
            self.feature_in = feature_in
            self.feature_out = feature_out
        else:
            self.feature_in = [feature_in]
            self.feature_out = [feature_out]

        self.remove_feature_in = remove_feature_in

    def execute(self, eopatch, n_reference=1):
        """
        Executes the split.
        :param eopatch: Input EOPatch
        :param n_reference: Number of reference observations to split
        :return: The modified patch
        """
        for feature_in_, feature_out_ in zip(self.feature_in, self.feature_out):
            feature_in_data = eopatch[feature_in_]
            if self.remove_feature_in:
                del eopatch[feature_in_]
            else:
                eopatch[feature_in_] = feature_in_data[:-n_reference, ...]

            eopatch[feature_out_] = feature_in_data[-n_reference:, ...]

        return eopatch
