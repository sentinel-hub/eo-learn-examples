from eolearn.core import EOTask
class PickIdxTask(EOTask):
    def __init__(self, in_feature, out_feature=None, idx=[-1], *args, **kwargs):
        """
        This class enables the user to pick a specific index (e.g. the most recent observation) of a given feature.
        Please have the indexing rules of numpy in mind!
        For example, for keeping dimensions and slicing the last dimension, you have to insert integers in brackets: idx=[[0],...,slice(start,stop(,step))].
        """
        super().__init__(*args, **kwargs)

        self.in_feature = in_feature
        if out_feature == None:
            self.out_feature = self.in_feature
        else:
            self.out_feature = out_feature

        self.idx = tuple(idx) # take care of slices!!!
        pass

    def execute(self, eopatch, *args, **kwargs):
        """
        Execute method that computes the TDigest of the chosen features.

        :param eopatch: EOPatch which will be saved if suitable
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        """

        eopatch[self.out_feature] = eopatch[self.in_feature].__getitem__(
            self.idx)
        return eopatch