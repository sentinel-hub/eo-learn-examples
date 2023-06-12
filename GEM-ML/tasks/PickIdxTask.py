from typing import Tuple, Iterable, Union, Optional

from eolearn.core import EOTask, FeatureType, EOPatch


class PickIdxTask(EOTask):
    def __init__(self,
                 in_feature: Tuple[FeatureType, str],
                 out_feature: Optional[Tuple[FeatureType, str]] = None,
                 idx: Union[int, Iterable[int]]=[-1], *args, **kwargs):
        """
        This class enables the user to pick a specific index (e.g. the most recent observation) of a given feature.
        Please have the indexing rules of numpy in mind!
        For example, for keeping dimensions and slicing the last dimension, you have to insert integers in brackets: idx=[[0],...,slice(start,stop(,step))].
        :param in_feature: Feature to pick the index from.
        :param out_feature: Feature to store the picked index in.
        :param idx: Index to pick.
        :param args: Further args for super.
        :param kwargs: Further kwargs for super.
        """
        super().__init__(*args, **kwargs)

        self.in_feature = in_feature
        if out_feature == None:
            self.out_feature = self.in_feature
        else:
            self.out_feature = out_feature

        self.idx = tuple(idx) # take care of slices!!!
        pass

    def execute(self, eopatch: EOPatch, *args, **kwargs) -> EOPatch:
        """
        Execute method that computes the TDigest of the chosen features.

        :param eopatch: EOPatch which will be saved if suitable
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        """

        eopatch[self.out_feature] = eopatch[self.in_feature].__getitem__(self.idx)
        return eopatch