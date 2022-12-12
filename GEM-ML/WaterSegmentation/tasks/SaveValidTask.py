### Michael Engel ### 2022-04-29 ### SaveValidTask.py ###
from eolearn.core import SaveTask
import numpy as np

class SaveValidTask(SaveTask):
    """
    A class for saving an EOPatch if the chosen feature is True.
    """
    def __init__(self,feature_to_check,path,filesystem=None,config=None,**kwargs):
        """
        :param feature_to_check: Type of EOPatch feature and name of feature to be checked.
        :type feature_to_check: (FeatureType, str)
        
        :param path: root path where all EOPatches are saved
        :type path: str
        :param filesystem: An existing filesystem object. If not given it will be initialized according to the EOPatch
            path. If you intend to run this task in multiprocessing mode you shouldn't specify this parameter.
        :type filesystem: fs.base.FS or None
        :param features: A collection of features types specifying features of which type will be saved. By default
            all features will be saved.
        :type features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param overwrite_permission: A level of permission for overwriting an existing EOPatch
        :type overwrite_permission: OverwritePermission or int
        :param compress_level: A level of data compression and can be specified with an integer from 0 (no compression)
            to 9 (highest compression).
        :type compress_level: int
        :param config: A configuration object with AWS credentials. By default is set to None and in this case the
            default configuration will be taken.
        :type config: SHConfig or None
        :param **kwargs: keyword-arguments for EOPatch.save
        :type **kwargs: dict
        """
        
        super().__init__(path, filesystem=filesystem, config=config, **kwargs)
        
        self.feature_to_check = self.parse_feature(feature_to_check)
        pass
    
    def execute(self, eopatch,*args,eopatch_folder="",**kwargs):
        """
        Execute method that checks feature for suitability and saves EOPatch if True.
        
        :param eopatch: EOPatch which will be saved if suitable
        :type eopatch: EOPatch
        :param eopatch_folder: name of EOPatch folder containing data
        :type eopatch_folder: str
        """
        
        feature_to_check_type, feature_to_check_name = self.feature_to_check
        
        if np.all(eopatch[feature_to_check_type][feature_to_check_name]):
            return super().execute(eopatch,*args,eopatch_folder=eopatch_folder,**kwargs)
        else:
            return eopatch