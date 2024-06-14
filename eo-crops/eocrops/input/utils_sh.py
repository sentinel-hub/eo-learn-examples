import eolearn
from eolearn.core import EOTask, FeatureType
from sentinelhub import SHConfig
import numpy as np
from eolearn.core import AddFeatureTask, RemoveFeatureTask

def config_sentinelhub_cred(api, client_id, client_secret):
    """
    Configure properly Sentinelhub credential fetching information from the configuration tool python class.
    :return:
    SHConfig
    """

    config = SHConfig()

    if client_id and client_secret and api:
        config.instance_id = api
        config.sh_client_id = client_id
        config.sh_client_secret = client_secret

    return config


def calculate_valid_data_mask(eopatch) :
    ''' Define which pixel will be masked from eo-patch
    Inputs :
        - eopatch : patch downloaded from eo-learn packages (eopatch object with arrays)
    '''
    return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                          np.logical_not(eopatch.mask['CLM'].astype(np.bool)))


def calculate_coverage(array) :
    ''' Share of pixels not contaminated by clouds
    Inputs :
        - array : array from an eopatch object that contains information about cloud coverage (e.g Sentinel-2 L1C)
    '''
    return 1.0-np.count_nonzero(array)/np.size(array)


class AddValidDataCoverage(EOTask) :
    ''' Share of pixels not contaminated by clouds
    Inputs :
        - EOTask : workflow as EOTask object
    '''

    def execute(self, eopatch) :
        valid_data = eopatch[eolearn.core.FeatureType.MASK]['VALID_DATA']
        time, height, width, channels = valid_data.shape

        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height*width*channels)))
        add_coverage = AddFeatureTask((eolearn.core.FeatureType.SCALAR, 'COVERAGE'))
        return add_coverage.execute(eopatch=eopatch, data=coverage[:, np.newaxis])


class AddValidDataMaskTask(EOTask):
    def execute(self, eopatch):
        eopatch.mask["VALID_DATA"] = eopatch.mask["IS_DATA"].astype(bool) & ~(eopatch.mask["CLM"].astype(bool))
        return eopatch


class ValidDataS2(EOTask) :
    """
    The tasks recognize clouds from Sentinel Scene Layers (SCL) obtained from Sen2Corr
    """
    def execute(self, eopatch) :
        add_cloud = AddFeatureTask((eolearn.core.FeatureType.MASK, 'VALID_DATA'))
        return add_cloud.execute(eopatch=eopatch, data=(eopatch.mask['IS_DATA']).astype(bool))

class ValidDataVHRS(EOTask) :
    """
    The tasks recognize clouds from Sentinel Scene Layers (SCL) obtained from Sen2Corr
    """

    def execute(self, eopatch) :
        cloud_mask_ = np.invert(eopatch[FeatureType.MASK]['CLM'].astype(bool))

        add_bool = AddFeatureTask((eolearn.core.FeatureType.MASK, 'IS_DATA'))
        add_bool.execute(eopatch=eopatch, data=np.invert(cloud_mask_))
        add_cloud = AddFeatureTask((eolearn.core.FeatureType.MASK, 'CLM'))
        add_cloud.execute(eopatch=eopatch, data=cloud_mask_)
        add_valid = AddFeatureTask((eolearn.core.FeatureType.MASK, 'VALID_DATA'))
        add_valid.execute(eopatch=eopatch, data=np.invert(cloud_mask_))
        return eopatch

class ValidPixel :
    def __call__(self, eopatch) :
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CountValid(EOTask) :
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name) :
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch) :
        add_count = AddFeatureTask((eolearn.core.FeatureType.MASK_TIMELESS, self.name))
        add_count.execute(eopatch=eopatch, data=np.count_nonzero(eopatch.mask[self.what], axis=0))
        return eopatch


class ValidDataCoveragePredicate :
    '''
    Keep an image only if below % of non contaminated pixels
    Inputs :
        - threshold (float) : upper bound of percentage of pixel predicted as cloudy
    '''

    def __init__(self, threshold) :
        self.threshold = threshold

    def __call__(self, array) :
        return calculate_coverage(array)<self.threshold


class EmptyTask(EOTask) :
    '''
    This task does not make any change. It is just to avoid to duplicate the LinearWorflow with if/else
    For example, saving a EOPatch in the workflow would depend if the user specify a path in the parameters of the function workflow
    '''

    def __init__(self) :
        self.Nothing = 'Nothing'

    def execute(self, eopatch) :
        return eopatch