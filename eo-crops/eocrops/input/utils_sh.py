import eolearn
from eolearn.core import EOTask, FeatureType
from sentinelhub import SHConfig
import numpy as np


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
        valid_data = eopatch.get_feature(eolearn.core.FeatureType.MASK, 'VALID_DATA')
        time, height, width, channels = valid_data.shape

        coverage = np.apply_along_axis(calculate_coverage, 1, valid_data.reshape((time, height*width*channels)))

        eopatch.add_feature(eolearn.core.FeatureType.SCALAR, 'COVERAGE', coverage[:, np.newaxis])

        return eopatch


class SentinelHubValidData :
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The sentinel_hub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __call__(self, eopatch) :
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CloudMaskS2L2A(EOTask) :
    """
    The tasks recognize clouds from Sentinel Scene Layers (SCL) obtained from Sen2Corr
    """
    def execute(self, eopatch) :
        eopatch.add_feature(FeatureType.MASK, "VALID_DATA", (eopatch.mask['IS_DATA']).astype(bool))
        return eopatch



class CloudMaskFromCLM(EOTask) :
    """
    The tasks recognize clouds from Sentinel Scene Layers (SCL) obtained from Sen2Corr
    NDI = (A-B)/(A+B).
    """

    def execute(self, eopatch) :

        CLM = eopatch.get_feature(FeatureType.MASK, 'CLM')
        cloudy_f = list(CLM.flatten())

        def return_na(x) :
            if x in [1] :  # [3, 4] :
                return True
            else :
                return False

        g = np.array(list(map(lambda x : return_na(x), cloudy_f)))
        g = g.reshape(CLM.shape[0], CLM.shape[1], CLM.shape[2])
        eopatch.add_feature(FeatureType.MASK, "IS_DATA", (1-g[..., np.newaxis]).astype(bool))
        eopatch.add_feature(FeatureType.MASK, "CLM", g[..., np.newaxis])
        eopatch.add_feature(FeatureType.MASK, "VALID_DATA", (1-g[..., np.newaxis]).astype(bool))

        return eopatch


class CountValid(EOTask) :
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """

    def __init__(self, count_what, feature_name) :
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch) :
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what], axis=0))

        return eopatch


class ValidDataCoveragePredicate :
    ''' Keep an image only if below % of non contaminated pixels
    Inputs :
        - threshold (float) : upper bound of percentage of pixel predicted as cloudy
    '''

    def __init__(self, threshold) :
        self.threshold = threshold

    def __call__(self, array) :
        return calculate_coverage(array)<self.threshold


class EmptyTask(EOTask) :
    '''This task does not make any change. It is just to avoid to duplicate the LinearWorflow with if/else
    For example, saving a EOPatch in the workflow would depend if the user specify a path in the parameters of the function workflow
    '''

    def __init__(self) :
        self.Nothing = 'Nothing'

    def execute(self, eopatch) :
        return eopatch