### Michael Engel ### 2022-10-06 ### parse_time_interval_observations.py ###
from typing import Optional, Tuple, List, Iterable, Callable
from sentinelhub import BBox, DataCollection, SHConfig
from sentinelhub.time_utils import TimeType
from eolearn.io import get_available_timestamps
import numpy as np
import datetime as dt
import pytz

#%% main patched method to plug in IO Tasks for patching
def parse_time_interval_observations(time_interval,
                                     bbox: BBox,
                                     data_collection: DataCollection,
                                     check_timedelta: dt.timedelta = dt.timedelta(days=21),
                                     include_borders: bool = True,
                                     time_difference: dt.timedelta = None,
                                     maxcc: Optional[float] = None,
                                     config: Optional[SHConfig] = None):
    '''
    This method parses a given time_interval:
        - (int,date) -> returns the time interval of the int observations before and including (include_borders==True) or rejecting (include_borders==False) the specified date.
        - [int,date] -> returns the time interval of the int observations before and including (include_borders==True) or rejecting (include_borders==False) the specified date.
        - (date,int) -> returns the time interval of the int observations after and including (include_borders==True) or rejecting (include_borders==False) the specified date.
        - [date,int] -> returns the time interval of the int observations after and including (include_borders==True) or rejecting (include_borders==False) the specified date.
    
    :param time_interval: desired observations based time interval
    :type time_interval: tuple, list or numpy.ndarray
    :param bbox: desired bounding box to check observations for
    :type bbox: sentinelhub-py bounding box (BBox)
    :param data_collection: data collection to check observations for
    :type data_collection: sentinelhub-py data collection (DataCollection)
    :param check_timedelta: chosen time difference to be checked for observations starting from or ending at date given by time_interval
    :type check_timedelta: datetime.timedelta
    :param include_borders: decider whether the specified date is included in the desired time interval or not
    :type include_borders: bool
    :param time_difference: minimum allowed time difference between two observations
    :type time_difference: datetime.timedelta
    :param maxcc: maximum accepted cloud coverage for observations ranging from 0 to 1
    :type maxcc: float
    :param config: Sentinel Hub configuration
    :type config: sentinelhub-py configuration file (SHConfig)
    '''
    
    if (isinstance(time_interval, tuple) or isinstance(time_interval, list) or isinstance(time_interval,np.ndarray)) and len(time_interval)==2:
        time_interval_ = list(time_interval)
        ### (int, date) or [int, date]
        if type(time_interval_[0])==int and type(time_interval_[1])==dt.datetime:
            if time_interval_[1].tzinfo==None:
                time_interval_[1] = time_interval_[1].replace(tzinfo=pytz.utc)
                
            end = time_interval_[1]
            n_obs = -time_interval_[0]
            
            checkstart = end-abs(check_timedelta)
            checkend = end
        ### (date, int) or [date, int]
        elif type(time_interval_[1])==int and type(time_interval_[0])==dt.datetime:
            if time_interval_[0].tzinfo==None:
                time_interval_[0] = time_interval_[0].replace(tzinfo=pytz.utc)
                
            start = time_interval_[0]
            n_obs = time_interval_[1]
            
            checkstart = start
            checkend = start+abs(check_timedelta)
        elif (type(time_interval_[1])==str or type(time_interval_[1])==dt.datetime) and (type(time_interval_[0])==str or type(time_interval_[0])==dt.datetime):
            return time_interval
        else:
            raise ValueError("time_interval must be a list or tuple of length 2 either providing [datetime.datetime,int], [int,datetime.datetime] or [str/datetime.datetime,str/datetime.datetime]!")
    else:
        return time_interval
    
    ### query timestemps in check_interval
    timestamps = get_available_timestamps(
        bbox = bbox, 
        data_collection = data_collection, 
        time_interval = [checkstart,checkend], 
        time_difference = time_difference, 
        maxcc = maxcc, 
        config = config
    )
    timestamps = [ts.replace(tzinfo=pytz.utc) for ts in timestamps]

    ### return
    if len(timestamps)<abs(n_obs) or include_borders==False and len(timestamps)==abs(n_obs) and (checkstart==timestamps[0] or checkend==timestamps[-1]):
        print('parse_time_interval_observations: not enough observations found - returning checkinterval!')
        return (checkstart,checkend)
    else:
        ### (date, int) or [date, int]
        if n_obs>0:
            ### closed interval
            if include_borders:
                start_relevant = checkstart
                end_relevant = timestamps[n_obs-1]
                
            ### open interval
            else:
                if checkstart==timestamps[0]:
                    start_relevant = timestamps[1]
                    end_relevant = timestamps[n_obs]
                else:
                    start_relevant = timestamps[0]
                    end_relevant = timestamps[n_obs-1]
                    
        ### (int, date) or [int, date]
        else:
            ### closed interval
            if include_borders:
                start_relevant = timestamps[n_obs]
                end_relevant = checkend
                
            ### open interval
            else:
                if checkend==timestamps[-1]:
                    start_relevant = timestamps[n_obs-1]
                    end_relevant = timestamps[-2]
                else:
                    start_relevant = timestamps[n_obs]
                    end_relevant = timestamps[-1]
        
        return (start_relevant,end_relevant)


def filter_times_reverse_n_obs(timestamps: Iterable[TimeType],
                               time_difference: dt.timedelta,   # obsolete parameter to support sentinelhub_process.get_available_timestamps signature)
                               time_difference_observations: dt.timedelta,
                               n_obs: int) -> List[TimeType]:
    timestamps = sorted(set(timestamps))[::-1]

    filtered_timestamps: List[TimeType] = []

    for current_timestamp in timestamps:
        if not filtered_timestamps or filtered_timestamps[-1] - current_timestamp > time_difference_observations:
            filtered_timestamps.append(current_timestamp)
            if len(filtered_timestamps) == n_obs:
                break

    assert len(filtered_timestamps) == n_obs
    return filtered_timestamps[::-1]
