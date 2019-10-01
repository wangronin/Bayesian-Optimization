# procedures for retrieving the time series data
import time, datetime, pytz, dateutil.parser
import pandas as pd
import numpy as np

from datetime import timedelta
from copy import copy


def add_ts_segment(df, window_size=300, align=True, inplace=False):
    """Segmenting rows (records) according to a constant window size
    and add the timestamp of the leading edge (of each segment) as the
    index of the resulting dataframe

    Params:
        df: pd.dataframe,
            the inoput dataframe
        window_size: int,
            the window size (in secconds) used to segment the records
        align: bool,
            whether `id` timestamps should aligned to `window_size`, meaning `id` is always
            a multiple of `window_size`
        inplace: bool,
            whether to modify `df` in place

    Returns:
        df : pd.dataframe,
            the dataframe with column `id` added, representing IDs of segments
    """
    if not inplace:
        df_ = df.copy()

    ts = df_.index
    ts_delta = timedelta(seconds=window_size)
    ts_start = df_.index[0]

    if align:
        s = int(ts_start.second / window_size) * window_size - ts_start.second
        delta = datetime.timedelta(seconds=s, microseconds=-ts_start.microsecond)
        ts_start += delta

    ts_end = df_.index[-1]

    i = int(0)
    index = np.empty(df_.shape[0], dtype='object')
    t_next = ts_start + ts_delta
    t = ts_start

    while t <= ts_end:
        idx = np.nonzero(np.bitwise_and(ts >= t, ts < t_next))[0]

        # in case there is a large gap in the data
        if len(idx) == 0:
            i += 1
            t = t_next
            t_next += ts_delta
            continue
    
        # in case ts_delta is the same as the data gradularity
        if len(idx) == 1:  
            index[idx] = copy(t)
        else:
            # NOTE: the index must be t_next as this index will be used 
            # to match the target value
            index[idx] = copy(t_next)

        i += 1
        t = t_next
        t_next += ts_delta
        
    df_['id'] = index
    return df_

def get_ts_from_partial(partial, ts, offset=None):
    """ return datetime object from a partial string, e.g., '10:00:00', where the remaining 
        required fields (e.g., year) are filled using values in `ts`
    """
    ts = to_datetime(ts)
    tm = dateutil.parser.parse(partial)
    tm = tm.replace(day = ts.day)
    tm = tm.replace(month = ts.month)
    tm = tm.replace(year = ts.year)
    
    if offset is not None:
        if offset.endswith('d'):
            delta = datetime.timedelta(days=eval(offset[:-1]))
        elif offset.endswith('m'):
            delta = datetime.timedelta(minutes=eval(offset[:-1]))
        elif offset.endswith('h'):
            delta = datetime.timedelta(hours=eval(offset[:-1]))
        tm += delta
    
    return to_datetime(tm.timestamp())

def get_timestamp(as_datetime=True):
    """ return datetime object in UTC
    """
    t = time.time()
    return (datetime.datetime.fromtimestamp(t, tz=pytz.utc) if as_datetime else t)
    
def to_datetime(t):
    if isinstance(t, str):
        if t.isnumeric():
            dt = datetime.datetime.fromtimestamp(eval(t), tz=pytz.utc)
        else:
            dt = dateutil.parser.parse(t)
            dt = datetime.datetime.fromtimestamp(dt.timestamp(), tz=pytz.utc)
    elif isinstance(t, int) or isinstance(t, float):
        dt = datetime.datetime.fromtimestamp(t, tz=pytz.utc)
    elif isinstance(t, datetime.datetime):
        dt = datetime.datetime.fromtimestamp(t.timestamp(), tz=pytz.utc)
    else:
        dt = t
    return dt

# TODO: move the functions below to other modules
def compute_target__last_price(df):
    """
    Params:
        df: pd.dataframe,
            the input dataframe

    Returns:
        y : vector,
            the target value for prediction / forcasting, of shape (n_sample, )
    """
    y = df.filter(regex='^(id|ask_price1|bid_price1)').groupby('id').tail(1)
    y.index = y['id']
    y.drop(['id'], axis=1, inplace=True)
    return np.mean(y.values, axis=1)

def compute_target__avg_price(df, func='mean'):
    """
    Params:
        df: pd.dataframe,
            the input dataframe

    Returns:
        y : vector,
            the target value for prediction / forcasting, of shape (n_sample, )
    """
    y = df.filter(regex='^(id|ask_price1|bid_price1)').groupby('id').mean()
    return np.mean(y.values, axis=1)

def compute_target__open_close_high_low(df):
    """
    Params:
        df: pd.dataframe,
            the inoput dataframe

    Returns:
        y : vector,
            the target value for prediction / forcasting, of shape (n_sample, )
    """
    df['y'] = (df.high + df.low) / 2
    return df[['id', 'y']].groupby('id').mean().values.ravel()
    
def compute_target__avg_ask_price(df):
    """
    Params:
        df: pd.dataframe,
            the input dataframe

    Returns:
        y : vector,
            the target value for prediction / forcasting, of shape (n_sample, )
    """
    y = df.filter(regex='^(id|ask_price)').groupby('id').mean()
    return np.mean(y.values, axis=1)

def compute_target__weighted_avg_ask_price(df):
    """
    Params:
        df: pd.dataframe,
            the inoput dataframe

    Returns:
        y : vector,
            the target value for prediction / forcasting, of shape (n_sample, )
    """
    ask_price = df.filter(regex='^(id|ask_price)').values
    ask_size = df.filter(regex='^(id|ask_size)').values
    df_ = pd.DataFrame(np.c_[df.id.values, 
                             np.sum(ask_price * ask_size, axis=1) / np.sum(ask_size, axis=1)], 
                             columns=['id', 'target'])
    
    y = df_.groupby('id').mean()
    return y.values.ravel()

def compute_target(df, target_fun='compute_target__last_price', target_type='original'):
    y = eval(target_fun)(df)
    
    if target_type == 'original':         # to predict the value itself
        y = y[1:]                  
    elif target_type == 'differential':   # to predict the differential values
        y = y[1:] - y[0:-1]        
    elif target_type == 'binary':         # binarized differential values
        y = np.array(list(map(int, y[1:] - y[0:-1] > 0)))  

    return y