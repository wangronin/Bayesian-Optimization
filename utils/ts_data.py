# procedures for retrieving the time series data
import datetime, re
import pandas as pd
import numpy as np

from .ts import to_datetime

# TODO: maybe convert all functions here to data source class
def data_src(**kwargs):
    """Dispatch data source functions
    """
    if 'data_file' in kwargs:
        func_prefix = 'csv_data_src'
    elif 'host' in kwargs:
        func_prefix = 'db_data_src'
    func = eval(func_prefix + '__' + kwargs['src'])
    del kwargs['src']

    return func(**kwargs)

def db_data_src__binance(discretize_y=False, q=0.7, tables=['candles'], interval='1m', ws=60,
                         **kwargs):
    """
    Params
    ------
        discretize_y : bool,
            if the target (y) value should be discretized for the classification task
        q : float,
            the threshold used to discretize y values
        tables : list,
            it determines the source tables on which data are extracted. Valid tables are
            'candles' (candlestick) and 'books' (booking)
        interval : str,
            the frequency for extracting candlestick information. Supported frequencies are
            '1m', '15m', '1h'.
    """
    from pymysql import connect
    from pymysql.cursors import DictCursor

    connection = connect(charset='utf8mb4', cursorclass=DictCursor, **kwargs)
    connection.autocommit(1)

    t = ws * 1e3
    if interval == '1m':
        candles_table = 'candles'
    elif interval == '15m':
        candles_table = 'candles_15m'
    elif interval == '1h':
        candles_table = 'candles_1h'
    else:
        raise ValueError('Unsupported interval')
        
    if 'candles' in tables:
        with connection.cursor() as cursor:
            sql = 'SHOW COLUMNS FROM `{}`'.format(candles_table)
            cursor.execute(sql)
            col_candles = [_['Field'] for _ in  cursor.fetchall()]
            col_candles = '`' + '`,`'.join(col_candles) + '`'

    if 'books' in tables:
        col_books = ['timestamp'] + \
                    ['bid_price' + str(i) for i in range(1, 6)] + \
                    ['ask_price' + str(i) for i in range(1, 6)] + \
                    ['bid_size' + str(i) for i in range(1, 6)] + \
                    ['ask_size' + str(i) for i in range(1, 6)]
        col_books = '`' + '`,`'.join(col_books) + '`'

    def __func__(start, stop):
        start = to_datetime(start)
        stop = to_datetime(stop)

        # timestamps in mysql has more granularity in mysql!!
        # mysql stores timestamps in milliseconds!!
        if isinstance(start, datetime.datetime):
            start = int(start.timestamp() * 1e3)
        if isinstance(stop, datetime.datetime):
            stop = int(stop.timestamp() * 1e3)

        fun = lambda t: to_datetime(t / 1e3)
        with connection.cursor() as cursor:
            df_candles, df_books, y = None, None, None
            
            if 'candles' in tables:
                # NOTE: the time interval (`start`, `stop`] is taken here in
                # case the provided `start` is already rounded to `interval` and
                # hence one more data point will be retrieved if a close interval 
                # is used  
                sql = 'SELECT %s FROM `%s` WHERE `timestamp` > %s '\
                    'and `timestamp` <= %s'%(col_candles, candles_table, start, stop)
                cursor.execute(sql)
                data = cursor.fetchall()

                if len(data) > 0:
                    df_candles = pd.DataFrame(data) 
                    df_candles.set_index(['timestamp'], inplace=True)
                    df_candles.index = df_candles.index.map(fun)
                    y = df_candles['close']
                    df_candles.drop(['close'], axis=1, inplace=True)
                    
            if 'books' in tables:
                # NOTE: the time interval (`start` - `t`, `stop`] is taken because the data 
                # from books would undergo the feature extraction process, which requires one
                # more `interval`
                sql = 'SELECT %s FROM `books` WHERE `timestamp` > %s '\
                    'and `timestamp` <= %s'%(col_books, start - t, stop)
                cursor.execute(sql)
                data = cursor.fetchall()

                if len(data) > 0:
                    df_books = pd.DataFrame(data) 
                    df_books.set_index(['timestamp'], inplace=True)
                    df_books.index = df_books.index.map(fun)
                    # the target is calculated from books, in case candles is not chosen
                    if y is None:
                        y = df_books.filter(regex='^(ask_price1|bid_price1)').mean(axis=1)
                        y.name = 'avg_price1'

            if discretize_y and y is not None:      # discretize `y` for classification 
                diff = y.values[1:] - y.values[0:-1]
                diff_abs = np.abs(diff / y.values[0:-1])
                threshold = np.quantile(diff_abs, q=q)

                y_ = -1 * np.ones(len(y) - 1, dtype='int')
                mask = diff_abs >= threshold
                idx = np.nonzero(mask)[0][diff[mask] > 0]
                y_[~mask] = 0
                y_[idx] = 1
                y = pd.Series(np.r_[0, y_], index=y.index, name=y.name)
                
            X = []
            if df_candles is not None:
                X += [df_candles]
                
            if df_books is not None:
                X += [df_books]
            return tuple(X), y

    return __func__

def csv_data_src__binance(data_file, **kwargs):
    """ CSV data src for binance source 
    """
    df = pd.read_csv(data_file, header=0, index_col=0)
    df.index = df.index.map(lambda t: to_datetime(t / 1e3))
    tMin, tMax = min(df.index), max(df.index)
    
    def __func__(start=None, stop=None):
        df_ = df.copy()
        if start is not None:
            start = to_datetime(start)
            df_ = df_[start:]
            
        if stop is not None:
            stop = to_datetime(stop)
            df_ = df_[:stop]

        return (df_.drop(['close'], axis=1), ), df_['close']
    
    __func__.tMin = tMin
    __func__.tMax = tMax
    return __func__

def csv_data_src__user(data_file, target, header=0, **kwargs):
    """ CSV data src for user uploaded data set
    """
    # TODO: maybe we should also handle inputs `discretize_y` and `q`
    df = pd.read_csv(data_file, header=header, index_col=0)

    try:
        df.index = df.index.map(to_datetime)
    except ValueError:
        df.index = df.index.map(lambda t: to_datetime(t / 1e3))
        
    tMin, tMax = min(df.index), max(df.index)

    if isinstance(target, int):
        target = df.columns[target]
    assert target in df.columns

    def __func__(start=None, stop=None):
        if start is not None:
            start = to_datetime(start)
            df_ = df[start:].copy()
        else:
            df_ = df.copy()
            
        if stop is not None:
            stop = to_datetime(stop)
            time_delta = datetime.timedelta(milliseconds=1)
            stop -= time_delta
            df_ = df_[:stop].copy()
        
        return (df_.drop(target, axis=1), ), df_[target]
    
    __func__.tMin = tMin
    __func__.tMax = tMax
    return __func__

# TODO: the functions below should be checked
def db_data_src__guzhi(with_books = False,  discretize_y=False, q=0.7,**kwargs):
    from pymysql import connect
    from pymysql.cursors import DictCursor

    connection = connect(charset='utf8mb4', cursorclass=DictCursor, **kwargs)
    connection.autocommit(1)
    with connection.cursor() as cursor:
        sql = 'SHOW COLUMNS FROM `stock_index_futures`'
        cursor.execute(sql)
        col_name = [_['Field'] for _ in  cursor.fetchall()]
        col_name = '`' + '`,`'.join(col_name) + '`'

    def __func__(start, stop):
        start = to_datetime(start)
        stop = to_datetime(stop)
        # timestamps in mysql has more granularity in mysql!!
        if isinstance(start, datetime.datetime):
            start = int(start.timestamp() * 1e3)
        if isinstance(stop, datetime.datetime):
            stop = int(stop.timestamp() * 1e3)
        
        fun = lambda t: to_datetime(t / 1e3)
        with connection.cursor() as cursor:
            sql = 'SELECT %s FROM `stock_index_futures` '\
                'WHERE `timestamp` >= %s and `timestamp` < %s'%(col_name, start, stop)
            cursor.execute(sql)
            data = cursor.fetchall()

            if len(data) > 0:
                df_guzhi = pd.DataFrame(data) 
                df_guzhi.set_index(['timestamp'], inplace=True)
                df_guzhi.index = df_guzhi.index.map(fun)
                y = df_guzhi['close']
                df_guzhi.drop(['close'], axis=1, inplace=True)
            else:
                df_guzhi, y = None, None
            
            if discretize_y and y is not None:
                diff = y.values[1:] - y.values[0:-1]
                diff_abs = np.abs(diff / y.values[0:-1])
                threshold = np.quantile(diff_abs, q=q)

                y_ = -1 * np.ones(len(y) - 1, dtype='int')
                mask = diff_abs >= threshold
                idx = np.nonzero(mask)[0][diff[mask] > 0]
                y_[~mask] = 0
                y_[idx] = 1
                y = pd.Series(np.r_[0, y_], index=y.index, name=y.name)

            X = df_guzhi
            return X, y

    return __func__

# TODO: modify this function according to df_data_src__biance
def csv_data_src__bitmex(data_file, header='../bitmex_data/bitmex.head.csv', **kwargs):
    """ CSV data src ONLY for testing 
    """
    df = pd.read_csv(data_file, header=None, index_col=False)
    columns = pd.read_csv(header, header=None, index_col=False)
    df.columns = columns.values.ravel()
    df.localtime = df.localtime.map(lambda t: to_datetime(t / 1e3))
    
    def __func__(start, end):
        start = to_datetime(start)
        end = to_datetime(end)
        return df[(df.localtime >= start) & (df.localtime <= end)]
    
    return __func__, min(df.localtime), max(df.localtime)

def db_data_src__bitmex(**kwargs):
    from pymysql import connect
    from pymysql.cursors import DictCursor
    connection = connect(charset='utf8mb4', cursorclass=DictCursor, **kwargs)
    connection.autocommit(1)
    
    with connection.cursor() as cursor:
        sql = 'SHOW COLUMNS FROM `bitmex`'
        cursor.execute(sql)
        col_name = [_['Field'] for _ in  cursor.fetchall()]
        col_name = [s for s in col_name \
                    if re.match('^localtime|.*(price|size)[1-5]$', s) is not None]
        col_name = '`' + '`,`'.join(col_name) + '`'

    def __func__(start, stop):
        start = to_datetime(start)
        stop = to_datetime(stop)
        # timestamps in mysql has more granularity in mysql!!
        if isinstance(start, datetime.datetime):
            start = int(start.timestamp() * 1e3)
        if isinstance(stop, datetime.datetime):
            stop = int(stop.timestamp() * 1e3)

        with connection.cursor() as cursor:
            sql = 'SELECT %s FROM `bitmex` WHERE `localtime` >= %s '\
                'and `localtime` < %s'%(col_name, start, stop)
            cursor.execute(sql)
            data = cursor.fetchall()

            if len(data) > 0:
                df = pd.DataFrame(data) 
                df.localtime = df.localtime.map(lambda t: to_datetime(t / 1e3))
                return df
            else:
                return None
            
    return __func__