# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime, timedelta, time
from pandas import Timestamp
from dateutil import tz
import pandas as pd
import numpy as np



from typing import Union

def _validate_numeric_dtype(series, column_name):
    """
    Checks if a pandas Series does not contain datetime, time, or timestamp data.

    Args:
        series (pd.Series): The pandas Series to check.
        column_name (str): The name of the column being checked (for error messages).

    Raises:
        TypeError: If the Series contains datetime, time, or timestamp data.
    """
    disallowed_dtypes = (datetime, time, Timestamp)
    if series.dtype.name == 'object':
        try:
            pd.to_datetime(series)
            raise TypeError(f"[INPUT ERROR]: '{column_name}' column cannot be of datetime, time, or timestamp type.")
        except:
            pass  # It's not easily convertible to datetime, so we assume it's numeric
    elif not series.empty and any(isinstance(series.iloc[0], dtype) for dtype in disallowed_dtypes):
        raise TypeError(f"[INPUT ERROR]: '{column_name}' column cannot be of datetime, time, or timestamp type.")

def _validate_dataframe_annotation_columns(df):
    """Check for annotation dataframe - start, end must be in timestamp format (float / int)"""
    if not isinstance(df, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: An input variable dfs must be of type pandas.DataFrame.')

    if not 'start' in df.columns or not 'end' in df.columns or not 'annotation' in df.columns:
        raise ValueError('[INPUT ERROR]: The dataframe must have [start, end, annotation] columns.]')

    for cname in df.columns:
        if not cname in ['start', 'end', 'annotation', 'duration']:
            raise Warning(f'[WARNING] tile_annotations - {cname} is not a valid column name. Information '
                          f'from this column will be lost during annotation tiling'
                          )

def _convert_to_timestamp(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        return x.timestamp()
    if isinstance(x, (float, int)): return x
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')

def _convert_to_datetime_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int)):
        utc = datetime.utcfromtimestamp(x)
        utc = utc.replace(tzinfo=tz.tzutc())

        return utc
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')

def _convert_to_pandas_timestamp_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int)):
        utc = datetime.utcfromtimestamp(x)
        utc = utc.replace(tzinfo=tz.tzutc())
        utc = Timestamp(utc)
        return utc
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + type(x) + ' recieved.')

def _convert_to_utc(x):
    x = _convert_to_datetime_utc(x)
    return x

def _convert_to_local(x):
    x = _convert_to_datetime_utc(x)
    x = x.astimezone(tz.tzlocal())
    return x

def _convert_to_timezone(x, tzinfo):
    x = _convert_to_datetime_utc(x)
    x = x.astimezone(tzinfo)
    return x

def time_to_local(dfHyp):
    """
    Converts the time into the local timezone. Default by python and PC. Does not enter the timezone explicitely. Cannot be used for creating a annotations figure.
    """
    def convert(x, col_key):
        return _convert_to_local(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def time_to_utc(dfHyp):
    """
    Converts time to the UTC format.

    """

    def convert(x, col_key):
        return _convert_to_utc(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp

def time_to_timezone(dfHyp, tzinfo):
    """
    Converts the annotations into a timezone. The timezone has to be from a python library dateutil
    """
    def convert(x, col_key):
        return _convert_to_timezone(x[col_key], tzinfo)

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp

def time_to_timestamp(dfHyp):
    """
    Converts the annotations time to timestamp.
    """
    def convert(x, col_key):
        return _convert_to_timestamp(x[col_key])

    dfHyp['start'] = dfHyp.apply(lambda x: convert(x, 'start'), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: convert(x, 'end'), axis=1)
    return dfHyp


def create_duration(dfHyp):
    """
    Creates duration for each epoch within the annotations. (Faster on timestamp)
    """
    def duration(x):
        if type(x['start']) in (datetime, Timestamp):
            return _convert_to_timestamp(x['end']) - _convert_to_timestamp(x['start'])
        else:
            return x['end'] - x['start']
    dfHyp['duration'] = dfHyp.apply(lambda x: duration(x), axis=1)
    return dfHyp

def merge_annotations(df: pd.DataFrame):
    """
    Merges epochs with the same annotation and end[i-1] == start[i]

    Updated on 2025-03-27 by F. Mivalt for pandas in Python 3.12

    Args:
        df (pd.DataFrame): DataFrame with 'start', 'end', and 'annotation' columns
                             representing merged annotations.
        dur_threshold (int): The desired size of each chunk in seconds.

    Returns:
        pd.DataFrame: DataFrame with tiled annotations.

    """

    _validate_dataframe_annotation_columns(df)
    _validate_numeric_dtype(df['start'], 'start')
    _validate_numeric_dtype(df['end'], 'end')

    new_df = []
    for idx, row in enumerate(df.iterrows()):
        appbl = True
        if idx > 0:
            if new_df[-1]['annotation'] == row[1].annotation and new_df[-1]['end'] == row[1]['start']:
                appbl = False

        if appbl == True:
            new_df += [row[1][['start', 'end', 'annotation']].to_dict()]
        else:
            new_df[-1]['end'] = row[1]['end']

    new_df = pd.DataFrame(new_df)
    if 'duration' in df.keys():
        new_df = create_duration(new_df)

    return new_df

def tile_annotations(df: pd.DataFrame, dur_threshold:Union[int, float]=30):
    """
    Optimized tile_annotations using vectorized operations.
    Tiles epochs to the max duration given by dur_threshold in seconds. Reverse to the 'merge annotations'.

    Args:
        df (pd.DataFrame): DataFrame with 'start', 'end', and 'annotation' columns
                             representing merged annotations.
        dur_threshold (int): The desired size of each chunk in seconds.

    Returns:
        pd.DataFrame: DataFrame with tiled annotations.
    """

    _validate_dataframe_annotation_columns(df)
    _validate_numeric_dtype(df['start'], 'start')
    _validate_numeric_dtype(df['end'], 'end')

    if not isinstance(dur_threshold, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: dur_threshold must be float or int format giving the maximum duration '
            'of a single annotation. All anotations above this duration threshold will be tiled.'
        )

    if np.isnan(dur_threshold) or np.isinf(dur_threshold) or dur_threshold <= 0:
        raise AssertionError('[INPUT ERROR]: dur_threshold must be a valid number bigger than 0, not nan and not inf')


    if df.empty:
        return pd.DataFrame(columns=['start', 'end', 'annotation'])

    starts = df['start'].to_numpy()
    ends = df['end'].to_numpy()
    annotations = df['annotation'].to_numpy()

    num_annotations = len(df)
    all_tiled_starts = []
    all_tiled_ends = []
    all_tiled_annotations = []

    for i in range(num_annotations):
        start = starts[i]
        end = ends[i]
        annotation = annotations[i]

        chunk_starts = np.arange(start, end, dur_threshold)
        chunk_ends = np.minimum(chunk_starts + dur_threshold, end)
        chunk_annotations = np.full(len(chunk_starts), annotation)

        all_tiled_starts.extend(chunk_starts)
        all_tiled_ends.extend(chunk_ends)
        all_tiled_annotations.extend(chunk_annotations)

    new_df = pd.DataFrame({
        'start': np.array(all_tiled_starts),
        'end': np.array(all_tiled_ends),
        'annotation': np.array(all_tiled_annotations)
    })

    if 'duration' in df.keys():
        new_df = create_duration(new_df)

    return new_df

def create_day_indexes(df: pd.DataFrame, hour: Union[int, float]=12, tzinfo: tz.tz._tzinfo=tz.tzlocal):
    """
    Creates a day index for each epoch within the annotations, given the day-time hour supplied as input parameter.
    The format of start and end has to be an integer or a float in a form representing a timestamp, or a timezone aware datetime  or Timestamp object.
    If the start and end format do not include timezone, the local timezone will be used.
    """
    
    if not isinstance(df, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfHyp must be of a type pandas.DataFrame.')

    if hour < 0 or hour > 23:
        raise ValueError(
            '[VALUE ERROR] - An input variable hour_cut indicating at which hour days are separated from each other must be on the range between 0 - 23. Pasted value: ',
            hour)

    timezone_counter = 0 # timezone_counter == dfHyp.__len__() if timeaware; == 0 if not timeaware; >0 & <dfHyp.__len__() if mismatch
    datetime_format = False
    tzinfo = None
    # check if the data is in
    for ridx, row in df.iterrows():
        if isinstance(row['start'], (Timestamp, datetime)) and isinstance(row['end'], (Timestamp, datetime)):
            if row['start'].tzinfo and (row['start'].tzinfo == row['start'].tzinfo == row['end'].tzinfo == row['end'].tzinfo):
                if ridx == 0:
                    tzinfo = row['start'].tzinfo
                    timezone_counter += 1
                elif row['start'].tzinfo == tzinfo:
                    timezone_counter += 1

    if timezone_counter > 0 and timezone_counter != df.__len__():
        raise ValueError('[VALUE ERROR] - Time zones in the start and end fields are inconsistent')


    df = df.sort_values('start').reset_index(drop=True)
    df['day'] = 0

    max_day = int(np.ceil((df.iloc[-1]['end'] - df.iloc[0]['start']).total_seconds() / (24 * 3600)))
    ref = df['start'][0].replace(hour=hour, minute=0, second=0, microsecond=0)

    for idx in range(max_day):
        df['day'][df['start'] >= ref] = idx + 1
        ref += timedelta(days=1)
    df['day'] -= df['day'].min()
    return df

def filter_by_duration(dfAnnotations: pd.DataFrame, duration: Union[int, float]):
    """
    Keeps only epochs of the duration given by the input.
    """
    if not isinstance(dfAnnotations, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfAnnotations must be of a type pandas.DataFrame.')

    if not isinstance(duration, (int, float)):
        raise AssertionError(
            '[INPUT ERROR]: duration must be float or int format giving the maximum duration of a single annotation. All anotations above this duration threshold will be tiled.')

    if np.isnan(duration) or np.isinf(duration) or duration <= 0:
        raise AssertionError('[INPUT ERROR]: duration must be a valid number bigger than 0, not nan and not inf')

    dfAnnotations = dfAnnotations.loc[dfAnnotations['duration'] == duration].reset_index(drop=True)
    return dfAnnotations

def filter_by_key(dfAnnotations: pd.DataFrame, key: str, value: Union[int, float, str]):
    """
    Removes annotations whose ``key`` column equals ``value``, keeping all others.

    Note
    ----
    This *drops* the rows matching ``value`` (the inverse of what the name may suggest);
    e.g. ``filter_by_key(df, 'annotation', 'Arrousal')`` returns the frame with arousals
    removed.
    """
    return dfAnnotations.loc[dfAnnotations[key] != value].reset_index(drop=True)




