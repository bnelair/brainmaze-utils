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
    """Check for annotation dataframe - must have start, end, annotation columns.

    Additional columns (e.g. ``channel``) are allowed; the functions using this check
    document how they are carried over.
    """
    if not isinstance(df, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: An input variable dfs must be of type pandas.DataFrame.')

    if not 'start' in df.columns or not 'end' in df.columns or not 'annotation' in df.columns:
        raise ValueError('[INPUT ERROR]: The dataframe must have [start, end, annotation] columns.]')


def _extra_columns(df):
    """Columns other than start/end/annotation/duration, in their original order."""
    return [c for c in df.columns if c not in ('start', 'end', 'annotation', 'duration')]

def _convert_to_timestamp(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        return x.timestamp()
    if isinstance(x, (float, int, np.integer, np.floating)): return x
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + str(type(x)) + ' received.')

def _convert_to_datetime_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int, np.integer, np.floating)):
        return datetime.fromtimestamp(x, tz=tz.tzutc())
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + str(type(x)) + ' received.')

def _convert_to_pandas_timestamp_utc(x):
    if isinstance(x, (datetime, Timestamp)):
        assert x.tzinfo, '[TIMEZONE ERROR] We allow operating with timezone-aware datatypes. This helps preventing inconsistency and errors.'
        x = x.timestamp()
    if isinstance(x, (float, int, np.integer, np.floating)):
        return Timestamp(datetime.fromtimestamp(x, tz=tz.tzutc()))
    raise TypeError('[TYPE ERROR]: input variable has to be of a type pandas Timestamp, datetime, float, or int. However ' + str(type(x)) + ' received.')

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

def _convert_columns(dfHyp, fn):
    dfHyp = dfHyp.copy()
    dfHyp['start'] = dfHyp.apply(lambda x: fn(x['start']), axis=1)
    dfHyp['end'] = dfHyp.apply(lambda x: fn(x['end']), axis=1)
    return dfHyp


def time_to_local(dfHyp):
    """
    Convert ``start``/``end`` to timezone-aware datetimes in the machine's local
    timezone (:func:`dateutil.tz.tzlocal`).

    Accepted inputs per cell: POSIX timestamps in seconds (int/float) or
    timezone-aware ``datetime``/``Timestamp`` (naive datetimes raise).

    Returns
    -------
    pandas.DataFrame
        A **copy**; the input frame is not modified.
    """
    return _convert_columns(dfHyp, _convert_to_local)


def time_to_utc(dfHyp):
    """
    Convert ``start``/``end`` to timezone-aware UTC datetimes.

    Accepted inputs per cell: POSIX timestamps in seconds (int/float) or
    timezone-aware ``datetime``/``Timestamp`` (naive datetimes raise).

    Returns
    -------
    pandas.DataFrame
        A **copy**; the input frame is not modified.
    """
    return _convert_columns(dfHyp, _convert_to_utc)


def time_to_timezone(dfHyp, tzinfo):
    """
    Convert ``start``/``end`` to timezone-aware datetimes in ``tzinfo``
    (a ``datetime.tzinfo``, e.g. from :mod:`dateutil.tz`).

    Returns
    -------
    pandas.DataFrame
        A **copy**; the input frame is not modified.
    """
    return _convert_columns(dfHyp, lambda v: _convert_to_timezone(v, tzinfo))


def time_to_timestamp(dfHyp):
    """
    Convert ``start``/``end`` to POSIX timestamps in seconds (float).

    Accepted inputs per cell: timezone-aware ``datetime``/``Timestamp`` or numbers
    (passed through).

    Returns
    -------
    pandas.DataFrame
        A **copy**; the input frame is not modified.
    """
    return _convert_columns(dfHyp, _convert_to_timestamp)


def create_duration(dfHyp):
    """
    Add a ``duration`` column (``end - start`` in seconds).

    Works for numeric timestamps and for timezone-aware datetimes.

    Returns
    -------
    pandas.DataFrame
        A **copy** with the ``duration`` column; the input frame is not modified.
    """
    def duration(x):
        if isinstance(x['start'], (datetime, Timestamp)):
            return _convert_to_timestamp(x['end']) - _convert_to_timestamp(x['start'])
        else:
            return x['end'] - x['start']
    dfHyp = dfHyp.copy()
    if dfHyp.empty:
        dfHyp['duration'] = pd.Series(dtype=float)
        return dfHyp
    dfHyp['duration'] = dfHyp.apply(lambda x: duration(x), axis=1)
    return dfHyp

def _group_key(values):
    """Hashable merge key; every missing value (None, NaN, NaT, pd.NA) maps to None."""
    key = []
    for v in values:
        if np.ndim(v) == 0 and pd.isna(v):
            v = None
        try:
            hash(v)
        except TypeError:
            v = repr(v)
        key.append(v)
    return tuple(key)


def merge_annotations(df: pd.DataFrame):
    """
    Merge epochs with the same annotation that touch in time (``end == start`` of the
    next one). Reverse of :func:`tile_annotations`.

    Epochs are grouped by ``annotation`` and by the values of every extra column
    (e.g. ``channel``; missing values - None, NaN, ``pd.NA`` - are equal to each
    other). Within a group, epochs are sorted by ``start`` and chains of touching
    epochs are merged; the merged epoch keeps the group's values. Epochs of other
    groups in between (e.g. interleaved channels, or an overlapping label from a
    second scorer) do not prevent merging. A ``duration`` column, if present, is
    recomputed.

    Args:
        df (pd.DataFrame): DataFrame with numeric (timestamp) 'start', 'end' and
            'annotation' columns, optionally more.

    Returns:
        pd.DataFrame: merged annotations sorted by ``start`` (stable), columns
        ``start, end, annotation``, then the extra columns in their original order
        (dtypes kept), then ``duration`` if it was present. The input frame is not
        modified.

    Notes:
        .. note:: **Changed after v2.0.0:**
           v2.0.0 merged only rows that were consecutive in the input order, dropped
           every extra column and raised on some of them. Now the frame is grouped as
           described above, so unsorted input, interleaved channels and overlapping
           labels merge as well; for sorted, non-overlapping single-channel input the
           result is the same as before.
    """

    _validate_dataframe_annotation_columns(df)
    _validate_numeric_dtype(df['start'], 'start')
    _validate_numeric_dtype(df['end'], 'end')

    extra = _extra_columns(df)
    keep = ['start', 'end', 'annotation'] + extra
    d = df[keep].sort_values('start', kind='mergesort').reset_index(drop=True)

    if len(d):
        starts = d['start'].to_numpy()
        ends = d['end'].to_numpy().copy()
        groups = {}
        for i, vals in enumerate(d[['annotation'] + extra].itertuples(index=False, name=None)):
            groups.setdefault(_group_key(vals), []).append(i)
        first_rows, new_ends = [], []
        for rows in groups.values():
            head, cur_end = rows[0], ends[rows[0]]
            for i in rows[1:]:
                if starts[i] == cur_end:
                    cur_end = ends[i]
                    continue
                first_rows.append(head)
                new_ends.append(cur_end)
                head, cur_end = i, ends[i]
            first_rows.append(head)
            new_ends.append(cur_end)
        order = np.argsort(np.asarray(first_rows), kind='mergesort')
        sel = np.asarray(first_rows)[order]
        d_out = d.iloc[sel].reset_index(drop=True)
        d_out['end'] = np.asarray(new_ends, dtype=ends.dtype)[order]
        d = d_out

    if 'duration' in df.keys():
        d = create_duration(d)
    return d

def tile_annotations(df: pd.DataFrame, dur_threshold:Union[int, float]=30):
    """
    Optimized tile_annotations using vectorized operations.
    Tiles epochs to the max duration given by dur_threshold in seconds. Reverse to the 'merge annotations'.

    Args:
        df (pd.DataFrame): DataFrame with numeric (timestamp) 'start', 'end', and
            'annotation' columns representing merged annotations. Extra columns
            (e.g. 'channel') are allowed and copied to every tile of their epoch.
        dur_threshold (int): The desired size of each chunk in seconds.

    Returns:
        pd.DataFrame: DataFrame with tiled annotations (``start, end, annotation``,
        extra columns, then ``duration`` if present in the input).
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


    extra = _extra_columns(df)

    if df.empty:
        cols = ['start', 'end', 'annotation'] + extra + (['duration'] if 'duration' in df.keys() else [])
        return pd.DataFrame(columns=cols)

    starts = df['start'].to_numpy()
    ends = df['end'].to_numpy()

    num_annotations = len(df)
    all_tiled_starts = []
    all_tiled_ends = []
    source_row = []

    for i in range(num_annotations):
        start = starts[i]
        end = ends[i]

        chunk_starts = np.arange(start, end, dur_threshold)
        chunk_ends = np.minimum(chunk_starts + dur_threshold, end)

        all_tiled_starts.extend(chunk_starts)
        all_tiled_ends.extend(chunk_ends)
        source_row.extend([i] * len(chunk_starts))

    source_row = np.asarray(source_row, dtype=np.int64)
    new_df = pd.DataFrame({
        'start': np.array(all_tiled_starts),
        'end': np.array(all_tiled_ends),
        'annotation': df['annotation'].to_numpy()[source_row],
    })
    for c in extra:
        new_df[c] = df[c].to_numpy()[source_row]

    if 'duration' in df.keys():
        new_df = create_duration(new_df)

    return new_df

def create_day_indexes(df: pd.DataFrame, hour: Union[int, float]=12, tzinfo=None):
    """
    Add a ``day`` column: an integer day index for each epoch, where days are cut at
    the given wall-clock ``hour`` (default noon, so one night falls within one day).

    The index of an epoch is the number of calendar "days" (``hour`` to ``hour``)
    between the day containing the earliest epoch's start and the day containing
    its own start; the earliest epoch is day 0. Day boundaries are computed on the
    local wall clock of the chosen timezone, so DST transitions are handled.

    Parameters
    ----------
    df : pandas.DataFrame
        Annotations with a ``start`` column (and ``end``). ``start`` can be POSIX
        timestamps in seconds (int/float) or timezone-aware ``datetime``/``Timestamp``
        values (all in the same timezone).
    hour : int or float
        Hour of day (``0 <= hour < 24``, fractions allowed) at which days are split.
    tzinfo : datetime.tzinfo, optional
        Timezone whose wall clock defines the day boundaries. Default: the timezone of
        the data if ``start`` is timezone-aware, otherwise the machine's local
        timezone (:func:`dateutil.tz.tzlocal`).

        .. warning::
           For **UTC** data (e.g. the output of ``load_CyberPSG``) the default cuts
           days at ``hour`` **UTC**: noon UTC is 6 or 7 AM in Chicago and 1 or 2 PM in
           Prague, not local noon. Pass the recording site's timezone, e.g.
           ``tzinfo=dateutil.tz.gettz('America/Chicago')``, to cut at local ``hour``.
           (Same as v2.0.0.)

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` **sorted by start** (index reset) with an integer ``day``
        column. ``start``/``end`` are returned unchanged.

    Notes
    -----
    .. note:: **Changed after v2.0.0:**
       Rewritten. The previous implementation assigned via chained indexing, which is
       a silent no-op under pandas copy-on-write (pandas >= 3: every epoch got day 0),
       could miss the last day, and ignored the ``tzinfo`` argument.
    """

    if not isinstance(df, pd.DataFrame):
        raise AssertionError('[INPUT ERROR]: Variable dfHyp must be of a type pandas.DataFrame.')

    if hour < 0 or hour >= 24:
        raise ValueError(
            '[VALUE ERROR] - An input variable hour indicating at which hour days are separated from each other must be '
            f'in the range 0 <= hour < 24. Passed value: {hour}')

    if isinstance(tzinfo, type) :  # e.g. tz.tzlocal passed as a class (old default)
        tzinfo = tzinfo()

    df = df.sort_values('start', kind='mergesort').reset_index(drop=True)
    if df.empty:
        df['day'] = pd.Series(dtype=np.int64)
        return df

    starts = df['start']
    first = starts.iloc[0]
    if isinstance(first, (datetime, Timestamp)) or pd.api.types.is_datetime64_any_dtype(starts):
        tzs = {getattr(v, 'tzinfo', None) is not None for v in starts}
        if tzs != {True}:
            raise ValueError('[VALUE ERROR] - start must be timezone-aware datetimes (or numeric timestamps)')
        ts = [_convert_to_timestamp(v) for v in starts]
        if tzinfo is None:
            tzinfo = first.tzinfo
    else:
        ts = starts.astype(float).tolist()
        if tzinfo is None:
            tzinfo = tz.tzlocal()

    # wall-clock time in the target timezone (python datetimes: robust for any tzinfo)
    wall = [datetime.fromtimestamp(t, tz=tzinfo).replace(tzinfo=None) for t in ts]
    shifted_dates = [(w - timedelta(hours=hour)).date() for w in wall]
    d0 = min(shifted_dates)
    df['day'] = np.array([(d - d0).days for d in shifted_dates], dtype=np.int64)
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




