# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the annotation fixes (day indexes, extra columns, copies, file formats)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dateutil import tz

from brainmaze_utils.annotations import (
    create_day_indexes, merge_annotations, tile_annotations, time_to_utc, time_to_timestamp,
    time_to_timezone, create_duration, save_CyberPSG, load_CyberPSG, load_NSRR,
)

DATA = Path(__file__).parent


# --------------------------------------------------------------------------- day indexes
def _ts(s, zone='UTC'):
    return pd.Timestamp(s, tz=zone).timestamp()


def test_create_day_indexes_numeric_utc():
    # pandas >= 3: chained assignment was a no-op -> all 0
    starts = [_ts('2024-01-01 20:00'), _ts('2024-01-02 13:00'), _ts('2024-01-03 11:00'), _ts('2024-01-03 13:00')]
    df = pd.DataFrame({'start': starts, 'end': np.add(starts, 30), 'annotation': ['N2'] * 4})
    out = create_day_indexes(df, hour=12, tzinfo=tz.tzutc())
    assert out['day'].tolist() == [0, 1, 1, 2]  # the last day used to be missed on pandas 2
    assert 'day' not in df.columns  # input not modified


def test_create_day_indexes_datetimes_and_unsorted():
    t = [pd.Timestamp('2024-01-03 22:00', tz='UTC'), pd.Timestamp('2024-01-01 22:00', tz='UTC'),
         pd.Timestamp('2024-01-02 22:00', tz='UTC')]
    df = pd.DataFrame({'start': t, 'end': [x + pd.Timedelta(seconds=30) for x in t], 'annotation': ['a', 'b', 'c']})
    out = create_day_indexes(df, hour=12)
    assert out['annotation'].tolist() == ['b', 'c', 'a']  # sorted by start
    assert out['day'].tolist() == [0, 1, 2]


def test_create_day_indexes_respects_tzinfo():
    # 2024-01-02 15:00 UTC is 09:00 in Chicago (before noon -> previous day there)
    starts = [_ts('2024-01-01 20:00'), _ts('2024-01-02 15:00')]
    df = pd.DataFrame({'start': starts, 'end': np.add(starts, 30), 'annotation': ['x', 'y']})
    assert create_day_indexes(df, 12, tzinfo=tz.tzutc())['day'].tolist() == [0, 1]
    assert create_day_indexes(df, 12, tzinfo=tz.gettz('America/Chicago'))['day'].tolist() == [0, 0]


def test_create_day_indexes_validation():
    df = pd.DataFrame({'start': [0.0], 'end': [1.0], 'annotation': ['a']})
    with pytest.raises(ValueError):
        create_day_indexes(df, hour=24)


# --------------------------------------------------------------------------- merge / tile
def test_merge_annotations_sorts_and_keeps_extra_columns():
    df = pd.DataFrame({
        'start': [30., 0., 60., 90.],
        'end': [60., 30., 90., 120.],
        'annotation': ['N2', 'N2', 'N2', 'N2'],
        'channel': ['a', 'a', 'b', 'b'],
    })
    out = merge_annotations(df)
    assert out.to_dict('list') == {'start': [0., 60.], 'end': [60., 120.], 'annotation': ['N2', 'N2'],
                                   'channel': ['a', 'b']}
    assert df['start'].tolist() == [30., 0., 60., 90.]


def test_tile_annotations_keeps_extra_columns():
    df = pd.DataFrame({'start': [0.], 'end': [70.], 'annotation': ['Seizure'], 'channel': ['LA1']})
    out = tile_annotations(df, 30)
    assert out['start'].tolist() == [0., 30., 60.]
    assert out['end'].tolist() == [30., 60., 70.]
    assert out['channel'].tolist() == ['LA1'] * 3


def test_time_conversions_return_copies():
    df = pd.DataFrame({'start': [0., 30.], 'end': [30., 60.], 'annotation': ['a', 'b']})
    ref = df.copy()
    utc = time_to_utc(df)
    pd.testing.assert_frame_equal(df, ref)
    assert utc['start'].iloc[1] == pd.Timestamp(30, unit='s', tz='UTC')
    back = time_to_timestamp(utc)
    np.testing.assert_allclose(back['start'].astype(float), [0., 30.])
    tz_df = time_to_timezone(df, tz.gettz('Europe/Prague'))
    pd.testing.assert_frame_equal(df, ref)
    assert tz_df['start'].iloc[0].utcoffset().total_seconds() == 3600
    dur = create_duration(df)
    assert 'duration' not in df.columns and dur['duration'].tolist() == [30., 30.]


# --------------------------------------------------------------------------- CyberPSG
def test_load_a2_with_tiling_and_channel_column():
    # extra 'channel' column used to raise a Warning exception when tiling
    df = load_CyberPSG(str(DATA / 'a2.xml'), tile=30)
    assert len(df) > 0
    assert (df['duration'] <= 30 + 1e-9).all()


def _save(tmp_path, df, name='x.xml'):
    p = tmp_path / name
    save_CyberPSG(str(p), df)
    return p


def test_cyberpsg_round_trip_preserves_labels(tmp_path):
    t0 = 1.7e9
    df = pd.DataFrame({'start': [t0, t0 + 30, t0 + 60], 'end': [t0 + 30, t0 + 60, t0 + 61.5],
                       'annotation': ['N2', 'REM', 'Spike'], 'channel': [None, None, 'LA1']})
    out = load_CyberPSG(str(_save(tmp_path, df)))
    assert out['annotation'].tolist() == ['N2', 'REM', 'Spike']  # 'N2' came back as 'N2_best'
    np.testing.assert_allclose(time_to_timestamp(out)['start'].astype(float), df['start'])
    assert out['channel'].iloc[2] == 'LA1'


def test_cyberpsg_duplicate_type_names_and_7_digit_fraction(tmp_path):
    t0 = 1.7e9
    df = pd.DataFrame({'start': [t0, t0 + 30], 'end': [t0 + 30, t0 + 60], 'annotation': ['N2', 'OTHER']})
    p = _save(tmp_path, df)
    xml = p.read_text()
    # two annotation types with the same name (e.g. two scorers) + .NET 7-digit fractions
    xml = xml.replace('<name>OTHER</name>', '<name>N2</name>')
    xml = xml.replace('.000000</startTimeUtc>', '.1234567</startTimeUtc>', 1)
    p.write_text(xml)
    out = load_CyberPSG(str(p))
    assert out['annotation'].tolist() == ['N2', 'N2']  # second used to become 'error_unknown'
    assert out['start'].iloc[0].microsecond == 123456


# --------------------------------------------------------------------------- NSRR
_NSRR = """<?xml version="1.0" encoding="UTF-8"?>
<PSGAnnotation>
<SoftwareVersion>Compumedics</SoftwareVersion>
<EpochLength>30</EpochLength>
<ScoredEvents>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Wake|0</EventConcept><Start>0.0</Start><Duration>30.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 3 sleep|3</EventConcept><Start>30.0</Start><Duration>30.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>Stage 4 sleep|4</EventConcept><Start>60.0</Start><Duration>60.0</Duration></ScoredEvent>
<ScoredEvent><EventType>Stages|Stages</EventType><EventConcept>REM sleep|5</EventConcept><Start>120.0</Start><Duration>30.0</Duration></ScoredEvent>
</ScoredEvents>
</PSGAnnotation>
"""


def test_nsrr_stage3_and_stage4_map_to_n3(tmp_path):
    # Stage 3 was deleted from the TwoWayDict mapping -> KeyError for any file with Stage 3
    p = tmp_path / 'nsrr.xml'
    p.write_text(_NSRR)
    hyp = load_NSRR(str(p))
    assert hyp['annotation'].tolist() == ['WAKE', 'N3', 'N3', 'REM']
    assert hyp['end'].tolist() == [30., 60., 120., 150.]


def test_merge_interleaved_channels():
    # Copilot: A:0-30, B:0-30, A:30-60, B:30-60 sorted by start interleaves the channels -> nothing merged
    df = pd.DataFrame({'start': [0., 0., 30., 30.], 'end': [30., 30., 60., 60.],
                       'annotation': ['N2'] * 4, 'channel': ['A', 'B', 'A', 'B']})
    out = merge_annotations(df)
    assert out[['start', 'end', 'channel']].values.tolist() == [[0., 60., 'A'], [0., 60., 'B']]
    # reverse of tile, also with an overlapping second label in between
    df = pd.DataFrame({'start': [0., 0., 30., 60.], 'end': [30., 90., 60., 90.],
                       'annotation': ['N2', 'IED', 'N2', 'N2']})
    out = merge_annotations(df)
    assert out[['start', 'end', 'annotation']].values.tolist() == [[0., 90., 'N2'], [0., 90., 'IED']]


def test_merge_missing_values_and_dtypes():
    # Copilot: pd.NA == x is pd.NA, and all() on it raises. Missing values now form one group.
    for col in (pd.array([pd.NA, pd.NA, 'c1', 'c1'], dtype='string'),
                pd.array([pd.NA, pd.NA, 1, 1], dtype='Int64'),
                pd.Series([None, np.nan, 'c1', 'c1'], dtype=object)):
        df = pd.DataFrame({'start': [0., 30., 60., 90.], 'end': [30., 60., 90., 120.],
                           'annotation': ['A'] * 4, 'channel': col})
        out = merge_annotations(df)
        assert out['start'].tolist() == [0., 60.] and out['end'].tolist() == [60., 120.]
        assert out['channel'].isna().tolist() == [True, False]
        assert out['channel'].dtype == df['channel'].dtype


def test_merge_matches_v2_on_sorted_single_channel():
    df = pd.DataFrame({'start': [0., 30., 60., 90., 150.], 'end': [30., 60., 90., 120., 180.],
                       'annotation': ['N2', 'N2', 'N3', 'N3', 'N3'], 'duration': 30.})
    out = merge_annotations(df)
    assert out.values.tolist() == [[0., 60., 'N2', 60.], [60., 120., 'N3', 60.], [150., 180., 'N3', 30.]]


def test_tile_empty_keeps_duration_column():
    # Copilot: the empty path dropped 'duration'
    df = pd.DataFrame({'start': pd.Series(dtype=float), 'end': pd.Series(dtype=float),
                       'annotation': pd.Series(dtype=object), 'channel': pd.Series(dtype=object),
                       'duration': pd.Series(dtype=float)})
    assert list(tile_annotations(df, 30).columns) == ['start', 'end', 'annotation', 'channel', 'duration']
    assert list(tile_annotations(df.drop(columns='duration'), 30).columns) == ['start', 'end', 'annotation', 'channel']


# --------------------------------------------------------------------------- CyberPSG _bm suffix (maintainer decision)
_V2_FILE = DATA / 'data' / 'cyberpsg_written_by_v2.0.0.xml'  # written by the v2.0.0 writer (labels *_best)
_STANDARD = ['AWAKE', 'N1', 'N2', 'N3', 'REM', 'UNKNOWN', 'Arousal', 'N', 'SLP', 'IED', 'seizure', 'seizure_05', 'seizure_08']


def _types(path):
    from brainmaze_utils.annotations._formats.CyberPSG import CyberPSGFile
    return {name: uid for uid, name in CyberPSGFile(str(path)).get_annotation_types().items()}


def _frame(labels, t0=1.7e9):
    return pd.DataFrame({'start': [t0 + 30 * i for i in range(len(labels))],
                         'end': [t0 + 30 * i + 30 for i in range(len(labels))], 'annotation': labels})


def test_cyberpsg_writer_bm_suffix_and_v2_uuids(tmp_path):
    labels = _STANDARD + ['Sleep stage N2', 'MyLabel']
    p = _save(tmp_path, _frame(labels))
    new, old = _types(p), _types(_V2_FILE)
    # standard labels: '<label>_bm' on disk, with exactly the UUID v2.0.0 wrote for '<label>_best'
    for lab in _STANDARD:
        assert new[lab + '_bm'] == old[lab + '_best'], lab
    assert new['IED_bm'].endswith('000000000011') and new['seizure_bm'].endswith('000000000013')
    # other labels: as given; standard_UUID entries kept
    assert new['Sleep stage N2'] == old['Sleep stage N2']
    assert 'MyLabel' in new and not any(k.endswith('_best') for k in new)


def test_cyberpsg_round_trip_new_file(tmp_path):
    labels = _STANDARD + ['Sleep stage N2', 'MyLabel']
    out = load_CyberPSG(str(_save(tmp_path, _frame(labels))))
    assert out['annotation'].tolist() == labels
    raw = load_CyberPSG(str(_save(tmp_path, _frame(labels))), strip_suffixes=False)
    assert raw['annotation'].tolist() == [l + '_bm' for l in _STANDARD] + ['Sleep stage N2', 'MyLabel']


def test_cyberpsg_legacy_best_file_loads_to_same_labels():
    out = load_CyberPSG(str(_V2_FILE))
    assert out['annotation'].tolist() == _STANDARD + ['Sleep stage N2', 'MyLabel', 'IED']
    assert out['channel'].iloc[-1] == 'LA1'
    raw = load_CyberPSG(str(_V2_FILE), strip_suffixes=False)
    assert raw['annotation'].iloc[0] == 'AWAKE_best'


def test_cyberpsg_mixed_collection_same_label_set(tmp_path):
    new = _save(tmp_path, _frame(_STANDARD + ['Sleep stage N2', 'MyLabel']))
    dfs = load_CyberPSG([str(_V2_FILE), str(new)], verbose=False)
    assert set(dfs[0]['annotation']) == set(dfs[1]['annotation'])
    # re-saving a legacy file gives the same types (names *_bm, same UUIDs) as a new file
    resaved = _types(_save(tmp_path, load_CyberPSG(str(_V2_FILE)), name='resaved.xml'))
    assert {k: v for k, v in resaved.items() if k != 'MyLabel'} == {k: v for k, v in _types(new).items() if k != 'MyLabel'}


def test_cyberpsg_best_and_bare_label_are_one_type(tmp_path):
    # v2.0.0 raised KeyError (both became 'N2_best'); the PR wrote two types, order-dependent UUIDs
    df = _frame(['N2', 'N2_best', 'N2_bm', 'IED_best'])
    p = _save(tmp_path, df)
    types = _types(p)
    assert sorted(types) == ['IED_bm', 'N2_bm']
    assert load_CyberPSG(str(p))['annotation'].tolist() == ['N2', 'N2', 'N2', 'IED']
