
import pytest

import pandas as pd
from datetime import datetime


from brainmaze_utils.annotations import merge_annotations, tile_annotations



def test_merge_annotations():

    df_test = pd.DataFrame({
        'start':        [ 0,  30,  60,  70,   80, 100, 150, 201, 240],
        'end':          [30,  60,  70,  80,  100, 150, 200, 230, 250],
        'annotation':   ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'a', 'a']
    })

    df_reference = pd.DataFrame({
        'start':        [ 0,  60,  80,  201, 240],
        'end':          [60,  80,  200, 230, 250],
        'annotation':   ['a', 'b', 'a', 'a', 'a']
    })

    df_merged = merge_annotations(df_test)

    assert df_reference.equals(df_merged)


def test_tile_annotations():
    df_test = pd.DataFrame({
        'start':        [  0,  60,  90, 221, 240],
        'end':          [ 60,  90, 210, 240, 250],
        'annotation':   ['a', 'b', 'a', 'a', 'a']
    })

    df_reference = pd.DataFrame({
        'start':        [ 0,  30,  60,   90, 120, 150, 180, 221, 240],
        'end':          [30,  60,  90,  120, 150, 180, 210, 240, 250],
        'annotation':   ['a', 'a', 'b', 'a', 'a', 'a', 'a', 'a', 'a']
    })

    df_tiled = tile_annotations(df_test, 30)

    assert df_reference.equals(df_tiled)






def test_filter_by_key_removes_matching_rows():
    # documents the actual behaviour: rows equal to `value` are DROPPED, others kept
    from brainmaze_utils.annotations import filter_by_key
    df = pd.DataFrame({'annotation': ['AWAKE', 'Arrousal', 'N2', 'Arrousal'],
                       'start': [0, 1, 2, 3]})
    out = filter_by_key(df, 'annotation', 'Arrousal')
    assert list(out['annotation']) == ['AWAKE', 'N2']
    assert list(out['start']) == [0, 2]
