from pathlib import Path

from brainmaze_utils.annotations import load_CyberPSG
import pytest
import pandas as pd

DATA = Path(__file__).parent


def test_load_a2():
    # path relative to this file, so the test works from any working directory
    df = load_CyberPSG(str(DATA / 'a2.xml'))
    assert isinstance(df, pd.DataFrame)
    print(df.head())
