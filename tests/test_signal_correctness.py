# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Regression tests for the signal-processing fixes (reference values from analytic signals)."""

import numpy as np
import pytest

from brainmaze_utils.signal import (
    get_datarate, decimate, nandecimate, resample, fft_filter, buffer, detrend,
    unify_sampling_frequency, downsample_min_max,
)


def _tone(fs, dur, f, phase=0.0, amp=1.0):
    t = np.arange(int(round(fs * dur))) / fs
    return amp * np.sin(2 * np.pi * f * t + phase)


# --------------------------------------------------------------------------- get_datarate
def test_get_datarate_1d_2d_list():
    x = np.ones(10)
    x[:2] = np.nan
    assert get_datarate(x) == pytest.approx(0.8)
    X = np.vstack([x, np.ones(10)])
    np.testing.assert_allclose(get_datarate(X), [0.8, 1.0])
    assert get_datarate([x, np.ones(4)]) == pytest.approx([0.8, 1.0])


# --------------------------------------------------------------------------- decimate
@pytest.mark.parametrize('fs, fs_new', [
    (2500, 250), (3000, 250), (4000, 250), (1000, 50), (32000, 1000), (32000, 2000),
    (5000, 200), (2048, 500), (32556, 1000), (30000, 256), (499.9, 200),
])
def test_decimate_stable_and_accurate(fs, fs_new):
    # 16th-order b/a Butterworth used to go unstable for ratios >= ~10 -> all-NaN output
    f0 = min(10.0, fs_new / 8)
    x = _tone(fs, 6, f0)
    y = decimate(x, fs, fs_new)
    assert y.shape == (int(round(len(x) * fs_new / fs)),)
    assert np.all(np.isfinite(y))
    ref = _tone(fs_new, 6, f0)[:len(y)]
    core = slice(int(fs_new), -int(fs_new))
    assert np.abs(y[core] - ref[core]).max() < 2e-3


def test_decimate_no_time_drift_non_integer_ratio():
    # 32556 -> 1000 Hz over 10 min: the old limit_denominator(1000) ratio drifted ~0.4 rad
    fs, fs_new = 32556, 1000
    x = _tone(fs, 600, 7.0)
    y = decimate(x, fs, fs_new)
    ref = _tone(fs_new, 600, 7.0)
    tail = slice(len(y) - 5 * fs_new, len(y) - fs_new)
    assert np.abs(y[tail] - ref[tail]).max() < 1e-3


def test_decimate_anti_aliasing():
    # a tone above the new Nyquist must be removed, not aliased
    fs, fs_new = 1000, 100
    y = decimate(_tone(fs, 10, 80.0), fs, fs_new)
    assert np.sqrt(np.mean(y[100:-100] ** 2)) < 1e-3


def test_decimate_nan_gap_remasked():
    fs, fs_new = 1000, 100
    x = _tone(fs, 10, 2.0) + 100
    x[3000:4000] = np.nan
    y = decimate(x, fs, fs_new)
    ref = _tone(fs_new, 10, 2.0) + 100
    nan_out = np.isnan(y)
    # gap 3.0-4.0 s -> output samples 300..399 (+ the boundary sample at 4.0 s)
    assert nan_out[300:400].all()
    assert nan_out.sum() <= 102
    ok = ~nan_out
    ok[250:450] = False  # filter transients next to the gap are documented
    assert np.abs(y[ok] - ref[ok])[20:-20].max() < 1e-3


def test_decimate_all_nan_channel_and_datarate_multichannel():
    x = np.vstack([_tone(1000, 2, 5.0), np.full(2000, np.nan)])
    x[0, :100] = np.nan
    y, dr = decimate(x, 1000, 250, datarate=True)
    assert y.shape == (2, 500)
    assert np.isnan(y[1]).all()
    np.testing.assert_allclose(dr, [0.95, 0.0])
    # single all-NaN channel still returns the tuple
    y1, dr1 = decimate(np.full(1000, np.nan), 1000, 250, datarate=True)
    assert y1.shape == (250,) and np.isnan(y1).all() and dr1 == 0.0


def test_decimate_does_not_modify_input_and_validates():
    x = _tone(1000, 1, 5.0)
    x[10:20] = np.nan
    x0 = x.copy()
    decimate(x, 1000, 100)
    np.testing.assert_array_equal(x, x0)
    with pytest.raises(ValueError):
        decimate(x, 0, 100)
    with pytest.raises(ValueError):
        decimate(x, 1000, 100, cutoff=600)
    with pytest.raises(ValueError):  # above the NEW Nyquist (50 Hz): would alias
        decimate(x, 1000, 100, cutoff=60)
    with pytest.raises(ValueError):
        decimate(x, 1000, 100, cutoff=0)


# --------------------------------------------------------------------------- nandecimate
def test_nandecimate_fills_with_mean_not_nan_fraction():
    # used to fill gaps with the NaN fraction (0..1) -> error 47 next to the gap for DC=100
    fs, fs_new = 1000, 100
    x = _tone(fs, 10, 2.0) + 100
    x[3000:4000] = np.nan
    y = nandecimate(x, fs, fs_new)
    ref = _tone(fs_new, 10, 2.0) + 100
    assert np.isnan(y[300:400]).all()
    ok = np.isfinite(y)
    ok[:20] = False
    ok[-20:] = False
    assert np.abs(y[ok] - ref[ok]).max() < 0.05


# --------------------------------------------------------------------------- resample
def test_resample_time_axis_exact():
    # linspace(0,1,N) time axes stretched time: 1000->300 Hz gave err 0.073 on a 5 Hz sine
    x = _tone(1000, 1, 5.0)
    y = resample(x, 1000, 300)
    assert y.shape == (300,)
    np.testing.assert_allclose(y, _tone(300, 1, 5.0), atol=2e-4)


def test_resample_integer_ratio_picks_samples():
    x = np.arange(20, dtype=float)
    np.testing.assert_array_equal(resample(x, 10, 5), x[::2])


def test_resample_upsample_linear():
    x = np.array([0.0, 2.0, 4.0])
    np.testing.assert_allclose(resample(x, 1, 2), [0, 1, 2, 3, 4, 4])


def test_resample_nan_numpy2_and_gap():
    # np.NaN crashed on numpy 2
    x = _tone(1000, 1, 5.0)
    x[100:200] = np.nan
    y = resample(x, 1000, 250)
    assert np.isnan(y[25:50]).all()
    assert np.isfinite(y[:25]).all() and np.isfinite(y[51:]).all()


def test_resample_does_not_antialias_by_design():
    # documented: no anti-aliasing. A 400 Hz tone at 1000 -> 250 Hz aliases to 100 Hz.
    y = resample(_tone(1000, 2, 400.0), 1000, 250)
    assert np.sqrt(np.mean(y ** 2)) > 0.5


def test_resample_constant_and_nd():
    np.testing.assert_allclose(resample(np.full(100, 3.0), 100, 50), np.full(50, 3.0))
    X = np.vstack([_tone(100, 1, 1.0), _tone(100, 1, 2.0)])[None]
    Y = resample(X, 100, 50)
    assert Y.shape == (1, 2, 50)
    np.testing.assert_allclose(Y[0, 1], X[0, 1, ::2])


# --------------------------------------------------------------------------- fft_filter
def test_fft_filter_cutoff_bin_symmetric():
    # 10 Hz tone, cutoff 9.5: old code gave RMS 0.354 for both lp and hp
    x = _tone(1000, 1, 10.0)
    lp = fft_filter(x, 1000, 9.5, 'lp')
    hp = fft_filter(x, 1000, 9.5, 'hp')
    assert np.sqrt(np.mean(lp ** 2)) < 1e-10
    assert np.sqrt(np.mean(hp ** 2)) == pytest.approx(np.sqrt(0.5), rel=1e-9)
    np.testing.assert_allclose(fft_filter(x, 1000, 10.5, 'lp'), x, atol=1e-10)


def test_fft_filter_multichannel_and_nan():
    X = np.vstack([_tone(1000, 1, 5.0), _tone(1000, 1, 50.0)])
    Y = fft_filter(X, 1000, 20, 'lp')
    np.testing.assert_allclose(Y[0], X[0], atol=1e-10)
    assert np.abs(Y[1]).max() < 1e-10
    X[0, 3] = np.nan
    with pytest.raises(ValueError):
        fft_filter(X, 1000, 20)


# --------------------------------------------------------------------------- buffer
def test_buffer_overlap_validation():
    with pytest.raises(ValueError):
        buffer(np.arange(100.0), fs=10, segm_size=1, overlap=1)  # used to hang forever
    with pytest.raises(ValueError):
        buffer(np.zeros((2, 100)), fs=10, segm_size=1)  # 2-D was silently buffered along channels


def test_buffer_no_drift_non_integer_fs():
    fs = 499.9
    x = np.arange(int(3600 * fs), dtype=float)
    b = buffer(x, fs=fs, segm_size=30, overlap=15)
    starts = b[:, 0]
    k = np.arange(len(starts))
    np.testing.assert_array_equal(starts, np.round(k * 15 * fs))
    assert b.shape[1] == int(round(30 * fs))


def test_buffer_basic_and_drop():
    x = np.arange(10.0)
    np.testing.assert_array_equal(buffer(x, fs=1, segm_size=4, overlap=2),
                                  [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9]])
    b = buffer(np.arange(9.0), fs=1, segm_size=4, overlap=0, drop=False)
    np.testing.assert_array_equal(b, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 0, 0, 0]])


# --------------------------------------------------------------------------- detrend
def test_detrend_least_squares_and_endpoints():
    x = np.linspace(0, 1, 101)
    y = 5 * x + 2
    y[0] += 10
    r = detrend(y)
    assert abs(np.polyfit(x[1:], r[1:], 1)[0]) < 1.0  # endpoint line left slope 10
    r_old = detrend(y, method='endpoints')
    assert np.polyfit(x[1:], r_old[1:], 1)[0] == pytest.approx(10, rel=1e-6)
    np.testing.assert_allclose(detrend(5 * x + 2), 0, atol=1e-12)


# --------------------------------------------------------------------------- unify / min-max
def test_unify_sampling_frequency_does_not_mutate():
    a = np.random.default_rng(0).standard_normal(2000)
    lst = [a, np.zeros(1000)]
    out, fs = unify_sampling_frequency(lst, [2000, 1000])
    assert fs == 1000
    assert len(lst[0]) == 2000 and lst[0] is a
    assert len(out[0]) == 1000


def test_downsample_min_max_ignores_nan():
    x = np.arange(40, dtype=float)
    x[3] = np.nan
    y, _ = downsample_min_max(x, 100, 10)
    np.testing.assert_array_equal(y, [0, 19, 20, 39])
