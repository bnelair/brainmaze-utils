# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""decimate: non-integer and non-round rates (no exception, no drift), upsampling, cutoff validation."""

import numpy as np
import pytest
import scipy.signal as ss

from brainmaze_utils.signal import decimate, unify_sampling_frequency


def _tone(fs, n, f, phase=0.3):
    return np.sin(2 * np.pi * f * np.arange(n) / fs + phase)


def _bw_gain(fs, cutoff, f):
    """Forward-backward gain of decimate's Butterworth at f."""
    sos = ss.butter(16, cutoff, 'lp', fs=fs, output='sos')
    return abs(ss.sosfreqz(sos, worN=[f], fs=fs)[1][0]) ** 2


@pytest.mark.parametrize('fs, fs_new, seconds', [
    (30000.5, 250, 60),      # exact ratio 500/60001: used to raise ValueError
    (511.9999, 256, 4 * 3600),  # no small fraction; best approximation 1/2 drifted 0.7 samples over 4 h
    (1017.2526, 256, 3600),
    (32556, 1000, 120),
    (499.907, 200, 600),
])
def test_decimate_non_round_rates_exact_timing(fs, fs_new, seconds):
    n = int(fs * seconds)
    y = decimate(_tone(fs, n, 7.0), fs, fs_new)
    assert y.shape == (int(round(n * fs_new / fs)),)
    ref = _tone(fs_new, y.size, 7.0)
    # the END of the record shows any timing drift (one-sample shift ~0.17 at 256 Hz)
    tail = slice(y.size - 20 * int(fs_new), y.size - 3 * int(fs_new))
    assert np.abs(y[tail] - ref[tail]).max() < 1e-5


def test_decimate_non_integer_passband_accuracy():
    # R9: the resample_poly path had ~1e-3 error in the passband; now ~1e-7
    fs, fs_new = 30000, 256
    x = _tone(fs, 60 * fs, 7.0) + _tone(fs, 60 * fs, 70.0, 1.0)
    y = decimate(x, fs, fs_new)
    ref = _tone(fs_new, y.size, 7.0) + _bw_gain(fs, fs_new / 3, 70.0) * _tone(fs_new, y.size, 70.0, 1.0)
    core = slice(5 * fs_new, -5 * fs_new)
    assert np.abs(y[core] - ref[core]).max() < 1e-5


@pytest.mark.parametrize('fs, fs_new', [(150, 200), (180, 200), (134, 200), (100, 200), (128, 256), (100, 256)])
def test_decimate_upsamples(fs, fs_new):
    # R3: v2.0.0 upsampled 150->200 (err 7e-5); the PR raised. 134->200 gave NaN and >=1.5x raised in v2.0.0.
    for f in (5.0, 30.0):
        n = 120 * fs
        y = decimate(_tone(fs, n, f), fs, fs_new)
        assert y.shape == (int(round(n * fs_new / fs)),)
        cutoff = fs_new / 3
        g = _bw_gain(fs, cutoff, f) if cutoff < 0.45 * fs else 1.0
        ref = g * _tone(fs_new, y.size, f)
        core = slice(5 * fs_new, -5 * fs_new)
        assert np.abs(y[core] - ref[core]).max() < 1e-5


def test_decimate_upsample_nan_mask():
    fs, fs_new = 150, 200
    x = _tone(fs, 1500, 5.0)
    x[300:310] = np.nan  # 2.000 .. 2.060 s
    y = decimate(x, fs, fs_new)
    t = np.arange(y.size) / fs_new
    nan_out = np.isnan(y)
    # every output whose bracketing input samples include a NaN is masked, nothing far away
    assert nan_out[(t >= 2.0) & (t <= 309 / fs)].all()
    assert not nan_out[(t < 1.99) | (t > 310 / fs + 0.01)].any()


def test_unify_sampling_frequency_upsamples_to_fs_new():
    # brainmaze-eeg classifiers call unify(..., fs_new=200) on channels that may be < 200 Hz
    x150, x200 = _tone(150, 1500, 5.0), _tone(200, 2000, 5.0)
    out, fs = unify_sampling_frequency([x150, x200], [150, 200], fs_new=200)
    assert fs == 200 and out[0].shape == (2000,) and out[1] is x200
    core = slice(200, -200)
    assert np.abs(out[0][core] - _bw_gain(150, 200 / 3, 5.0) * x200[core]).max() < 1e-5


def test_decimate_same_rate_lowpasses_like_v2():
    # v2.0.0 low-passed at fs/3 when fs_new == fs; kept
    x = _tone(200, 2000, 5.0) + _tone(200, 2000, 90.0)
    y = decimate(x, 200, 200)
    core = slice(200, -200)
    assert np.abs(y[core] - _tone(200, 2000, 5.0)[core]).max() < 1e-3


def test_decimate_cutoff_must_be_below_new_nyquist():
    # Copilot: cutoff=200 with 1000->100 let 70 Hz alias (rms 0.7) through
    x = _tone(1000, 10000, 70.0)
    with pytest.raises(ValueError, match='new Nyquist'):
        decimate(x, 1000, 100, cutoff=80)
    y = decimate(x, 1000, 100, cutoff=45)
    assert np.sqrt(np.mean(y[100:-100] ** 2)) < 1e-3


@pytest.mark.parametrize('n', [0, 1, 2, 5, 31])
def test_decimate_tiny_inputs(n):
    for fs_new in (250, 333.3, 16000):
        y = decimate(np.full(n, 3.0), 8000, fs_new)
        assert y.shape == (int(round(n * fs_new / 8000)),)
        if y.size:
            np.testing.assert_allclose(y, 3.0)
