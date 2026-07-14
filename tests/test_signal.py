# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from brainmaze_utils.signal import downsample_min_max

import pytest
import numpy as np

def test_import():
    print("Testing import 'from brainmaze_utils.signal'")
    try :
        import brainmaze_utils.signal
        assert True
    except ImportError:
        assert False


def test_simple_1d_signal():
    original_fs = 100.0
    final_fs = 10.0
    # effective_segment_fs = 5 Hz, window_size = 100/5 = 20 samples
    test_signal = np.array([
        1, 2, 0, 3, 4, 5, 1, 6, 2, 7,  # Window 1: min 0 (idx 2), max 7 (idx 9) -> [0, 7]
        8, 3, 9, 1, 2, 8, 0, 4, 3, 5,  # Window 2: max 9 (idx 2), min 0 (idx 6) -> [9, 0]
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Window 3: min 1, max 1 -> [1, 1]
        9, 8, 7, 6, 5, 4, 3, 2, 1, 0  # Window 4: max 9, min 0 -> [9, 0]
    ], dtype=float)  # 40 samples

    expected_downsampled_signal = np.array([0., 9., 9., 0.])

    downsampled_signal, actual_fs = downsample_min_max(test_signal, original_fs, final_fs)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert len(downsampled_signal) == 4
    assert actual_fs == pytest.approx(final_fs)



# --- Test Case 2: 2D Signal (Multiple Channels) ---
def test_2d_signal():
    original_fs = 200.0
    final_fs = 20.0
    # effective_segment_fs = 10 Hz, window_size = 200/10 = 20 samples
    test_signal = np.array([
        [1, 5, 0, 6, 2, 8, 3, 9, 4, 7, 1, 5, 0, 6, 2, 8, 3, 9, 4, 7],
        # Ch1, Win1: min 0 (idx 2), max 9 (idx 7) -> [0, 9]
        [9, 1, 8, 2, 7, 3, 6, 0, 5, 4, 9, 1, 8, 2, 7, 3, 6, 0, 5, 4]
        # Ch2, Win1: min 0 (idx 7), max 9 (idx 0) -> [9, 0]
    ], dtype=float)  # 2 channels, 20 samples. num_windows = 1. Output length = 2 per channel.

    expected_downsampled_signal = np.array([
        [0., 9.],
        [9., 0.]
    ])

    downsampled_signal, actual_fs = downsample_min_max(test_signal, original_fs, final_fs)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert downsampled_signal.shape == (2, 2)
    assert actual_fs == pytest.approx(final_fs)


# --- Test Case 3: Short Signal (Shorter than window_size) ---
def test_short_signal():
    original_fs = 100.0
    final_fs = 10.0  # window_size = 20
    short_signal = np.arange(10, dtype=float)  # Length 10, window_size 20

    expected_downsampled_signal = np.array([])
    expected_actual_fs = 0.0  # As per current implementation for num_windows = 0

    downsampled_signal, actual_fs = downsample_min_max(short_signal, original_fs, final_fs)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert len(downsampled_signal) == 0
    assert actual_fs == pytest.approx(expected_actual_fs)


# --- Test Case 4: Edge Case - window_size = 1 ---
def test_window_size_one():
    original_fs = 100.0
    final_fs = 200.0  # This makes effective_segment_fs = 100, so window_size = 100/100 = 1
    test_signal = np.array([1., 3., 2., 5., 4.], dtype=float)

    # Each sample is a window. Min and Max are the sample itself.
    # argmin=0, argmax=0 for each window. Goes to max_first_or_simult path.
    # Output for [1]: max_val=1, min_val=1 -> [1, 1]
    expected_downsampled_signal = np.array([1., 1., 3., 3., 2., 2., 5., 5., 4., 4.])
    expected_actual_fs = 200.0  # (2 * 100) / 1 = 200

    downsampled_signal, actual_fs = downsample_min_max(test_signal, original_fs, final_fs)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert len(downsampled_signal) == 10
    assert actual_fs == pytest.approx(expected_actual_fs)


# --- Test Case 5: Test with Rounding of Window Size ---
def test_rounding_of_window_size():
    original_fs = 1024.0
    final_fs = 100.0
    # effective_segment_fs = 50.0
    # window_size = round(1024.0 / 50.0) = round(20.48) = 20
    # expected_actual_fs = (2 * 1024.0) / 20 = 2048.0 / 20 = 102.4

    # Create a signal long enough for several windows
    num_samples = 1024 * 2  # 2 seconds of data
    test_signal = np.random.rand(num_samples)

    num_windows = num_samples // 20
    expected_output_length = num_windows * 2
    expected_actual_fs = 102.4

    downsampled_signal, actual_fs = downsample_min_max(test_signal, original_fs, final_fs)

    assert len(downsampled_signal) == expected_output_length
    assert actual_fs == pytest.approx(expected_actual_fs)


# --- Test Case 6: Empty Input Signals ---
def test_empty_input_1d():
    empty_signal_1d = np.array([], dtype=float)
    expected_downsampled_signal = np.array([])
    expected_actual_fs = 0.0

    downsampled_signal, actual_fs = downsample_min_max(empty_signal_1d, 100.0, 10.0)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert actual_fs == pytest.approx(expected_actual_fs)


def test_empty_input_2d():
    empty_signal_2d = np.empty((2, 0), dtype=float)
    # Expected shape based on implementation: (num_channels, 0)
    expected_downsampled_signal = np.empty((2, 0), dtype=float)
    expected_actual_fs = 0.0

    downsampled_signal, actual_fs = downsample_min_max(empty_signal_2d, 100.0, 10.0)

    assert downsampled_signal.shape == expected_downsampled_signal.shape
    np.testing.assert_array_equal(downsampled_signal,
                                  expected_downsampled_signal)  # Checks values too, which is fine for empty
    assert actual_fs == pytest.approx(expected_actual_fs)


# --- Additional Test: Signal with length not a multiple of window_size ---
def test_signal_not_multiple_of_window_size():
    original_fs = 100.0
    final_fs = 10.0  # window_size = 20
    test_signal = np.arange(25, dtype=float)  # Length 25. One full window of 20, 5 samples ignored.

    # Expected processing for first 20 samples:
    # Window 1: np.arange(20) -> min 0, max 19. Output: [0, 19]
    expected_downsampled_signal = np.array([0., 19.])
    expected_output_length = 1 * 2  # 1 window * 2 points
    expected_actual_fs = 10.0

    downsampled_signal, actual_fs = downsample_min_max(test_signal, original_fs, final_fs)

    np.testing.assert_array_equal(downsampled_signal, expected_downsampled_signal)
    assert len(downsampled_signal) == expected_output_length
    assert actual_fs == pytest.approx(expected_actual_fs)


# --------------------------------------------------------------------------------------
# fft_filter
# --------------------------------------------------------------------------------------
from brainmaze_utils.signal import fft_filter

_FS = 200.0
_T = np.arange(0, 10, 1 / _FS)
_LOW = np.sin(2 * np.pi * 2 * _T)     # 2 Hz  -> passes a 4 Hz lowpass
_HIGH = np.sin(2 * np.pi * 30 * _T)   # 30 Hz -> removed by a 4 Hz lowpass


@pytest.mark.parametrize('kind', ['lp', 'hp'])
def test_fft_filter_isolates_the_right_band_1d(kind):
    out = fft_filter(_LOW + _HIGH, _FS, 4, kind)
    keep, drop = (_LOW, _HIGH) if kind == 'lp' else (_HIGH, _LOW)
    # the kept component survives; the rejected one is gone
    assert np.corrcoef(out, keep)[0, 1] > 0.99
    assert np.abs(fft_filter(drop, _FS, 4, kind)).max() < 1e-6


@pytest.mark.parametrize('kind', ['lp', 'hp'])
def test_fft_filter_2d_is_per_channel(kind):
    X = np.vstack([_LOW, _HIGH])
    out = fft_filter(X, _FS, 4, kind)
    expected = np.vstack([fft_filter(_LOW, _FS, 4, kind),
                          fft_filter(_HIGH, _FS, 4, kind)])
    assert out.shape == X.shape
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_fft_filter_2d_lowpass_removes_out_of_band_channel():
    # regression: a 30 Hz channel must be annihilated by a 4 Hz lowpass, not passed through
    X = np.vstack([_LOW, _HIGH])
    out = fft_filter(X, _FS, 4, 'lp')
    assert np.abs(out[0]).max() > 0.9    # 2 Hz kept
    assert np.abs(out[1]).max() < 1e-6   # 30 Hz removed


def test_fft_filter_nd_matches_per_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2, 3, _T.size))
    out = fft_filter(X, _FS, 4, 'hp')
    expected = np.stack([[fft_filter(X[i, j], _FS, 4, 'hp') for j in range(3)]
                         for i in range(2)])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_fft_filter_cutoff_at_or_above_nyquist_does_not_raise():
    np.testing.assert_allclose(fft_filter(_LOW, _FS, 150, 'lp'), _LOW, atol=1e-9)
    np.testing.assert_allclose(fft_filter(_LOW, _FS, 150, 'hp'), 0.0, atol=1e-9)


def test_fft_filter_rejects_unknown_type():
    with pytest.raises(ValueError):
        fft_filter(_LOW, _FS, 4, 'bandpass')


def test_fft_filter_does_not_mutate_input():
    x = (_LOW + _HIGH).copy()
    before = x.copy()
    fft_filter(x, _FS, 4, 'lp')
    np.testing.assert_array_equal(x, before)
