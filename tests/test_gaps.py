import numpy as np
import pytest

from brainmaze_utils.gaps import (find_gaps, gap_intervals, fill_gaps, pink_noise,
                                  mask_in_gaps, drop_in_gaps)

FS = 250.0


def _eeg_like(n_s=120, seed=0):
    """Pink background (~30 uV) with a DC offset, as a stand-in for real EEG."""
    rng = np.random.default_rng(seed)
    return 30 * pink_noise(int(n_s * FS), rng=rng) + 100.0


def _with_gaps(x, spans_s):
    y = x.copy()
    for a, b in spans_s:
        y[int(a * FS):int(b * FS)] = np.nan
    return y


# ---------------------------------------------------------------- find / intervals
def test_find_gaps_runs_and_edges():
    x = np.array([np.nan, 1, 2, np.nan, np.nan, 3, np.nan])
    assert find_gaps(x).tolist() == [[0, 1], [3, 5], [6, 7]]
    assert find_gaps(np.arange(5.0)).shape == (0, 2)
    assert find_gaps(np.full(3, np.nan)).tolist() == [[0, 3]]


def test_find_gaps_rejects_2d():
    with pytest.raises(ValueError):
        find_gaps(np.zeros((2, 10)))


def test_gap_intervals_seconds():
    x = _with_gaps(np.zeros(1000), [(1.0, 2.0)])
    assert gap_intervals(x, FS).tolist() == [[1.0, 2.0]]


# ---------------------------------------------------------------- pink noise
def test_pink_noise_spectrum_slope_and_normalisation():
    y = pink_noise(2 ** 16, beta=1.0, rng=np.random.default_rng(1))
    assert abs(y.mean()) < 1e-12 and y.std() == pytest.approx(1.0)
    p = np.abs(np.fft.rfft(y)) ** 2
    k = np.arange(p.size)
    sel = (k > 10) & (k < p.size // 2)
    slope = np.polyfit(np.log(k[sel]), np.log(p[sel]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)


# ---------------------------------------------------------------- fill
@pytest.mark.parametrize('method', ['pink', 'mirror', 'linear'])
@pytest.mark.parametrize('gap_s', [0.02, 0.5, 2.0, 10.0])
def test_fill_is_finite_keeps_valid_samples_and_has_no_edge_step(method, gap_s):
    x0 = _eeg_like()
    x = _with_gaps(x0, [(30, 30 + gap_s), (70, 70 + gap_s)])
    y = fill_gaps(x, FS, method=method)
    assert np.isfinite(y).all()
    valid = np.isfinite(x)
    assert np.array_equal(y[valid], x[valid])               # never touches real data
    typical = np.median(np.abs(np.diff(x0)))
    for s, e in find_gaps(x):
        assert abs(y[s] - y[s - 1]) <= 3 * typical           # continuous into the gap
        assert abs(y[e] - y[e - 1]) <= 3 * typical           # and out of it


def test_short_gaps_are_linearly_interpolated():
    x = np.arange(100.0)
    x[40:45] = np.nan                                       # 5 samples = 20 ms < 0.1 s
    y = fill_gaps(x, FS)
    assert np.allclose(y, np.arange(100.0))


@pytest.mark.parametrize('method', ['pink', 'mirror'])
def test_long_gap_fill_has_background_amplitude_and_level(method):
    x0 = _eeg_like(seed=3)
    x = _with_gaps(x0, [(50, 60)])
    y = fill_gaps(x, FS, method=method)
    seg = y[int(51 * FS):int(59 * FS)]                      # away from the cross-fades
    ref = np.r_[x0[int(40 * FS):int(50 * FS)], x0[int(60 * FS):int(70 * FS)]]
    level = 0.5 * (np.median(x0[int(49.5 * FS):int(50 * FS)]) +
                   np.median(x0[int(60 * FS):int(60.5 * FS)]))   # bridged edge levels
    assert np.std(seg) == pytest.approx(np.std(ref), rel=0.5)
    assert abs(np.median(seg) - level) < np.std(ref)
    assert np.std(seg) > 0.3 * np.std(ref)                 # not a flat line


def test_linear_fill_of_long_gap_is_flat_documented_behaviour():
    x = _with_gaps(_eeg_like(), [(50, 60)])
    y = fill_gaps(x, FS, method='linear')
    seg = y[int(50 * FS):int(60 * FS)]
    assert np.allclose(np.diff(seg, 2), 0, atol=1e-9)


def test_leading_trailing_and_adjacent_gaps():
    x = _with_gaps(_eeg_like(), [(0, 1.5), (20, 22), (22.01, 25), (118, 120)])
    for method in ('pink', 'mirror', 'linear'):
        y = fill_gaps(x, FS, method=method)
        assert np.isfinite(y).all()


def test_fill_is_reproducible_and_does_not_modify_input():
    x = _with_gaps(_eeg_like(), [(50, 55)])
    x_copy = x.copy()
    a, b = fill_gaps(x, FS), fill_gaps(x, FS)
    assert np.array_equal(a, b)
    assert np.array_equal(np.isnan(x), np.isnan(x_copy))
    c = fill_gaps(x, FS, seed=1)
    assert not np.array_equal(a, c)


def test_multichannel_axis():
    x = np.stack([_with_gaps(_eeg_like(seed=s), [(10, 12)]) for s in range(3)])
    y = fill_gaps(x, FS, axis=-1)
    assert y.shape == x.shape and np.isfinite(y).all()
    yt = fill_gaps(x.T, FS, axis=0)
    assert np.allclose(yt.T, y)


def test_all_nan_channel_warns_and_stays_nan():
    with pytest.warns(RuntimeWarning):
        y = fill_gaps(np.full(100, np.nan), FS)
    assert np.isnan(y).all()


def test_invalid_arguments():
    with pytest.raises(ValueError):
        fill_gaps(np.zeros(10), 0)
    with pytest.raises(ValueError):
        fill_gaps(_with_gaps(_eeg_like(), [(10, 12)]), FS, method='zeros')


# ---------------------------------------------------------------- post-processing
def test_mask_in_gaps_points_with_margin():
    gaps = np.array([[500, 750]])                           # 2.0 - 3.0 s at 250 Hz
    t = np.array([1.85, 1.95, 2.5, 2.99, 3.05, 3.15])
    assert mask_in_gaps(t, gaps, fs=FS, margin_s=0.1).tolist() == [
        False, True, True, True, True, False]
    assert mask_in_gaps(t, gaps / FS, margin_s=0.0).tolist() == [
        False, False, True, True, False, False]


def test_mask_in_gaps_intervals_overlap():
    gaps_s = np.array([[2.0, 3.0], [10.0, 10.5]])
    start = np.array([1.0, 1.5, 9.0, 11.0])
    end = np.array([1.5, 2.2, 12.0, 11.5])
    assert mask_in_gaps(start, gaps_s, margin_s=0, end_s=end).tolist() == [
        False, True, True, False]


def test_mask_handles_unsorted_and_nested_gaps_and_empty():
    gaps_s = np.array([[10.0, 20.0], [1.0, 2.0], [12.0, 13.0]])
    assert mask_in_gaps([15.0, 5.0, 1.5], gaps_s, margin_s=0).tolist() == [True, False, True]
    assert mask_in_gaps([], gaps_s).size == 0
    assert mask_in_gaps([1.0], np.zeros((0, 2))).tolist() == [False]


def test_drop_in_gaps():
    gaps = find_gaps(_with_gaps(np.zeros(5000), [(4.0, 6.0)]))
    assert drop_in_gaps([1.0, 4.5, 5.95, 6.2, 19.0], gaps, fs=FS).tolist() == [1.0, 6.2, 19.0]
