import time
import warnings

import numpy as np
import pytest
from scipy import signal as ss
from scipy.ndimage import uniform_filter1d

from brainmaze_utils.gaps import (find_gaps, gap_intervals, fill_gaps, pink_noise,
                                  mask_in_gaps, drop_in_gaps)

FS = 250.0
METHODS = ['spectral', 'pink', 'mirror', 'linear']


def _eeg_like(n_s=120, seed=0):
    """Pink background (~30 uV) with a DC offset, as a stand-in for real EEG."""
    rng = np.random.default_rng(seed)
    return 30 * pink_noise(int(n_s * FS), rng=rng) + 100.0


def _brown_white(fs, n_s, seed=0):
    """1/f^2 background + 10 Hz rhythm + white floor (more realistic than pure 1/f)."""
    r = np.random.default_rng(seed)
    n = int(n_s * fs)
    b = np.cumsum(r.standard_normal(n))
    b = ss.sosfiltfilt(ss.butter(2, 0.5, 'high', fs=fs, output='sos'), b)
    t = np.arange(n) / fs
    return b / b.std() * 30 + 10 * np.sin(2 * np.pi * 10 * t) + 3 * r.standard_normal(n)


def _with_gaps(x, spans_s, fs=FS):
    y = x.copy()
    for a, b in spans_s:
        y[int(a * fs):int(b * fs)] = np.nan
    return y


def _band(y, fs, lo, hi):
    return ss.sosfiltfilt(ss.butter(4, [lo, hi], 'bandpass', fs=fs, output='sos'), y)


# ---------------------------------------------------------------- find / intervals
def test_find_gaps_runs_and_edges():
    x = np.array([np.nan, 1, 2, np.nan, np.nan, 3, np.nan])
    assert find_gaps(x).tolist() == [[0, 1], [3, 5], [6, 7]]
    assert find_gaps(np.arange(5.0)).shape == (0, 2)
    assert find_gaps(np.full(3, np.nan)).tolist() == [[0, 3]]
    assert find_gaps(np.arange(5)).dtype == np.int64


def test_find_gaps_treats_inf_as_gap():                               # R5
    x = np.array([0.0, np.inf, np.nan, 1.0, -np.inf, 2.0])
    assert find_gaps(x).tolist() == [[1, 3], [4, 5]]


def test_find_gaps_rejects_2d():
    with pytest.raises(ValueError):
        find_gaps(np.zeros((2, 10)))


def test_gap_intervals_seconds():
    x = _with_gaps(np.zeros(1000), [(1.0, 2.0)])
    assert gap_intervals(x, FS).tolist() == [[1.0, 2.0]]
    with pytest.raises(ValueError):
        gap_intervals(x, np.nan)


# ---------------------------------------------------------------- pink noise
def test_pink_noise_spectrum_slope_and_normalisation():
    y = pink_noise(2 ** 16, beta=1.0, rng=np.random.default_rng(1))
    assert abs(y.mean()) < 1e-12 and y.std() == pytest.approx(1.0)
    p = np.abs(np.fft.rfft(y)) ** 2
    k = np.arange(p.size)
    sel = (k > 10) & (k < p.size // 2)
    slope = np.polyfit(np.log(k[sel]), np.log(p[sel]), 1)[0]
    assert slope == pytest.approx(-1.0, abs=0.1)


# ---------------------------------------------------------------- fill: basics
@pytest.mark.parametrize('method', METHODS)
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
    x[60] = np.nan
    x[0:3] = np.nan                                         # leading: constant extension
    y = fill_gaps(x, FS)
    assert np.allclose(y[3:], np.arange(3.0, 100.0))
    assert np.allclose(y[:3], 3.0)


def test_short_gap_threshold_is_not_rounded_up():
    x = np.arange(1000.0)
    x[100:126] = np.nan                                     # 26 samples = 0.1016 s > 0.1 s
    y = fill_gaps(x, 256.0, method='mirror')
    assert not np.allclose(y[100:126], np.arange(100.0, 126.0))
    x[100:126] = np.arange(100.0, 126.0)
    x[100:125] = np.nan                                     # 25 samples = 0.0977 s
    assert np.allclose(fill_gaps(x, 256.0), np.arange(1000.0))


@pytest.mark.parametrize('method', ['spectral', 'pink', 'mirror'])
def test_long_gap_fill_has_background_amplitude_and_level(method):
    x0 = _eeg_like(seed=3)
    x = _with_gaps(x0, [(50, 60)])
    y = fill_gaps(x, FS, method=method)
    seg = y[int(51 * FS):int(59 * FS)]                      # away from the tapers
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


@pytest.mark.parametrize('method', METHODS)
def test_leading_trailing_adjacent_and_tiny_context_gaps(method):
    x = _with_gaps(_eeg_like(), [(0, 1.5), (20, 22), (22.004, 25), (30, 31), (31.008, 33),
                                 (118, 120)])
    y = fill_gaps(x, FS, method=method)
    assert np.isfinite(y).all()
    assert np.array_equal(y[np.isfinite(x)], x[np.isfinite(x)])


def test_fill_does_not_modify_input_and_is_reproducible():
    x = _with_gaps(_eeg_like(), [(50, 55)])
    x_copy = x.copy()
    a, b = fill_gaps(x, FS), fill_gaps(x, FS)
    assert np.array_equal(a, b)
    assert np.array_equal(x, x_copy, equal_nan=True)
    assert not np.array_equal(a, fill_gaps(x, FS, seed=1))


def test_multichannel_axis():
    x = np.stack([_with_gaps(_eeg_like(seed=s), [(10, 12)]) for s in range(3)])
    y = fill_gaps(x, FS, axis=-1)
    assert y.shape == x.shape and np.isfinite(y).all()
    yt = fill_gaps(x.T, FS, axis=0)
    assert np.array_equal(yt.T, y)
    x3 = np.stack([x, x[::-1]])                             # (2, 3, n)
    y3 = fill_gaps(x3, FS)
    assert np.array_equal(y3[0], y) and np.isfinite(y3).all()


# ---------------------------------------------------------------- R1: spectrum
@pytest.mark.parametrize('gap_s', [2.0, 10.0])
def test_spectral_fill_matches_neighbour_band_power(gap_s):
    fs = 2000.0
    x0 = _brown_white(fs, 120, seed=4)
    bands = [(2, 8), (10, 60), (80, 500), (500, 900)]
    ratios = {b: [] for b in bands}
    for k, s in enumerate((15.0, 30.0, 45.0, 60.0, 75.0, 90.0)):
        x = _with_gaps(x0, [(s, s + gap_s)], fs)
        y = fill_gaps(x, fs, seed=k)
        core = slice(int((s + 0.5) * fs), int((s + gap_s - 0.5) * fs))
        nb = np.r_[np.arange(int((s - 10) * fs), int((s - 1) * fs)),
                   np.arange(int((s + gap_s + 1) * fs), int((s + gap_s + 10) * fs))]
        for b in bands:
            f = _band(y, fs, *b)
            ratios[b].append(np.sqrt(np.mean(f[core] ** 2) / np.mean(f[nb] ** 2)))
    for b in bands:      # mean of 6 gaps: 0.94-1.17 over 15 seeds; pink is ~5 at 80-900 Hz
        assert 0.8 < np.mean(ratios[b]) < 1.25, (b, ratios[b])
    # documented contrast: the fixed 1/f shape puts too much power at high frequencies
    x = _with_gaps(x0, [(50, 50 + gap_s)], fs)
    f = _band(fill_gaps(x, fs, method='pink'), fs, 80, 500)
    core = slice(int(50.5 * fs), int((50 + gap_s - 0.5) * fs))
    assert np.sqrt(np.mean(f[core] ** 2)) / np.std(f[int(30 * fs):int(49 * fs)]) > 2


def test_spectral_fill_does_not_inflate_rms_background_next_to_gap():
    fs = 2000.0
    x0 = _brown_white(fs, 100, seed=5)
    sos = ss.butter(4, [80, 500], 'bandpass', fs=fs, output='sos')

    def thr(y):
        return np.sqrt(uniform_filter1d(ss.sosfiltfilt(sos, y) ** 2, int(10 * fs)))

    s, e = int(40 * fs), int(50 * fs)
    x = x0.copy()
    x[s:e] = np.nan
    idx = np.r_[np.arange(s - int(4 * fs), s - int(0.1 * fs)),
                np.arange(e + int(0.1 * fs), e + int(4 * fs))]
    assert np.median(thr(fill_gaps(x, fs))[idx] / thr(x0)[idx]) == pytest.approx(1.0, abs=0.1)
    assert np.median(thr(fill_gaps(x, fs, method='pink'))[idx] / thr(x0)[idx]) > 1.5


def test_spectral_fill_is_robust_to_spikes_in_context():
    fs = 500.0
    x0 = _brown_white(fs, 60, seed=6)
    x = x0.copy()
    for c in (23.3, 27.9):                                  # two big spikes in the context
        x[int(c * fs):int(c * fs) + 10] += 400.0
    x[int(30 * fs):int(35 * fs)] = np.nan
    f = _band(fill_gaps(x, fs), fs, 10, 60)
    core = f[int(30.5 * fs):int(34.5 * fs)]
    clean = _band(x0, fs, 10, 60)[int(36 * fs):int(46 * fs)]
    assert np.std(core) < 1.3 * np.std(clean)


def test_spectral_fill_variance_white_noise():
    x = np.random.default_rng(0).standard_normal(200000)
    x[50000:150000] = np.nan
    y = fill_gaps(x, 1000.0)
    assert np.std(y[51000:149000]) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------- R3: randomness
def _two_channels_shared_gap():
    rng = np.random.default_rng(2)
    a, b = rng.standard_normal(20000), 3 * rng.standard_normal(20000)
    a[5000:10000] = np.nan
    b[5000:10000] = np.nan
    return a, b


@pytest.mark.parametrize('method', ['spectral', 'pink'])
def test_separate_calls_with_same_seed_give_independent_noise(method):
    a, b = _two_channels_shared_gap()
    fa, fb = fill_gaps(a, 500, method=method), fill_gaps(b, 500, method=method)
    mid = slice(5300, 9700)
    # was 0.9997 (identical noise). Independent fills: |corr| <= 0.05 (spectral) and
    # <= 0.30 (pink, few effective degrees of freedom) over 200 seeds
    assert abs(np.corrcoef(fa[mid], fb[mid])[0, 1]) < (0.1 if method == 'spectral' else 0.5)


def test_channel_fill_independent_of_other_channels_order_and_gaps():
    a, b = _two_channels_shared_gap()
    F = fill_gaps(np.stack([a, b]), 500)
    assert np.array_equal(F[0], fill_gaps(a, 500))          # 1-D == row of N-D
    assert np.array_equal(F[1], fill_gaps(np.stack([b, a]), 500)[0])
    c = a.copy()
    c[1000:1100] = np.nan                                   # extra gap in the other channel
    assert np.array_equal(F[1], fill_gaps(np.stack([c, b]), 500)[1])
    d = b.copy()
    d[17000:18000] = np.nan                                 # extra gap far away, same channel
    assert np.array_equal(F[1][:12000], fill_gaps(d, 500)[:12000])


def test_seed_types():
    a, _ = _two_channels_shared_gap()
    ref = fill_gaps(a, 500, seed=7)
    assert np.array_equal(ref, fill_gaps(a, 500, seed=np.random.SeedSequence(7)))
    g1 = fill_gaps(a, 500, seed=np.random.default_rng(1))
    g2 = fill_gaps(a, 500, seed=np.random.default_rng(1))
    assert np.array_equal(g1, g2)
    assert not np.array_equal(fill_gaps(a, 500, seed=None), fill_gaps(a, 500, seed=None))
    assert not np.array_equal(ref, fill_gaps(a, 500, seed=[7, 1]))
    with pytest.raises(TypeError):
        fill_gaps(a, 500, seed=np.random.RandomState(0))


# ---------------------------------------------------------------- R4: speed
def test_many_short_gaps_at_high_fs_are_fast():
    fs = 32000
    x = np.random.default_rng(0).standard_normal(int(120 * fs))
    idx = np.random.default_rng(1).choice(x.size - 10, 5000, replace=False)
    for i in idx:
        x[i:i + 3] = np.nan
    t = time.perf_counter()
    y = fill_gaps(x, fs)
    assert time.perf_counter() - t < 2.0                    # was ~10 s (O(fs*context) per gap)
    assert np.isfinite(y).all()
    for i in idx[:50]:
        if np.isfinite(x[i - 1]) and np.isfinite(x[i + 3]):
            assert np.allclose(y[i:i + 3],
                               x[i - 1] + (x[i + 3] - x[i - 1]) * np.arange(1, 4) / 4)


# ---------------------------------------------------------------- R5: inf
@pytest.mark.parametrize('method', METHODS)
def test_inf_is_filled(method):
    x = _eeg_like(20)
    x[1000] = np.inf
    x[2000:2600] = -np.inf
    x[2600:2700] = np.nan
    y = fill_gaps(x, FS, method=method)
    assert np.isfinite(y).all()
    assert fill_gaps(np.array([0., 1., np.inf, np.nan, np.nan, 2., 3.]), 100).tolist() == \
        [0., 1., 1.25, 1.5, 1.75, 2., 3.]


# ---------------------------------------------------------------- R6: smooth edges
@pytest.mark.parametrize('method', ['spectral', 'pink', 'mirror'])
def test_long_gap_edges_continue_value_and_slope(method):
    fs = 5000.0
    t = np.arange(int(20 * fs)) / fs
    x0 = 50 * np.sin(2 * np.pi * 3 * t + 0.3)               # smooth, oversampled
    x = x0.copy()
    s, e = int(8 * fs), int(10 * fs)
    x[s:e] = np.nan
    y = fill_gaps(x, fs, method=method)
    d2_data = np.max(np.abs(np.diff(x0, 2)))
    for i in (s - 1, s, e - 1, e):                           # 2nd difference at the junctions
        assert abs(y[i - 1] - 2 * y[i] + y[i + 1]) < 3 * d2_data
    assert (y[s] - y[s - 1]) == pytest.approx(x0[s - 1] - x0[s - 2], rel=0.05)
    assert (y[e] - y[e - 1]) == pytest.approx(x0[e + 1] - x0[e], rel=0.05)


def test_taper_does_not_dip_band_power():
    fs = 1000.0
    x0 = np.random.default_rng(3).standard_normal(int(60 * fs))
    rat = []
    for k, s in enumerate(range(10, 50, 4)):
        x = x0.copy()
        x[int(s * fs):int((s + 2) * fs)] = np.nan
        y = fill_gaps(x, fs, seed=k)
        rat.append(np.std(y[int((s + 0.15) * fs):int((s + 0.35) * fs)]))  # mid-taper
    assert np.median(rat) == pytest.approx(1.0, abs=0.15)


# ---------------------------------------------------------------- R7: mirror amplitude
def test_mirror_does_not_overshoot_coherent_line():
    fs = 5000.0
    t = np.arange(int(30 * fs)) / fs
    r = np.random.default_rng(0)
    x0 = 5 * np.sin(2 * np.pi * 60 * t) + 0.5 * r.standard_normal(t.size)
    rat = []
    for s in np.arange(5, 25, 1.37):
        x = x0.copy()
        a, b = int(s * fs), int((s + 0.2) * fs)
        x[a:b] = np.nan
        f = _band(fill_gaps(x, fs, method='mirror'), fs, 55, 65)
        rat.append(np.std(f[a + 100:b - 100]) / np.std(f[a - 5000:a - 500]))
    assert np.median(rat) < 1.25                            # up to 1.48 without correction


# ---------------------------------------------------------------- R8: all-NaN channels
def test_all_nan_channel_warns_with_indices_and_modes():
    x = np.vstack([np.full(100, np.nan), np.ones(100), np.full(100, np.inf)])
    with pytest.warns(RuntimeWarning, match=r'2 channel\(s\) \[0, 2\]'):
        y = fill_gaps(x, FS)
    assert np.isnan(y[0]).all() and np.array_equal(y[1], x[1])
    with pytest.warns(RuntimeWarning):
        z = fill_gaps(x, FS, all_nan='zero')
    assert np.array_equal(z[0], np.zeros(100)) and np.array_equal(z[2], np.zeros(100))
    with pytest.raises(ValueError, match=r'\[0, 2\]'):
        fill_gaps(x, FS, all_nan='raise')
    with pytest.warns(RuntimeWarning):
        assert np.isnan(fill_gaps(np.full(100, np.nan), FS)).all()


# ---------------------------------------------------------------- R9 + Copilot: validation
def test_invalid_arguments():
    x = _with_gaps(_eeg_like(), [(10, 12)])
    for fs in (0, -1, np.nan, np.inf, 'abc'):
        with pytest.raises(ValueError):
            fill_gaps(x, fs)
    with pytest.raises(ValueError):
        fill_gaps(x, FS, method='zeros')
    for kw in (dict(max_interp_s=-1), dict(context_s=0), dict(taper_s=np.nan),
               dict(beta=np.inf), dict(all_nan='drop'), dict(axis=3)):
        with pytest.raises(ValueError):
            fill_gaps(x, FS, **kw)


def test_method_validated_even_without_long_gaps():          # Copilot
    for x in (np.arange(10.0), np.array([1.0, np.nan, 2.0])):
        with pytest.raises(ValueError):
            fill_gaps(x, 100, method='bogus')


# ---------------------------------------------------------------- R10: dtype / copies
def test_dtype_preserved_and_in_place():
    x = _with_gaps(_eeg_like(), [(10, 12), (20, 20.02)]).astype(np.float32)
    y = fill_gaps(x, FS)
    assert y.dtype == np.float32 and np.isfinite(y).all() and np.isnan(x).any()
    z = fill_gaps(x, FS, copy=False)
    assert z is x and np.isfinite(x).all()
    assert np.array_equal(z, y)
    assert fill_gaps(np.arange(5), 10).dtype == np.float64
    clean = np.arange(10.0)
    assert fill_gaps(clean, 10, copy=False) is clean
    out = fill_gaps(clean, 10)
    assert out is not clean and np.array_equal(out, clean)


# ---------------------------------------------------------------- post-processing (R2, R9)
def test_mask_in_gaps_points_with_margin_seconds_and_samples():
    gaps = np.array([[500, 750]])                           # 2.0 - 3.0 s at 250 Hz
    t = np.array([1.85, 1.95, 2.5, 2.99, 3.05, 3.15])
    expect = [False, True, True, True, True, False]
    assert mask_in_gaps(t, gaps / FS, FS, units='seconds', margin_s=0.1).tolist() == expect
    assert mask_in_gaps(np.round(t * FS).astype(int), gaps, FS, units='samples',
                        margin_s=0.1).tolist() == expect
    assert mask_in_gaps(t, gaps / FS, FS, units='seconds', margin_s=0.0).tolist() == [
        False, False, True, True, False, False]


def test_units_are_required_and_mixups_raise():                # R2
    gaps = np.array([[100000, 100500]])
    with pytest.raises(TypeError):
        mask_in_gaps([200.5], gaps, 500.0)                  # no units
    with pytest.raises(ValueError):
        mask_in_gaps([200.5], gaps, 500.0, units='seconds')  # sample gaps as seconds
    with pytest.raises(ValueError):
        mask_in_gaps([200.5], gaps, 500.0, units='samples')  # seconds detections as samples
    with pytest.raises(ValueError):
        mask_in_gaps([100250], gaps / 500.3, 500.0, units='samples')
    with pytest.raises(ValueError):
        mask_in_gaps([1.0], gaps / 500.0, 500.0, units='hours')
    # eeg_forge Janca returns samples: 10 samples (20 ms) outside the gap must be dropped
    assert mask_in_gaps(np.array([99990, 100510]), gaps, 500.0, units='samples').all()


def test_mask_in_gaps_intervals_overlap():
    gaps_s = np.array([[2.0, 3.0], [10.0, 10.5]])
    start = np.array([1.0, 1.5, 9.0, 11.0])
    end = np.array([1.5, 2.2, 12.0, 11.5])
    assert mask_in_gaps(start, gaps_s, FS, units='seconds', margin_s=0, end=end).tolist() == [
        False, True, True, False]


def test_mask_handles_unsorted_and_nested_gaps_and_empty():
    gaps_s = np.array([[10.0, 20.0], [1.0, 2.0], [12.0, 13.0]])
    assert mask_in_gaps([15.0, 5.0, 1.5], gaps_s, FS, units='seconds',
                        margin_s=0).tolist() == [True, False, True]
    assert mask_in_gaps([], gaps_s, FS, units='seconds').size == 0
    assert mask_in_gaps([1.0], np.zeros((0, 2)), FS, units='seconds').tolist() == [False]
    assert mask_in_gaps([1], find_gaps(np.arange(5.0)), FS, units='samples').tolist() == [False]


def test_mask_validation():                                    # R9
    g = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, 100, units='seconds', margin_s=-0.6)
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, 100, units='seconds', end=[1.0])
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], np.array([[2.0, 1.0]]), 100, units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([np.nan], g, 100, units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, np.nan, units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], np.array([1.0, 2.0, 3.0]), 100, units='seconds')


def test_drop_in_gaps_points_and_intervals():
    gaps = find_gaps(_with_gaps(np.zeros(5000), [(4.0, 6.0)]))
    assert drop_in_gaps([1.0, 4.5, 5.95, 6.2, 19.0], gaps / FS, FS,
                        units='seconds').tolist() == [1.0, 6.2, 19.0]
    det = np.array([250, 1100, 1490, 1550], dtype=np.int64)
    out = drop_in_gaps(det, gaps, FS, units='samples')
    assert out.tolist() == [250, 1550] and out.dtype == np.int64
    s, e = drop_in_gaps([0.5, 1.5], np.array([[1.0, 2.0]]), 100, units='seconds',
                        end=[0.6, 1.6])
    assert s.tolist() == [0.5] and e.tolist() == [0.6]


def test_full_pipeline_example():
    """find -> fill -> detect -> drop, as in the module docstring."""
    x = _with_gaps(_eeg_like(60, seed=9), [(10, 12), (30, 30.05)])
    gaps = find_gaps(x)
    y = fill_gaps(x, FS)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        det = ss.find_peaks(np.abs(_band(y, FS, 10, 60)), distance=int(0.1 * FS))[0]
    kept = drop_in_gaps(det, gaps, FS, units='samples', margin_s=0.1)
    assert kept.size > 0
    for s, e in gaps:
        assert not np.any((kept >= s - 0.1 * FS) & (kept < e + 0.1 * FS))
