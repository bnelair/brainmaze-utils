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
    x0 = _eeg_like(n_s=400)
    starts = np.arange(20, 380, 12.0)
    x = _with_gaps(x0, [(a, a + gap_s) for a in starts])
    y = fill_gaps(x, FS, method=method)
    assert np.isfinite(y).all()
    valid = np.isfinite(x)
    assert np.array_equal(y[valid], x[valid])               # never touches real data
    typical = np.median(np.abs(np.diff(x0)))
    g = find_gaps(x)
    # the junctions are statistically like the data's own sample-to-sample steps (a
    # conditional simulation of the process), no systematic step into or out of the gap
    jump = np.abs(np.r_[y[g[:, 0]] - y[g[:, 0] - 1], y[g[:, 1]] - y[g[:, 1] - 1]])
    # ('pink': its fixed 1/f model puts the context's variance into >= 1/gap and is rougher
    # than the data, so its junctions are too; still no systematic step)
    assert np.median(jump) <= (2.5 if method == 'pink' else 1.5) * typical
    assert np.max(jump) <= (8 if method == 'pink' else 6) * typical


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
    for c in (21.1, 23.3, 25.4, 27.9):                      # big spikes in the context
        x[int(c * fs):int(c * fs) + 10] += 400.0
    x[int(30 * fs):int(35 * fs)] = np.nan
    clean = np.std(_band(x0, fs, 10, 60)[int(36 * fs):int(46 * fs)])
    r = [np.std(_band(fill_gaps(x, fs, seed=k), fs, 10, 60)[int(30.5 * fs):int(34.5 * fs)])
         for k in range(4)]
    # 1.00-1.09 with the rejection, 1.74-1.82 without (scratch/utils-gaps/r3/v3_rules.py)
    assert np.mean(r) < 1.2 * clean


def test_spectral_fill_variance_white_noise():
    x = np.random.default_rng(0).standard_normal(200000)
    x[50000:150000] = np.nan
    y = fill_gaps(x, 1000.0)
    assert np.std(y[51000:149000]) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------- round 3 (V1-V5, V8, V10)
def test_synthesis_has_exact_power_per_bin():
    """Random-phase synthesis: the mean square equals the requested power (Parseval)."""
    import brainmaze_utils.gaps as G
    rng = np.random.default_rng(0)
    for m in (64, 65):
        power = rng.random(m // 2 + 1)
        power[0] = 0.0
        z = G._synthesize(power, m, rng)
        assert np.mean(z ** 2) == pytest.approx(power.sum(), rel=1e-9)
        only_nyq = np.zeros(m // 2 + 1)
        only_nyq[-1] = 2.0
        assert np.mean(G._synthesize(only_nyq, m, rng) ** 2) == pytest.approx(
            2.0 if m % 2 == 0 else 2.0, rel=1e-9)


def test_noisy_edges_do_not_inject_low_frequency_power():      # V1
    """White-dominated data: the edge must not carry one noisy sample's offset into the
    gap (was 181x the 1-4 Hz power in the first 0.5 s of the fill)."""
    fs = 500.0
    x0 = ss.sosfiltfilt(ss.butter(2, 0.5, 'high', fs=fs, output='sos'),
                        10 * np.random.default_rng(1).standard_normal(int(400 * fs)))
    starts = np.arange(20, 380, 9.1)
    y = fill_gaps(_with_gaps(x0, [(a, a + 5) for a in starts], fs), fs)
    fy, fo = _band(y, fs, 1, 4), _band(x0, fs, 1, 4)
    w = np.concatenate([np.arange(int(a * fs), int((a + 0.5) * fs)) for a in starts])
    assert np.mean(fy[w] ** 2) / np.mean(fo[w] ** 2) < 2.0
    # and nothing leaks into the valid data 0.1-1 s before the gaps
    leak = [np.max(np.abs(fy[i:i + int(0.9 * fs)] - fo[i:i + int(0.9 * fs)]))
            for i in (int((a - 1) * fs) for a in starts)]
    assert np.median(leak) < 1.0 * np.std(fo)               # was 6.1 (p90 16.9)


def test_artifact_at_edge_is_not_continued_into_gap():         # V1
    fs = 500.0
    x0 = _brown_white(fs, 120, seed=3)
    sd = np.std(x0)
    big = []
    for k, s in enumerate(np.arange(20, 100, 7.7)):
        x = x0.copy()
        a, e = int(s * fs), int((s + 5) * fs)
        x[a - 10:a] += 300 * np.hanning(20)[:10]            # artifact cut by the dropout
        x[a:e] = np.nan
        y = fill_gaps(x, fs, seed=k)
        big.append(np.max(np.abs(y[a:a + int(0.5 * fs)] - np.median(x0[a - 250:a]))) / sd)
    assert np.median(big) < 6.0                             # was 19.9 (2x the artifact)


def test_slow_oscillation_is_not_smeared_into_delta():         # V2
    fs = 250.0
    t = np.arange(int(1200 * fs)) / fs
    x0 = 40 * np.sin(2 * np.pi * 0.75 * t) + 5 * np.random.default_rng(4).standard_normal(t.size)
    starts = np.arange(100, 1100, 90.0)
    y = fill_gaps(_with_gaps(x0, [(a, a + 30) for a in starts], fs), fs)
    core = np.concatenate([np.arange(int((a + 2) * fs), int((a + 28) * fs)) for a in starts])

    def ratio(lo, hi):
        b = ss.butter(2, [lo, hi], 'bandpass', fs=fs, output='sos')
        return np.sqrt(np.mean(ss.sosfiltfilt(b, y)[core] ** 2)
                       / np.mean(ss.sosfiltfilt(b, x0)[core] ** 2))
    assert 0.7 < ratio(0.6, 1.0) < 1.3                      # was 0.39 (1 s Welch cap)
    assert ratio(1, 2) < 2.0                                # was 6.5
    assert ratio(2, 4) < 1.3                                # was 7.4


def test_short_gaps_keep_delta_band():                         # 1 s minimum Welch segment
    """A 0.2 s gap still gets the delta-band continuation of its neighbours (the spectrum
    model includes it; with a 0.25 s Welch segment it had none: 0.36)."""
    fs = 250.0
    x0 = _brown_white(fs, 400, seed=12)
    st = np.arange(20, 380, 6.1)
    f = _band(fill_gaps(_with_gaps(x0, [(a, a + 0.2) for a in st], fs), fs), fs, 0.5, 4)
    r = []
    for a in st:
        s, e = int(a * fs), int(a * fs) + int(0.2 * fs)
        nb = np.r_[np.arange(s - int(10 * fs), s - int(fs)), np.arange(e + int(fs), e + int(10 * fs))]
        r.append(np.sqrt(np.mean(f[s + 12:e - 12] ** 2) / np.mean(f[nb] ** 2)))
    assert np.median(r) > 0.47                              # 0.57 (0.36 without)


def test_bursts_in_context_are_not_rejected_as_artifacts():    # V3
    fs = 500.0
    T = 600
    t = np.arange(int(T * fs)) / fs
    r = np.random.default_rng(2)
    env = np.zeros(t.size)
    for c in np.arange(1, T - 2, 6.0) + r.uniform(0, 3, int(np.ceil((T - 3) / 6.0))):
        i0, m = int(c * fs), int(1.0 * fs)
        if i0 + m <= env.size:
            env[i0:i0 + m] = np.hanning(m)
    x0 = _brown_white(fs, T, seed=21) + 50 * env * np.sin(2 * np.pi * 13 * t)   # spindles
    num = den = 0.0
    for k, s in enumerate(np.arange(15, T - 16, 13.0)):
        a, e = int(s * fs), int((s + 1) * fs)
        y = fill_gaps(_with_gaps(x0, [(s, s + 1)], fs), fs, seed=k)
        fy, fo = _band(y, fs, 11, 15), _band(x0, fs, 11, 15)
        num += np.mean(fy[a + 125:e - 125] ** 2)
        den += np.mean(np.r_[fo[a - 5000:a], fo[e:e + 5000]] ** 2)
    # 0.93 with the two-band rule; 0.66 with "any band > 3 MAD" (v3_rules.py)
    assert np.sqrt(num / den) > 0.85


def test_interpolated_short_gaps_do_not_bias_context():        # V5
    fs = 1000.0
    x0 = np.random.default_rng(5).standard_normal(int(200 * fs))
    x = x0.copy()
    for a in np.arange(1, 199, 0.1):                        # 50 % packet loss, 0.05 s each
        x[int(a * fs):int((a + 0.05) * fs)] = np.nan
    starts = np.arange(20, 180, 20.0)
    for a in starts:
        x[int(a * fs):int((a + 2) * fs)] = np.nan
    y = fill_gaps(x, fs)
    core = np.concatenate([np.arange(int((a + 0.3) * fs), int((a + 1.7) * fs)) for a in starts])
    f = _band(y, fs, 100, 400)
    ref = _band(x0, fs, 100, 400)
    assert np.sqrt(np.mean(f[core] ** 2) / np.mean(ref ** 2)) > 0.9    # was 0.70


def test_fill_is_invariant_to_ulps_offset_and_units():         # V8
    x0 = _brown_white(500.0, 120, seed=8)
    x = _with_gaps(x0, [(50, 55)], 500.0)
    y = fill_gaps(x, 500.0)[25000:27500]
    x1 = x.copy()
    x1[24000] = np.nextafter(x1[24000], np.inf)             # 1 ulp in the context
    assert np.allclose(fill_gaps(x1, 500.0)[25000:27500], y)
    x2 = np.r_[np.random.default_rng(1).standard_normal(10000), x]   # 20 s later
    assert np.allclose(fill_gaps(x2, 500.0)[35000:37500], y)
    assert np.allclose(fill_gaps(x * 1e-6, 500.0)[25000:27500] * 1e6, y)   # V vs uV
    y3 = fill_gaps(_with_gaps(x0, [(50, 55), (80, 85)], 500.0), 500.0)  # other gap: other noise
    f3 = _band(y3, 500.0, 10, 60)
    assert abs(np.corrcoef(f3[25200:27300], f3[40200:42300])[0, 1]) < 0.1


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
    d[17000:18000] = np.nan                    # extra gap outside the context, same channel
    assert np.array_equal(fill_gaps(b, 500, context_s=10)[:12000],
                          fill_gaps(d, 500, context_s=10)[:12000])


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
@pytest.mark.parametrize('method', ['spectral', 'mirror'])   # 'pink' has its own (rough) model
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
    # no step where the edge conditioning ends inside a long gap (faded out, not cut)
    assert np.max(np.abs(np.diff(y[s - 1:e + 1]))) < 3 * np.max(np.abs(np.diff(x0)))


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
    # 0.98 with the correlation-aware correction; 1.20 with rho ignored (the pre-R7
    # sqrt-compensation), up to 1.48 without any (scratch/review-utils26-round2/a9_mirror_test)
    assert np.median(rat) < 1.1


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
    assert mask_in_gaps(t, gaps / FS, FS, units='seconds', gap_units='seconds', margin_s=0.1).tolist() == expect
    assert mask_in_gaps(np.round(t * FS).astype(int), gaps, FS, units='samples', gap_units='samples',
                        margin_s=0.1).tolist() == expect
    assert mask_in_gaps(t, gaps / FS, FS, units='seconds', gap_units='seconds', margin_s=0.0).tolist() == [
        False, False, True, True, False, False]


def test_units_are_required_and_mixups_raise():                # R2, V6, V7
    gaps = np.array([[100000, 100500]])                     # samples, 200.0-201.0 s at 500 Hz
    with pytest.raises(TypeError):
        mask_in_gaps([200.5], gaps, 500.0)                  # no units
    with pytest.raises(TypeError):
        mask_in_gaps([200.5], gaps, 500.0, units='seconds')  # no gap_units
    with pytest.raises(ValueError):                         # sample gaps declared seconds
        mask_in_gaps([200.5], gaps, 500.0, units='seconds', gap_units='seconds')
    with pytest.raises(ValueError):                         # seconds detections as samples
        mask_in_gaps([200.5], gaps, 500.0, units='samples', gap_units='samples')
    with pytest.raises(ValueError):
        mask_in_gaps([100250], gaps / 500.3, 500.0, units='samples', gap_units='samples')
    with pytest.raises(ValueError):
        mask_in_gaps([1.0], gaps / 500.0, 500.0, units='hours', gap_units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([1.0], gaps / 500.0, 500.0, units='seconds', gap_units='hours')
    # V6: integer sample detections declared as seconds raise (was: silently nothing masked)
    with pytest.raises(ValueError, match='integer'):
        mask_in_gaps(np.array([99990, 100510]), gaps / 500.0, 500.0, units='seconds',
                     gap_units='seconds')
    # sample detections (int or float) against seconds gaps: state each unit, both work
    for det in (np.array([99990, 100510]), np.array([99990.0, 100510.0])):
        assert mask_in_gaps(det, gaps / 500.0, 500.0, units='samples',
                            gap_units='seconds').all()
    # eeg_forge Janca returns samples: 10 samples (20 ms) outside the gap must be dropped
    assert mask_in_gaps(np.array([99990, 100510]), gaps, 500.0, units='samples',
                        gap_units='samples').all()
    # seconds detections against sample gaps
    assert mask_in_gaps([199.98, 201.02, 205.0], gaps, 500.0, units='seconds',
                        gap_units='samples').tolist() == [True, True, False]


def test_units_integral_tolerance_and_whole_seconds():         # V7
    fs = 5000.0
    k = np.arange(1, 200000, 7)
    t = k / fs
    assert np.any(t * fs != k)                              # unrounded t*fs is not integral...
    m = mask_in_gaps(t * fs, np.array([[k[10], k[20]]]), fs, units='samples',
                     gap_units='samples', margin_s=0.0)
    assert m.sum() == 10                                    # ...but accepted as samples
    with pytest.raises(ValueError, match='non-integer'):
        mask_in_gaps(t * fs + 0.25, np.array([[0, 10]]), fs, units='samples',
                     gap_units='samples')
    # whole-second annotation gaps as Python ints: the message says how to pass them
    with pytest.raises(ValueError, match='as floats'):
        mask_in_gaps([200.5], [[200, 201]], 500.0, units='seconds', gap_units='seconds')
    assert mask_in_gaps([200.5], np.asarray([[200, 201]], float), 500.0, units='seconds',
                        gap_units='seconds').tolist() == [True]


def test_mask_boundary_is_half_open():                         # V10: stop + margin excluded
    g = np.array([[100, 200]])
    fs = 100.0
    # widened gap is [100 - 10, 200 + 10) samples with margin 0.1 s
    assert mask_in_gaps([89, 90, 209, 210], g, fs, units='samples', gap_units='samples',
                        margin_s=0.1).tolist() == [False, True, True, False]
    assert mask_in_gaps([0.0], g, fs, units='samples', gap_units='samples', margin_s=0.0,
                        end=[100]).tolist() == [True]       # interval touching the start
    assert mask_in_gaps([200], g, fs, units='samples', gap_units='samples',
                        margin_s=0.0).tolist() == [False]   # stop is exclusive


def test_mask_in_gaps_intervals_overlap():
    gaps_s = np.array([[2.0, 3.0], [10.0, 10.5]])
    start = np.array([1.0, 1.5, 9.0, 11.0])
    end = np.array([1.5, 2.2, 12.0, 11.5])
    assert mask_in_gaps(start, gaps_s, FS, units='seconds', gap_units='seconds', margin_s=0, end=end).tolist() == [
        False, True, True, False]


def test_mask_handles_unsorted_and_nested_gaps_and_empty():
    gaps_s = np.array([[10.0, 20.0], [1.0, 2.0], [12.0, 13.0]])
    assert mask_in_gaps([15.0, 5.0, 1.5], gaps_s, FS, units='seconds', gap_units='seconds',
                        margin_s=0).tolist() == [True, False, True]
    assert mask_in_gaps([], gaps_s, FS, units='seconds', gap_units='seconds').size == 0
    assert mask_in_gaps([1.0], np.zeros((0, 2)), FS, units='seconds', gap_units='seconds').tolist() == [False]
    assert mask_in_gaps([1], find_gaps(np.arange(5.0)), FS, units='samples', gap_units='samples').tolist() == [False]


def test_mask_validation():                                    # R9
    g = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, 100, units='seconds', gap_units='seconds', margin_s=-0.6)
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, 100, units='seconds', gap_units='seconds', end=[1.0])
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], np.array([[2.0, 1.0]]), 100, units='seconds', gap_units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([np.nan], g, 100, units='seconds', gap_units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], g, np.nan, units='seconds', gap_units='seconds')
    with pytest.raises(ValueError):
        mask_in_gaps([1.5], np.array([1.0, 2.0, 3.0]), 100, units='seconds', gap_units='seconds')


def test_drop_in_gaps_points_and_intervals():
    gaps = find_gaps(_with_gaps(np.zeros(5000), [(4.0, 6.0)]))
    assert drop_in_gaps([1.0, 4.5, 5.95, 6.2, 19.0], gaps / FS, FS,
                        units='seconds', gap_units='seconds').tolist() == [1.0, 6.2, 19.0]
    det = np.array([250, 1100, 1490, 1550], dtype=np.int64)
    out = drop_in_gaps(det, gaps, FS, units='samples', gap_units='samples')
    assert out.tolist() == [250, 1550] and out.dtype == np.int64
    s, e = drop_in_gaps([0.5, 1.5], np.array([[1.0, 2.0]]), 100, units='seconds', gap_units='seconds',
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
    kept = drop_in_gaps(det, gaps, FS, units='samples', gap_units='samples', margin_s=0.1)
    assert kept.size > 0
    for s, e in gaps:
        assert not np.any((kept >= s - 0.1 * FS) & (kept < e + 0.1 * FS))
