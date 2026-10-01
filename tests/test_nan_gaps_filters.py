# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NaN gaps in LowFrequencyFilter and fft_filter: NaN out exactly where NaN in, no jumps at gap edges,
and the decimate -> LowFrequencyFilter cascade on gappy data."""

import warnings

import numpy as np
import pytest

from brainmaze_utils.signal import LowFrequencyFilter, decimate, fft_filter

FS = 1000


def _eeg_like(seed=0, dur=120, fs=FS, offset=1500.0, drift=3.0):
    """1/f noise (SD 20) + offset + linear drift + a 10 Hz tone (A=10)."""
    rng = np.random.default_rng(seed)
    n = int(dur * fs)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    t = np.arange(n) / fs
    return 20 * x / x.std() + offset + drift * t + 10 * np.sin(2 * np.pi * 10 * t)


def _hp(**kw):
    return LowFrequencyFilter(fs=FS, cutoff=0.5, n_decimate=3, ftype='iir', n_order=3, filter_type='hp', **kw)


@pytest.mark.parametrize('ftype, kind', [('iir', 'hp'), ('iir', 'lp'), ('fir', 'hp')])
def test_lff_nan_exactly_where_input_nan(ftype, kind):
    x = _eeg_like(dur=60)
    x[:300] = np.nan            # leading
    x[10_000:10_003] = np.nan   # 3-sample dropout (bridged)
    x[20_000:25_000] = np.nan   # 5 s gap (split)
    x[-50:] = np.nan            # trailing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flt = LowFrequencyFilter(fs=FS, cutoff=0.5, n_decimate=3, ftype=ftype, filter_type=kind,
                                 n_order=3 if ftype == 'iir' else 101)
    y = flt(x)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))


def test_lff_all_nan_and_tiny_segments():
    hp = _hp()
    y = hp(np.full(5000, np.nan))
    assert np.isnan(y).all()
    x = _eeg_like(dur=20)
    x[::3000] = np.nan
    x[100:5000] = np.nan
    x[5001:9000] = np.nan  # a single valid sample between two long gaps
    y = hp(x)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(x))


def test_lff_long_gap_segments_filtered_like_records():
    # a gap longer than max_gap_fill splits the signal; each side is filtered exactly as if it
    # were a complete record (the record-edge method), independent of the other side
    x = _eeg_like(dur=120)
    a, b = 50_000, 60_000
    xg = x.copy()
    xg[a:b] = np.nan
    hp = _hp()
    y = hp(xg)
    np.testing.assert_allclose(y[:a], hp(x[:a]), atol=1e-9)
    np.testing.assert_allclose(y[b:], hp(x[b:]), atol=1e-9)


def test_lff_no_jump_at_gap_edges():
    # offset 1500 + drift: the old zero-padding edge handling would give errors ~750 at every gap edge
    x = _eeg_like(dur=120)
    ref = _hp()(x)
    xg = x.copy()
    gaps = [(20_000, 20_001), (40_000, 40_010), (60_000, 60_100), (80_000, 82_000), (95_000, 105_000)]
    for a, b in gaps:
        xg[a:b] = np.nan
    y = _hp()(xg)
    errs = [np.abs(np.r_[y[a - 10 * FS:a], y[b:b + 10 * FS]] - np.r_[ref[a - 10 * FS:a], ref[b:b + 10 * FS]]).max()
            for a, b in gaps]
    # dropouts (bridged) are nearly invisible; long gaps behave like record edges (inherent error,
    # ~ the 1/f power near the cutoff: a few units for SD 20)
    assert errs[0] < 0.05 and errs[1] < 0.5 and errs[2] < 3
    assert max(errs[3:]) < 15
    # same signal without offset/drift gives the same output: offset and drift do not leak at gap edges
    y0 = _hp()(xg - 1500 - 3 * np.arange(xg.size) / FS)
    np.testing.assert_allclose(y, y0, atol=1e-6)


def test_lff_multichannel_shared_and_different_masks():
    X = np.vstack([_eeg_like(0, dur=30), _eeg_like(1, dur=30)])
    X[:, 5000:9000] = np.nan
    hp = _hp()
    Y = hp(X)
    for i in range(2):
        np.testing.assert_allclose(Y[i], hp(X[i]), atol=1e-9, equal_nan=True)
    X[1, 20000:21000] = np.nan  # different masks -> row by row
    Y = hp(X)
    np.testing.assert_array_equal(np.isnan(Y), np.isnan(X))
    np.testing.assert_allclose(Y[1], hp(X[1]), atol=1e-9, equal_nan=True)


def test_lff_max_gap_fill_zero_always_splits():
    x = _eeg_like(dur=30)
    x[10_000:10_002] = np.nan
    y = _hp(max_gap_fill=0)(x)
    np.testing.assert_allclose(y[:10_000], _hp()(x[:10_000]), atol=1e-9)


def test_decimate_then_lff_cascade_with_gaps():
    # the intended cascade: NaN-aware decimate -> 0.5 Hz HP. Used to raise (PR) / give all-NaN (v2.0.0).
    fs0, fs1 = 4000, 1000
    rng = np.random.default_rng(3)
    t0 = np.arange(120 * fs0) / fs0
    x = 800 + 2 * t0 + 10 * np.sin(2 * np.pi * 7 * t0) + rng.standard_normal(t0.size)
    xg = x.copy()
    xg[40 * fs0:40 * fs0 + 20] = np.nan        # 5 ms dropout
    xg[70 * fs0:75 * fs0] = np.nan             # 5 s gap
    d = decimate(xg, fs0, fs1)
    hp = _hp()
    y = hp(d)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(d))
    assert np.isfinite(y).sum() > 0.9 * y.size
    ref = hp(decimate(x, fs0, fs1))
    t1 = np.arange(y.size) / fs1
    far = np.isfinite(y) & (np.abs(t1 - 40) > 3) & ((t1 < 64) | (t1 > 81)) & (t1 > 6) & (t1 < 114)
    assert np.abs(y[far] - ref[far]).max() < 0.05
    ok = np.isfinite(y)
    assert np.abs(y[ok] - ref[ok]).max() < 2.0  # next to the 5 s gap: record-edge-like error only


# --------------------------------------------------------------------------- fft_filter
def test_fft_filter_nan_exact_mask_and_segments():
    x = _eeg_like(dur=60)
    xg = x.copy()
    xg[:100] = np.nan
    xg[10_000:10_002] = np.nan
    xg[30_000:35_000] = np.nan
    for kind in ('lp', 'hp'):
        y = fft_filter(xg, FS, 2.0, kind)
        np.testing.assert_array_equal(np.isnan(y), np.isnan(xg))
        # split segment == that segment filtered on its own with edges='extend'
        np.testing.assert_allclose(y[35_000:], fft_filter(x[35_000:], FS, 2.0, kind, edges='extend'), atol=1e-8)
    lp = fft_filter(xg, FS, 2.0, 'lp')
    hp = fft_filter(xg, FS, 2.0, 'hp')
    ok = np.isfinite(xg)
    np.testing.assert_allclose(lp[ok] + hp[ok], xg[ok], atol=1e-8)


def test_fft_filter_default_unchanged_without_nan():
    x = _eeg_like(dur=10)
    Xs = np.fft.fft(x)
    f = np.abs(np.fft.fftfreq(x.size, 1 / FS))
    ref = np.real(np.fft.ifft(np.where(f > 2.0, Xs, 0)))
    np.testing.assert_allclose(fft_filter(x, FS, 2.0, 'hp'), ref, atol=1e-8)
    np.testing.assert_allclose(fft_filter(x, FS, 2.0, 'hp', edges='periodic'), ref, atol=1e-8)


def test_fft_filter_extend_removes_edge_jump():
    x = _eeg_like(dur=60, offset=1500, drift=5.0)
    ref = fft_filter(_eeg_like(dur=60, offset=0, drift=0.0), FS, 1.0, 'hp', edges='extend')
    per = fft_filter(x, FS, 1.0, 'hp', edges='periodic')
    ext = fft_filter(x, FS, 1.0, 'hp', edges='extend')
    edge = slice(0, 2 * FS)
    assert np.abs(per[edge] - ref[edge]).max() > 50       # drift wraps into a big edge jump
    assert np.abs(ext[edge] - ref[edge]).max() < 1e-6     # offset and drift leave no trace


def test_fft_filter_all_nan():
    y = fft_filter(np.full(100, np.nan), FS, 2.0, 'lp')
    assert np.isnan(y).all()


# --------------------------------------------------------------------------- R6: FIR too short
def test_lff_warns_when_fir_cannot_resolve_cutoff():
    with pytest.warns(UserWarning, match='cannot resolve'):
        LowFrequencyFilter(fs=8000, cutoff=0.5, n_decimate=5)  # default 101 taps: gain 0.94 at 0.5 Hz
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        LowFrequencyFilter(fs=8000, cutoff=0.5, n_decimate=5, n_order=751)
        LowFrequencyFilter(fs=8000, cutoff=0.5, n_decimate=5, ftype='iir', n_order=3)
