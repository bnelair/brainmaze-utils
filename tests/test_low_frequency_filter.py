# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LowFrequencyFilter: edge transients (DC offset, drift) and unchanged interior response."""

import numpy as np
import pytest
import scipy.signal as ss

from brainmaze_utils.signal import LowFrequencyFilter

FS = 8000
CUT = 0.5


def _signal(seed=0, dur=20):
    """Tones well above the cutoff (A=10 each) + white noise; the 'ideal' 0.5 Hz HP output."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * FS)) / FS
    tones = sum(10 * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in (3, 10, 40))
    noise = 2 * rng.standard_normal(t.size)
    return t, tones + noise


def _edge_mid(err, fs=FS):
    edge = max(np.abs(err[:fs]).max(), np.abs(err[-fs:]).max())  # first/last 1 s
    mid = np.abs(err[2 * fs:-2 * fs]).max()
    return edge, mid


@pytest.mark.parametrize('cfg', [
    dict(n_decimate=5, ftype='iir', n_order=3),
    dict(n_decimate=6, ftype='fir', n_order=301),
])
def test_hp_large_dc_offset_and_drift_no_edge_jump(cfg):
    """0.5 Hz HP at 8 kHz on DC 2000 + drift 40/s: edges as good as for a zero-mean signal.

    Before the fix the first/last second deviated by ~1000-1900 (half the offset + drift).
    """
    t, x0 = _signal()
    x = x0 + 2000 + 40 * t
    hp = LowFrequencyFilter(fs=FS, cutoff=CUT, filter_type='hp', **cfg)
    y = hp(x)
    y0 = hp(x0)
    # offset + linear drift have no effect at all, edges included
    np.testing.assert_allclose(y, y0, atol=1e-6)
    edge, mid = _edge_mid(y - x0)
    # edge error is set only by the irreducible unknown-continuation effect of the tones
    # (~A * fc / (pi * f) = 0.53 for the 3 Hz, A=10 tone), not by offset/drift (2000 / 40 per s)
    assert edge < 1.5, edge
    assert edge < 1e-3 * 2000
    assert mid < 0.05, mid


def test_hp_edge_error_vs_stationary_on_offset_drift_noise():
    """Offset + drift + broadband noise only (no low-frequency tones): the edge error must be
    of the same order as the stationary error (the noise power below the cutoff)."""
    rng = np.random.default_rng(3)
    t = np.arange(20 * FS) / FS
    noise = 5 * rng.standard_normal(t.size)
    x = noise + 1500 - 25 * t
    y = LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir', n_order=3, filter_type='hp')(x)
    edge, mid = _edge_mid(y - noise)
    assert edge < 3 * mid + 0.05, (edge, mid)
    assert edge < 0.5


def test_lp_returns_offset_and_drift_exactly():
    t = np.arange(10 * FS) / FS
    x = 300 + 12.5 * t
    y = LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir', filter_type='lp')(x)
    np.testing.assert_allclose(y, x, atol=1e-8)


def _legacy_lowpass_interior(lff, x):
    """Reference cascade (pre-fix algorithm, b/a filtfilt) with very long zero padding,
    applied to a zero-mean signal: its interior is the cascade's true response."""
    pad = 2 ** int(np.ceil(np.log2(x.size))) * 2
    z = np.concatenate((np.zeros(pad), x, np.zeros(pad)))
    C = (-z.size) % (2 ** lff.n_decimate)
    z = np.concatenate((np.zeros(C), z))
    for _ in range(lff.n_decimate):
        z = ss.filtfilt(lff.b_dec, lff.a_dec, z)[::2]
    z = ss.filtfilt(lff.b_filt, lff.a_filt, z)
    for _ in range(lff.n_decimate):
        u = np.zeros(2 * z.size)
        u[::2] = z
        z = ss.filtfilt(lff.b_dec, lff.a_dec, u) * 2
    return z[pad + C: pad + C + x.size]


@pytest.mark.parametrize('cfg', [
    dict(n_decimate=5, ftype='iir', n_order=3),
    dict(n_decimate=4, ftype='iir', n_order=2),
    dict(n_decimate=6, ftype='fir', n_order=301),
])
def test_interior_response_unchanged(cfg):
    """Far from the edges, the new filter equals the original cascade (same frequency response)."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(40 * FS)
    x -= x.mean()
    lff = LowFrequencyFilter(fs=FS, cutoff=CUT, filter_type='lp', **cfg)
    new = lff(x)
    ref = _legacy_lowpass_interior(lff, x)
    core = slice(12 * FS, -12 * FS)
    scale = np.abs(ref[core]).max()
    assert np.abs(new[core] - ref[core]).max() < 1e-4 * scale


def test_frequency_response_iir():
    """Zero-phase Butterworth: LP gain ~1 well below, ~0.5 at, ~0 well above the cutoff."""
    lff = LowFrequencyFilter(fs=1000, cutoff=CUT, n_decimate=3, ftype='iir', n_order=3, filter_type='lp')
    t = np.arange(200 * 1000) / 1000
    core = slice(60 * 1000, -60 * 1000)
    gains = {}
    for f in (0.05, 0.5, 5.0):
        x = np.sin(2 * np.pi * f * t)
        y = lff(x)
        gains[f] = np.sqrt(np.mean(y[core] ** 2) / np.mean(x[core] ** 2))
    assert gains[0.05] == pytest.approx(1.0, abs=0.01)
    assert gains[0.5] == pytest.approx(0.5, abs=0.03)
    assert gains[5.0] < 1e-3


def test_multichannel_matches_single_channel_and_no_mutation():
    t, x0 = _signal(dur=8)
    X = np.vstack([x0 + 100, -x0 + 5 * t])
    X_copy = X.copy()
    hp = LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir', filter_type='hp')
    Y = hp(X)
    np.testing.assert_array_equal(X, X_copy)
    assert Y.shape == X.shape
    for i in range(2):
        np.testing.assert_allclose(Y[i], hp(X[i]), atol=1e-9)


def test_short_signal_works():
    x = np.random.default_rng(0).standard_normal(500) + 50
    y = LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir', filter_type='hp')(x)
    assert y.shape == x.shape and np.all(np.isfinite(y))


def test_validation():
    with pytest.raises(ValueError):
        LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, filter_type='bp')
    with pytest.raises(ValueError):
        LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='cheby')
    with pytest.raises(ValueError):
        LowFrequencyFilter(fs=100, cutoff=10, n_decimate=3)  # cutoff above decimated Nyquist
    with pytest.raises(ValueError):
        LowFrequencyFilter(fs=FS, cutoff=None)
    with pytest.raises(ValueError):
        LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir', max_gap_fill=-1)
    lff = LowFrequencyFilter(fs=FS, cutoff=CUT, n_decimate=5, ftype='iir')
    x = np.ones(1000)
    x[5] = np.inf  # NaN is a gap; inf is not
    with pytest.raises(ValueError):
        lff(x)
