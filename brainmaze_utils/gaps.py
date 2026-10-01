# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gap (NaN) handling for detectors: find gaps, fill them before detection, and remove
detections that fall in or near them afterwards.

Why fill at all
---------------
Most filters (IIR ``filtfilt``, FFT, Hilbert) spread a single NaN over the whole signal,
so a detector run on raw gappy data silently returns nothing. Background-modelling
detectors (e.g. Janca) additionally estimate a running background from the signal; a
gap filled with zeros or a straight line lowers that background for several seconds
around the gap and produces bursts of false detections there, and a hard step at a gap
edge rings through every filter. The fill therefore has to *look like background* and
must not create rough edges:

- **short gaps** (``<= max_interp_s``, default 0.1 s) are linearly interpolated between
  the edge values;
- **long gaps**, ``method='pink'`` (default): 1/f^beta noise scaled to the robust
  amplitude (MAD) of the neighbouring data, offset to the local level (bridged between
  the two sides), and **cross-faded** into a mirror image of the neighbouring signal at
  both edges with a raised-cosine taper, so the filled signal is continuous at the edges;
- **long gaps**, ``method='mirror'``: the neighbouring signal is mirrored into the gap
  from both sides and the two images are cross-faded over the whole gap (amplitude dip of
  the cross-fade compensated). Keeps the true local spectrum, but copies real events
  (e.g. a spike next to the gap) into the gap;
- ``method='linear'``: straight line for every gap. Not suitable for gaps longer than
  ~0.1 s before background-modelling detectors (see benchmark).

The fill is *not* data. Whatever the method, detections inside a gap are meaningless, so
**always post-process** with :func:`mask_in_gaps` / :func:`drop_in_gaps`, using the gaps
found on the **original** signal.

Typical use
-----------
>>> gaps = find_gaps(x)                         # (n_gaps, 2) sample indices, stop exclusive
>>> y = fill_gaps(x, fs)                        # finite signal, same shape
>>> times = my_detector(y, fs)                  # detection times in seconds
>>> times = drop_in_gaps(times, gaps, fs, margin_s=0.1)

Benchmark behind the defaults
-----------------------------
Janca spike detectors (eeg_forge reference and the v24 port), 1 h of scalp EEG at 500 Hz,
23 NaN gaps per gap length, IED-like transients injected 0.15-1.2 s from the gap edges;
false detections within 3 s of the gaps (after dropping detections inside gaps +0.1 s):

=============  =======  =======  =======  =======
gap length     0.5 s    2 s      10 s     60 s
=============  =======  =======  =======  =======
linear         6        113-303  206-312  223-298
pink           1        0-1      0-2      0-2
mirror         1        0        0        0-1
=============  =======  =======  =======  =======

Sensitivity to the injected transients 0.15 s from the gap edge was 100 % for pink and
mirror at every gap length (linear: 64-93 % with the v24 detector). Gaps <= 0.05 s made
no difference for any method.
"""

import warnings

import numpy as np

__all__ = ['find_gaps', 'gap_intervals', 'fill_gaps', 'pink_noise', 'mask_in_gaps',
           'drop_in_gaps']


def find_gaps(x):
    """
    Find runs of NaN samples in a 1-D signal.

    Parameters
    ----------
    x : array_like, shape (n_samples,)

    Returns
    -------
    np.ndarray, shape (n_gaps, 2), int
        ``[start, stop)`` sample indices of each NaN run (``stop`` exclusive), in order.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'find_gaps expects a 1-D signal, got shape {x.shape}; '
                         'call it per channel.')
    m = np.isnan(x).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


def gap_intervals(x, fs):
    """Gaps of a 1-D signal as ``[start, stop)`` times in seconds, shape (n_gaps, 2)."""
    return find_gaps(x) / float(fs)


def pink_noise(n, beta=1.0, fmin_bins=1, rng=None):
    """
    Zero-mean, unit-variance 1/f^beta noise of length ``n`` (spectral synthesis).

    Parameters
    ----------
    n : int
        Number of samples.
    beta : float
        Spectral exponent of the power spectrum, ``P(f) ~ 1/f**beta`` (1 = pink, 0 = white,
        2 = brown).
    fmin_bins : int
        Lowest non-zero frequency bin kept. Bins below are zeroed so that the realisation
        does not wander on time scales longer than the segment itself.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray, shape (n,)
    """
    rng = np.random.default_rng() if rng is None else rng
    if n <= 0:
        return np.zeros(0)
    if n == 1:
        return np.zeros(1)
    k = np.arange(n // 2 + 1, dtype=float)
    amp = np.zeros_like(k)
    keep = k >= max(fmin_bins, 1)
    amp[keep] = k[keep] ** (-beta / 2.0)
    phase = np.exp(2j * np.pi * rng.random(k.size))
    y = np.fft.irfft(amp * phase, n)
    sd = y.std()
    return (y - y.mean()) / sd if sd > 0 else np.zeros(n)


def _robust_sd(v):
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    mad = np.median(np.abs(v - np.median(v)))
    sd = 1.4826 * mad
    return float(sd if sd > 0 else v.std())


def _raised_cosine(n):
    """Weights going 1 -> 0 over n samples (exclusive of both end points)."""
    if n <= 0:
        return np.zeros(0)
    return 0.5 * (1 + np.cos(np.pi * (np.arange(1, n + 1) / (n + 1))))


def _reflect_index(k, length):
    """Triangle-wave index 0,1,..,L-1,L-1,..,0,0,1,.. so mirroring can exceed the context."""
    if length <= 1:
        return np.zeros_like(k)
    period = 2 * length
    k = k % period
    return np.where(k < length, k, period - 1 - k)


def _mirror_fill(y, s, e, nxt, ctx, lvl_a, lvl_b, base):
    """
    Mirror the neighbouring signal into the gap from both sides and cross-fade the two
    images over the whole gap. The cross-fade of two independent signals would dip in
    amplitude in the middle (by sqrt(w^2 + (1-w)^2)); that is compensated.
    """
    n = e - s
    k = np.arange(n)
    n_left = min(ctx, s)                 # left of s everything is finite (filled in order)
    n_right = min(ctx, nxt - e)          # contiguous valid samples up to the next gap
    if n_left > 0:
        left_img = y[s - 1 - _reflect_index(k, n_left)] - lvl_a
    if n_right > 0:
        right_img = y[e + _reflect_index(n - 1 - k, n_right)] - lvl_b
    if n_left > 0 and n_right > 0:
        w = _raised_cosine(n)
        dev = (w * left_img + (1 - w) * right_img) / np.sqrt(w ** 2 + (1 - w) ** 2)
    elif n_left > 0:
        dev = left_img
    else:
        dev = right_img
    return base + dev


def _fill_1d(x, fs, max_interp_s, context_s, taper_s, beta, method, rng):
    y = np.array(x, dtype=np.float64, copy=True)
    gaps = find_gaps(y)
    if gaps.size == 0:
        return y
    if gaps.shape[0] == 1 and gaps[0, 0] == 0 and gaps[0, 1] == y.size:
        warnings.warn('fill_gaps: signal is entirely NaN; returned unchanged.', RuntimeWarning)
        return y

    n_total = y.size
    max_interp = int(round(max_interp_s * fs))
    ctx = max(int(round(context_s * fs)), 2)
    taper = int(round(taper_s * fs))
    finite = np.isfinite(y)

    next_start = np.r_[gaps[1:, 0], n_total]
    for (s, e), nxt in zip(gaps, next_start):
        n = e - s
        has_left, has_right = s > 0, e < n_total
        # context = finite samples of the ORIGINAL signal next to the gap
        left = y[max(s - ctx, 0):s][finite[max(s - ctx, 0):s]] if has_left else np.zeros(0)
        right = y[e:min(e + ctx, n_total)][finite[e:min(e + ctx, n_total)]] if has_right else np.zeros(0)

        if n <= max_interp or method == 'linear':
            a = y[s - 1] if has_left else y[e]
            b = y[e] if has_right else y[s - 1]
            r = np.arange(1, n + 1) / (n + 1)
            y[s:e] = a + (b - a) * r
            continue

        context = np.concatenate([left, right])
        sd = _robust_sd(context)
        # local level: median of the near-edge data on each side, linearly bridged
        near = max(min(int(round(0.5 * fs)), ctx), 1)
        lvl_a = np.median(left[-near:]) if left.size else np.median(right[:near])
        lvl_b = np.median(right[:near]) if right.size else np.median(left[-near:])
        r = np.arange(1, n + 1) / (n + 1)
        base = lvl_a + (lvl_b - lvl_a) * r

        if method == 'mirror':
            y[s:e] = _mirror_fill(y, s, e, nxt, ctx, lvl_a, lvl_b, base)
            continue
        if method != 'pink':
            raise ValueError(f"unknown method {method!r}; use 'pink', 'mirror' or 'linear'")
        fill = base + sd * pink_noise(n, beta=beta, rng=rng)

        # cross-fade into the mirror image of the neighbouring signal at each edge:
        # mirror_left[k] = y[s-1-k], so the fill starts exactly at the last valid value.
        t = min(taper, n // 2) if (has_left and has_right) else min(taper, n)
        if has_left and t > 0:
            tl = min(t, left.size)
            if tl > 0:
                w = _raised_cosine(tl)
                mirror = y[s - 1 - np.arange(tl)]
                fill[:tl] = w * mirror + (1 - w) * fill[:tl]
        if has_right and t > 0:
            tr = min(t, nxt - e)          # contiguous valid samples up to the next gap
            if tr > 0:
                w = _raised_cosine(tr)[::-1]
                mirror = y[e + np.arange(tr)][::-1]
                fill[n - tr:] = w * mirror + (1 - w) * fill[n - tr:]
        y[s:e] = fill
    return y


def fill_gaps(x, fs, max_interp_s=0.1, method='pink', context_s=10.0, taper_s=0.5,
              beta=1.0, seed=0, axis=-1):
    """
    Fill NaN gaps so a detector can run on the signal (see module docstring).

    Parameters
    ----------
    x : array_like
        Signal, 1-D ``(n_samples,)`` or N-D with time along ``axis``. Not modified.
    fs : float
        Sampling frequency (Hz).
    max_interp_s : float
        Gaps up to this length (seconds) are linearly interpolated; longer gaps use
        ``method``. Default 0.1 s.
    method : {'pink', 'mirror', 'linear'}
        Fill for long gaps. ``'pink'`` (default): 1/f^beta noise at the robust amplitude of
        the neighbouring data, cross-faded into the mirrored neighbouring signal at both
        edges. ``'mirror'``: the neighbouring signal mirrored in from both sides and
        cross-faded over the whole gap (copies real neighbouring events into the gap).
        ``'linear'``: straight line (not recommended for gaps longer than ~0.1 s before
        background-modelling detectors).
    context_s : float
        Seconds of valid data on each side used to estimate amplitude and level.
    taper_s : float
        ``'pink'`` only: length (seconds) of the raised-cosine cross-fade into the mirrored
        signal at each edge (capped at half the gap).
    beta : float
        Spectral exponent of the pink noise (1 = pink).
    seed : int or None
        Seed for the noise generator; the default makes fills reproducible. ``None`` = random.
    axis : int
        Time axis for N-D input; every other index is filled independently.

    Returns
    -------
    np.ndarray (float64)
        Filled copy of ``x``. An all-NaN channel is returned unchanged (still NaN) with a
        ``RuntimeWarning``.

    Notes
    -----
    Always remove detections in gaps afterwards with :func:`drop_in_gaps` /
    :func:`mask_in_gaps`, using gaps found on the original signal.
    """
    if fs <= 0:
        raise ValueError('fs must be positive')
    if max_interp_s < 0 or context_s <= 0 or taper_s < 0:
        raise ValueError('max_interp_s, taper_s must be >= 0 and context_s > 0')
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if x.ndim == 1:
        return _fill_1d(x, fs, max_interp_s, context_s, taper_s, beta, method, rng)
    xm = np.moveaxis(x, axis, -1)
    out = np.empty_like(xm)
    for idx in np.ndindex(xm.shape[:-1]):
        out[idx] = _fill_1d(xm[idx], fs, max_interp_s, context_s, taper_s, beta, method, rng)
    return np.moveaxis(out, -1, axis)


def mask_in_gaps(start_s, gaps, fs=None, margin_s=0.1, end_s=None):
    """
    Boolean mask of detections that overlap a gap (extended by ``margin_s`` on each side).

    Parameters
    ----------
    start_s : array_like
        Detection times (or interval starts) in seconds.
    gaps : array_like, shape (n_gaps, 2)
        Gaps from :func:`find_gaps` (sample indices; pass ``fs``) or from
        :func:`gap_intervals` (seconds; leave ``fs=None``). Stop is exclusive.
    fs : float, optional
        Sampling frequency, required when ``gaps`` are sample indices.
    margin_s : float
        Exclusion margin around every gap (seconds). Default 0.1 s.
    end_s : array_like, optional
        Interval ends for interval detections; a detection is masked if any part of
        ``[start_s, end_s]`` overlaps a widened gap.

    Returns
    -------
    np.ndarray of bool, True where the detection is in/near a gap.
    """
    start = np.atleast_1d(np.asarray(start_s, dtype=float))
    end = start if end_s is None else np.atleast_1d(np.asarray(end_s, dtype=float))
    g = np.asarray(gaps, dtype=float).reshape(-1, 2)
    if fs is not None:
        g = g / float(fs)
    out = np.zeros(start.shape, dtype=bool)
    if g.size == 0 or start.size == 0:
        return out
    lo = g[:, 0] - margin_s
    hi = g[:, 1] + margin_s
    # interval [start, end] overlaps [lo, hi) iff start < hi and end >= lo
    order = np.argsort(lo)
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    k = np.searchsorted(lo, end, side='right') - 1     # last widened gap starting <= end
    valid = k >= 0
    out[valid] = start[valid] < hi[k[valid]]
    return out


def drop_in_gaps(start_s, gaps, fs=None, margin_s=0.1, end_s=None):
    """Return ``start_s`` without the detections flagged by :func:`mask_in_gaps`."""
    start = np.atleast_1d(np.asarray(start_s, dtype=float))
    return start[~mask_in_gaps(start, gaps, fs=fs, margin_s=margin_s, end_s=end_s)]
