# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Gap (NaN / inf) handling for detectors: find gaps, fill them before detection, and remove
detections that fall in or near them afterwards.

Recommended pipeline
--------------------
.. code-block:: python

    from brainmaze_utils.gaps import find_gaps, fill_gaps, drop_in_gaps

    gaps = find_gaps(x)                       # 1. on the ORIGINAL signal: (n_gaps, 2) samples
    y = fill_gaps(x, fs)                      # 2. finite copy, same shape and float dtype
    det = my_detector(y, fs)                  # 3. any detector (here: times in seconds)
    det = drop_in_gaps(det, gaps / fs, fs,    # 4. ALWAYS drop detections in / near gaps
                       units='seconds',       #    units of BOTH det and gaps (see below)
                       margin_s=0.1)

Step 4 is not optional: whatever fills a gap is not data, and an event detected there (or
straddling a gap edge) is meaningless. Use the gaps found on the original signal; the
filled signal has none.

For N-D input, ``fill_gaps`` fills every channel along ``axis`` in one call; ``find_gaps``
and the post-filters work per channel (1-D), e.g. ``gaps = [find_gaps(x[ch]) for ch ...]``.

Why fill at all
---------------
Most filters (IIR ``filtfilt``, FFT, Hilbert) spread a single NaN or inf over the whole
signal, so a detector run on raw gappy data silently returns nothing. Background-modelling
detectors (Janca, RMS/HFO detectors, ...) additionally estimate a running background from
the signal: a gap filled with zeros or a straight line lowers that background for several
seconds around the gap and produces bursts of false detections there; a fill with too
much power in the detector's band raises it and hides real events next to the gap; and a
hard step or kink at a gap edge rings through every filter. The fill therefore has to
*look like the neighbouring background* (same level, same power in every band) and must
join the data without steps or kinks.

Fill methods
------------
- **Short gaps** (``<= max_interp_s``, default 0.1 s), and every gap with
  ``method='linear'``: linear interpolation between the two edge samples (constant
  extension for a gap at the start or end). All short gaps are filled in one vectorised
  step, so packet-loss-like recordings with thousands of tiny gaps are cheap.
- **Long gaps**, ``method='spectral'`` (**default**): Gaussian noise with the *power
  spectrum of the neighbouring data*. The PSD is estimated from up to ``context_s`` of
  valid data on each side of the gap (Welch, Hann window, half-overlapping segments of
  the gap length rounded up to a power of two, at most ~1 s). It is robust to spikes and
  artifacts in the context: segments whose log power in any of four log-spaced bands
  exceeds the median over segments by more than 3 robust SDs (MAD; at least 2x) are
  dropped before averaging. (A plain median Welch was biased low on real, non-stationary
  EEG and still let a few large spikes in the context inflate the fill by 20-30 %.) Each FFT bin of the fill gets the power that the context
  has in that frequency band (random phase, Rayleigh amplitude), so the fill matches the
  neighbours band by band
  (the 1/f slope, alpha or other peaks, line noise, the white noise floor) rather than
  only in total RMS. It rides on the local level (median of the 0.5 s next to each edge,
  linearly bridged across the gap).
- **Long gaps**, ``method='pink'``: unit-variance 1/f^``beta`` noise scaled to the robust
  (MAD) amplitude of the context, on the same level bridge. The spectral *shape* is fixed,
  so its band powers generally differ from the data (see below); kept for comparison and
  for data that really is 1/f^beta.
- **Long gaps**, ``method='mirror'``: the neighbouring signal mirrored into the gap from
  both sides, the two images cross-faded over the whole gap with an amplitude correction
  that accounts for the measured correlation of the two images. Keeps the local spectrum,
  but copies real neighbouring events (e.g. a spike next to the gap) into the gap.

Smooth edges (``'spectral'`` and ``'pink'``): over ``taper_s`` (default 0.5 s, at most
half the gap) at each edge the fill starts as the **point (odd) reflection** of the
neighbouring data, ``2*x[s-1] - x[s-2-k]``, which continues both the value *and the slope*
of the signal at the edge, and is cross-faded into the noise with **power-complementary**
raised-cosine weights (``cos``/``sin``), so the band power does not dip in the taper
either. ``'mirror'`` gets the same odd reflection over the first 10 ms. The filled signal
is continuous in value and slope at both edges.

Units (post-filters)
--------------------
``mask_in_gaps`` / ``drop_in_gaps`` take a **required** ``units='seconds'|'samples'``
that applies to the detections *and* to the gaps; ``margin_s`` is always in seconds and
``fs`` is always required. Gaps from :func:`find_gaps` are sample indices, from
:func:`gap_intervals` seconds; detections are whatever your detector returns (e.g.
eeg_forge's Janca returns sample indices). Mix-ups raise instead of silently masking
nothing: integer-dtype gaps with ``units='seconds'`` and non-integer values with
``units='samples'`` are rejected. Convert one side first when they differ
(``gaps / fs`` or ``det / fs``).

Randomness
----------
``seed`` (default 0, reproducible) may be an int, a sequence of ints, ``None`` (fresh
entropy), a :class:`numpy.random.SeedSequence` or a :class:`numpy.random.Generator`. The
noise of every long gap is drawn from its own stream derived from the seed, the gap's
position *and a hash of the context data next to it*. Consequently:

- a channel's fill does not depend on other channels, their gaps or their order, nor on
  the other gaps of the same channel;
- the same channel gives the same fill in a 1-D call and inside an N-D call;
- different channels sharing a gap (recording-wide dropouts) get **independent** noise,
  also when they are filled one at a time in separate calls with the same seed, so
  bipolar/CAR montages, coherence and connectivity do not see spuriously identical
  segments. (Two channels with bit-identical context data get identical fills.)

Defaults and the evidence behind them
-------------------------------------
``max_interp_s=0.1``: in the Janca detection benchmark (1 h scalp EEG, 500 Hz) gaps
``<= 0.05 s`` made no difference for any method, while straight lines over 0.5 s gaps
already produced false detections; 0.1 s also matches the default post-filter margin.

``method='spectral'``: measured on real Fz-Cz EEG (500 Hz, 30-40 gaps per length) and on
a synthetic 1/f^2 + 10 Hz + 60 Hz + white background at 5 kHz (probes and full tables in
brainmaze-utils PR #26):

- band RMS of the fill / band RMS of the neighbouring data, median over gaps, bands from
  0.5 Hz to 1 kHz: ``'spectral'`` 0.92-1.05 for 10-60 s gaps and 0.74-1.25 for 1-2 s
  gaps (the scatter of a short realisation in narrow low bands; 10-1000 Hz bands stay
  within 0.89-1.03 at every length). ``'pink'``: 1.3-2.1 in 4-200 Hz on real data and
  about 5 in 80-1000 Hz at 5 kHz, because its 1/f shape is fixed;
- an RMS (80-500 Hz, 10 s window) background threshold in the *valid* data 0.1-4 s
  outside a gap, filled / gap-free: ``'spectral'`` 1.00 and ``'mirror'`` 0.99-1.00 for
  0.5, 2 and 10 s gaps; ``'pink'`` 1.34 / 2.28 / 2.77, i.e. an RMS-threshold detector
  misses real events next to a pink-filled gap;
- eeg_forge's Janca threshold 0.1-2.5 s outside a gap, filled / gap-free (0.2-60 s gaps):
  ``'spectral'`` median 0.98-1.00, p90 <= 1.07; ``'mirror'`` p90 <= 1.04; ``'pink'``
  p90 up to 1.19;
- Janca detections (1 h, 23 gaps per length, 0.5-60 s, transients injected 0.15-1.2 s
  outside both edges, ``drop_in_gaps`` margin 0.1 s): no false detections near the gaps
  for ``'spectral'``, ``'pink'`` or ``'mirror'`` (``'linear'``: 9-415). With strong
  transients all three find 100 %. With weak ones (near the threshold) the sensitivity at
  0.15 s from the edge is 0.74-0.91 gap-free, 0.65-0.76 with ``'spectral'``, 0.41-0.78
  with ``'mirror'`` (it copies the transients into the gap, raising the background) and
  0.26-0.67 with ``'pink'``.

``'spectral'`` is the default because it is the only fill that keeps the background of
the neighbouring data in every band without copying neighbouring events (spikes,
artifacts) into the gap; its Welch estimate also rejects such events in the
context.
``context_s=10``: enough Welch segments for a stable PSD of a 60 s gap; shorter
gaps use proportionally less (about 16 half-overlapping Welch segments, segment = gap length rounded up to
a power of two, at most ~1 s), so the cost scales with the gap, not with
``fs * context_s``. ``taper_s=0.5``: removed the false detections that a hard junction
produced in the benchmark.

Limitations
-----------
- The fill is stationary noise: it does not reproduce oscillatory bursts, spikes, sleep
  spindles or other non-stationary structure, nor the phase relations between channels
  (each channel is filled independently). Do not compute features that depend on the
  content of the gap (event rates, coherence, phase) without excluding the gaps.
- The spectrum is estimated from at most ``context_s`` on each side; if the state changes
  across the gap the fill uses the average of both sides. With fewer than 16 valid
  context samples the fill falls back to ``'pink'`` scaled by the MAD of what is there.
- Frequencies below about 0.5 Hz (or ``1 / gap`` for short gaps) are represented only by
  the level bridge; the fill has no infra-slow drift.
- A detector's background estimate *straddling* a gap is still partly made of fill, so the
  post-filter margin should cover the detector's own edge sensitivity (default 0.1 s;
  increase it for detectors with long filters or windows).
- Short gaps are linear: continuous in value but not in slope. A slope-matching (cubic)
  interpolation would extrapolate sample-to-sample noise over the gap; the kink is
  covered by the post-filter margin.
- ``'mirror'`` corrects its cross-fade amplitude with the *broadband* correlation of the
  two mirror images; a narrow-band coherent component such as line noise can still come
  out ~1.1-1.3x too strong in short gaps (it was up to 1.5x without the correction).
- A channel that is entirely NaN/inf cannot be filled; see ``all_nan``.
"""

import functools
import hashlib
import numbers
import warnings

import numpy as np
from scipy import fft as _sfft
from scipy import signal as _ss

__all__ = ['find_gaps', 'gap_intervals', 'fill_gaps', 'pink_noise', 'mask_in_gaps',
           'drop_in_gaps']

_METHODS = ('spectral', 'pink', 'mirror', 'linear')
_UNITS = ('seconds', 'samples')
_ALL_NAN = ('keep', 'zero', 'raise')
_MIN_SPECTRAL_CONTEXT = 16      # valid samples needed to estimate a PSD
_MIRROR_KINK_S = 0.01           # odd-reflection blend at the edges of the mirror fill
_REJECT_MADS = 3.0              # Welch segment > median + 3 MAD in >= 2 log bands = artifact
_REJECT_BIG_MADS = 5.0          # ... or > median + max(5 MAD, log 16) in any band
_NPERSEG_MIN_S = 1.0            # shortest Welch segment (s): delta band in the spectrum model
_RAYLEIGH = False               # (experiments) complex-Gaussian bins instead of random phase
_NPERSEG_CAP_S = 8.0            # longest Welch segment (s) for long gaps (frequency resolution)
_COND_S = 0.02                  # observed samples per edge conditioning the fill (s; 4..64)
_CLIP_SDS = 4.0                 # edge residuals clipped to this many robust SDs (artifacts)
_NUGGET = 1e-9                  # relative diagonal loading of the conditioning covariance


# ------------------------------------------------------------------ validation helpers
def _check_fs(fs):
    try:
        fs = float(fs)
    except (TypeError, ValueError):
        raise ValueError(f'fs must be a positive finite number, got {fs!r}') from None
    if not (np.isfinite(fs) and fs > 0):
        raise ValueError(f'fs must be a positive finite number, got {fs!r}')
    return fs


def _check_nonneg(name, v, strict=False):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a finite number, got {v!r}') from None
    if not np.isfinite(v) or v < 0 or (strict and v == 0):
        raise ValueError(f'{name} must be finite and {"> 0" if strict else ">= 0"}, got {v!r}')
    return v


def _runs(bad):
    """[start, stop) of the True runs of a 1-D boolean array."""
    d = np.diff(bad.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], axis=1).astype(np.int64)


# ------------------------------------------------------------------ finding gaps
def find_gaps(x):
    """
    Find runs of non-finite samples (NaN, +inf, -inf) in a 1-D signal.

    Parameters
    ----------
    x : array_like, shape (n_samples,)

    Returns
    -------
    np.ndarray, shape (n_gaps, 2), int64
        ``[start, stop)`` **sample indices** of each run (``stop`` exclusive), in order.
        Use them in :func:`mask_in_gaps` / :func:`drop_in_gaps` with
        ``gap_units='samples'``.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f'find_gaps expects a 1-D signal, got shape {x.shape}; '
                         'call it per channel.')
    return _runs(~np.isfinite(x))


def gap_intervals(x, fs):
    """
    Gaps of a 1-D signal as ``[start, stop)`` **times in seconds**, shape (n_gaps, 2),
    float64. Use them with ``gap_units='seconds'`` in :func:`mask_in_gaps` /
    :func:`drop_in_gaps`.
    """
    return find_gaps(x) / _check_fs(fs)


# ------------------------------------------------------------------ noise generators
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
    rng : np.random.Generator, int or None
        Generator or seed (passed to :func:`numpy.random.default_rng`).

    Returns
    -------
    np.ndarray, shape (n,), float64
    """
    rng = np.random.default_rng(rng)
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


def _nperseg(n, fs, ctx):
    """
    Welch segment for a gap of n samples: the next power of two >= n, but at least ~1 s
    (so that the spectrum, and with it the edge conditioning, includes the delta band
    even for short gaps), at most ~8 s (long gaps: 0.12 Hz resolution, slow rhythms are
    not smeared into higher bands) and at most half the context (>= 3 segments per side).
    """
    def p2(v, rnd):
        return 1 << int(rnd(np.log2(max(v, 16.0))))
    cap = min(p2(_NPERSEG_CAP_S * fs, np.round), p2(ctx / 2.0, np.floor))
    lo = p2(_NPERSEG_MIN_S * fs, np.round)
    return int(max(16, min(max(p2(n, np.ceil), lo), cap)))


def _prep_context(ctxs):
    """[(values, ok)] -> [(values, cumsum of non-finite, cumsum of not-ok)] (computed once)."""
    out = []
    for c, ok in ctxs:
        out.append((c, np.r_[0, np.cumsum(~np.isfinite(c))], np.r_[0, np.cumsum(~ok)]))
    return out


def _segments(prep, nps):
    """
    Half-overlapping Welch segments of the prepared context pieces (``_prep_context``).
    Returns (segments, pure): only segments without non-finite samples; ``pure`` marks
    those made only of original (never filled or interpolated) samples.
    """
    segs, pure = [], []
    step = max(nps // 2, 1)
    for c, nf, nb in prep:
        if c.size < nps:
            continue
        st = np.arange(0, c.size - nps + 1, step)
        st = st[(nf[st + nps] - nf[st]) == 0]
        if st.size:
            segs.append(c[st[:, None] + np.arange(nps)])
            pure.append((nb[st + nps] - nb[st]) == 0)
    if not segs:
        return np.zeros((0, nps)), np.zeros(0, dtype=bool)
    return np.concatenate(segs), np.concatenate(pure)


@functools.lru_cache(maxsize=32)
def _hann(nps):
    w = _ss.get_window('hann', nps)
    w.setflags(write=False)
    return w


def _keep_segments(lb):
    """
    Segments (rows of log band powers ``lb``, 4 log-spaced bands) that are not artifacts.

    A segment is an artifact if its log power exceeds the median over segments by more
    than max(3 MAD, log 2) in at least TWO bands (spikes, pops and movement are broadband),
    or by more than max(5 MAD, log 16) in any band (a single huge narrow-band event).
    Physiological bursts (spindles, alpha/beta bursts) raise one band by less than 16x and
    are kept (scratch/utils-gaps/r3/v3_rules.py: burst band 0.93-0.96 of the context for
    1 s gaps, vs 0.64-0.85 with the former "any band > 3 MAD" rule).
    """
    med = np.median(lb, axis=0)
    mad = 1.4826 * np.median(np.abs(lb - med), axis=0)
    some = lb > med + np.maximum(_REJECT_MADS * mad, np.log(2.0))
    huge = lb > med + np.maximum(_REJECT_BIG_MADS * mad, np.log(16.0))
    return (some.sum(axis=1) < 2) & ~huge.any(axis=1)


def _cum_power(seg, fs):
    """
    Artifact-robust Welch PSD of the segments ``seg`` (n_seg, nps) as a cumulative power
    function: returns (edges, cum) such that ``np.interp(f, edges, cum)`` is the variance in
    (0, f] (DC excluded).
    """
    nps = seg.shape[1]
    win = _hann(nps)
    spec = np.fft.rfft((seg - seg.mean(axis=1, keepdims=True)) * win, axis=1)
    pw = spec.real ** 2 + spec.imag ** 2
    if seg.shape[0] > 2:
        # drop segments with an artifact/spike: log power in any of 4 log-spaced bands above
        # median + max(3 MAD, log 2) over segments; then a plain mean. (A median Welch was
        # biased low on non-stationary EEG, a fixed 4-8x threshold either biased or let
        # clusters of moderate spikes through; scratch/utils-gaps/exp_welch_variants*.)
        nb = pw.shape[1]
        edges = np.unique(np.clip([1, nb // 64, nb // 16, nb // 4], 1, nb - 1))
        lb = np.log(np.add.reduceat(pw, edges, axis=1) + np.finfo(float).tiny)
        keep = _keep_segments(lb)
        if keep.sum() >= 3:
            pw = pw[keep]
    p = pw.mean(axis=0)
    p = p / (fs * np.sum(win ** 2))
    p[1:] *= 2.0                                        # one-sided
    if nps % 2 == 0:
        p[-1] /= 2.0
    f = np.fft.rfftfreq(nps, 1.0 / fs)
    p[0] = 0.0                                          # the level is the bridge's job
    df = fs / nps
    upper = np.minimum(f + df / 2.0, fs / 2.0)
    width = np.diff(np.r_[0.0, upper])
    return np.r_[0.0, upper], np.r_[0.0, np.cumsum(p * width)]


def _interp_fn(ec):
    edges, cum = ec
    return lambda f: np.interp(f, edges, cum)


def _context_psd(ctxs, nps, fs):
    """
    Cumulative power function ``F(f)`` (variance in (0, f]) of the context, or None if
    there are fewer than ``_MIN_SPECTRAL_CONTEXT`` valid samples.

    Uses Welch segments made only of ORIGINAL samples when there are >= 3 of them.
    Interpolated short gaps and earlier fills have (almost) no high-frequency power, so
    when the context is riddled with them (packet loss) the spectrum is pieced together:
    below ``2 fs / k`` from the segments that include filled samples, above it from the
    longest segments (length k) that are made of original samples only.
    """
    prep = _prep_context(ctxs)
    seg, pure = _segments(prep, nps)
    while pure.sum() < 3 and seg.shape[0] < 3 and nps > 16:
        nps //= 2
        seg, pure = _segments(prep, nps)
    if pure.sum() >= 3:
        return _interp_fn(_cum_power(seg[pure], fs))
    if seg.shape[0]:
        lo_fn = _interp_fn(_cum_power(seg, fs))
        k = nps // 2
        while k >= 16:
            s2, p2 = _segments(prep, k)
            if p2.sum() >= 3:
                hi_fn = _interp_fn(_cum_power(s2[p2], fs))
                f0 = 2.0 * fs / k
                c0, h0 = float(lo_fn(f0)), float(hi_fn(f0))
                return lambda f: np.where(f <= f0, lo_fn(np.minimum(f, f0)),
                                          c0 + hi_fn(f) - h0)
            k //= 2
        return lo_fn
    c = np.concatenate([v[ok] for v, ok in ctxs])
    if c.size < _MIN_SPECTRAL_CONTEXT:
        return None
    nps = int(min(nps, c.size))
    step = max(nps // 2, 1)
    st = np.arange(0, c.size - nps + 1, step)
    return _interp_fn(_cum_power(c[st[:, None] + np.arange(nps)], fs))


def _bin_power(m, fs, cum_fn):
    """Variance per rfft bin of a length-m synthesis, from a cumulative power function."""
    df = fs / m
    edges = np.clip((np.arange(m // 2 + 2) - 0.5) * df, 0.0, fs / 2.0)
    power = np.diff(cum_fn(edges))
    power[0] = 0.0
    return power


def _synthesize(power, m, rng):
    """
    Noise of length m with variance ``power[k]`` from rfft bin k: random phase, amplitude
    ``sqrt(power)`` (fixed, so the band powers of the fill match the estimate exactly
    instead of scattering with a Rayleigh amplitude per bin; a narrow peak such as a
    line or an alpha rhythm keeps a constant envelope).
    """
    # a length-m irfft gets variance 2|X_k|^2/m^2 from bin k (|X_k|^2/m^2 at Nyquist)
    nb = power.size
    if _RAYLEIGH:
        X = (rng.standard_normal(nb) + 1j * rng.standard_normal(nb)) * (np.sqrt(power) * (m / 2.0))
    else:
        X = np.exp(2j * np.pi * rng.random(nb)) * (np.sqrt(power) * (m / np.sqrt(2.0)))
    if m % 2 == 0:
        X[-1] = (1.0 if rng.random() < 0.5 else -1.0) * np.sqrt(power[-1]) * m
    X[0] = 0.0
    return np.fft.irfft(X, m)


def _robust_sd(v):
    v = v[np.isfinite(v)]
    if v.size < 2:
        return 0.0
    mad = np.median(np.abs(v - np.median(v)))
    sd = 1.4826 * mad
    return float(sd if sd > 0 else v.std())


def _root_seed(seed):
    if isinstance(seed, np.random.SeedSequence):
        return seed
    if isinstance(seed, np.random.Generator):
        return np.random.SeedSequence(int(seed.integers(2 ** 63)))
    if isinstance(seed, (np.random.RandomState, np.random.BitGenerator)):
        raise TypeError('seed must be None, an int, a sequence of ints, a SeedSequence or '
                        f'a numpy Generator, got {type(seed).__name__}')
    return np.random.SeedSequence(seed)


def _gap_rng(root, n, *context):
    """
    Independent stream per gap from (seed, gap length, hash of the QUANTISED context data).
    The context is quantised to 1/1000 of its robust SD around its median, so ulp-level
    differences (numpy/scipy versions, platforms) and the units (uV vs V) do not change the
    fill; the gap position is not used, so the same data at another array offset (e.g.
    segment-wise processing) gives the same fill.
    """
    h = hashlib.blake2b(digest_size=8)
    for c in context:               # every k-th sample (<= ~4096 per side) keeps it cheap
        v = np.asarray(c[::max(1, c.size // 4096)], dtype=np.float64)
        h.update(np.int64(v.size).tobytes())
        if v.size:
            sd = _robust_sd(v)
            q = (v - np.median(v)) / (sd if sd > 0 else 1.0)
            h.update(np.round(q * 1000.0).astype('<i8').tobytes())
    key = int.from_bytes(h.digest(), 'little')
    ss = np.random.SeedSequence(entropy=root.entropy,
                                spawn_key=tuple(root.spawn_key) + (int(n), key))
    return np.random.default_rng(ss)


def _reflect_index(k, length):
    """Triangle-wave index 0,1,..,L-1,L-1,..,0,0,1,.. so mirroring can exceed the context."""
    if length <= 1:
        return np.zeros_like(k)
    period = 2 * length
    k = k % period
    return np.where(k < length, k, period - 1 - k)


# ------------------------------------------------------------------ fill internals
def _interp_short(y, gaps):
    """Linear interpolation of all given gaps at once, in place (gap edges are finite)."""
    if gaps.shape[0] == 0:
        return
    N = y.size
    s, e = gaps[:, 0], gaps[:, 1]
    n = e - s
    after = y[np.minimum(e, N - 1)].astype(np.float64)
    before = y[np.maximum(s - 1, 0)].astype(np.float64)
    a = np.where(s > 0, before, after)
    b = np.where(e < N, after, a)
    rep = np.repeat(np.arange(s.size), n)
    off = np.arange(rep.size) - np.repeat(np.cumsum(n) - n, n)
    y[s[rep] + off] = a[rep] + (b[rep] - a[rep]) * ((off + 1) / (n[rep] + 1.0))


def _ramp(T):
    """Raised cosine 0 -> 1 over T samples (end points excluded): zero slope at both ends."""
    return 0.5 * (1.0 - np.cos(np.pi * np.arange(1, T + 1) / (T + 1.0)))


def _blend_edges(y, s, e, nxt, fill, base, T, power_complementary):
    """
    Start/end the fill as the point (odd) reflection of the neighbouring data (value and
    slope continuous) and cross-fade it into ``fill`` over T samples at each edge.
    ``nxt``: first index right of the gap that is not yet finite (next long gap or N).
    """
    if T <= 0:
        return fill
    n = e - s
    r = _ramp(T)
    if power_complementary:      # independent signals: constant power through the fade
        w_odd, w_fill = np.cos(0.5 * np.pi * r), np.sin(0.5 * np.pi * r)
    else:                        # correlated signals (mirror images): linear cross-fade
        w_odd, w_fill = 1.0 - r, r
    k = np.arange(T)
    if s > 0:
        odd = 2.0 * float(y[s - 1]) - y[np.maximum(s - 2 - k, 0)].astype(np.float64)
        b = base[:T]
        fill[:T] = b + w_odd * (odd - b) + w_fill * (fill[:T] - b)
    if e < y.size:
        odd = 2.0 * float(y[e]) - y[np.minimum(e + 1 + k, nxt - 1)].astype(np.float64)
        pos = n - 1 - k
        b = base[pos]
        fill[pos] = b + w_odd * (odd - b) + w_fill * (fill[pos] - b)
    return fill


def _mirror_core(y, s, e, nxt, ctx, lvl_a, lvl_b, base):
    n = e - s
    k = np.arange(n)
    n_left = min(ctx, s)                 # left of s everything is finite (filled in order)
    n_right = min(ctx, nxt - e)          # contiguous finite samples up to the next long gap
    left_img = (y[s - 1 - _reflect_index(k, n_left)].astype(np.float64) - lvl_a
                if n_left else None)
    right_img = (y[e + _reflect_index(n - 1 - k, n_right)].astype(np.float64) - lvl_b
                 if n_right else None)
    if left_img is None:
        return base + right_img
    if right_img is None:
        return base + left_img
    w = 1.0 - _ramp(n)
    den = np.sqrt(np.sum(left_img ** 2) * np.sum(right_img ** 2))
    rho = float(np.clip(np.sum(left_img * right_img) / den, 0.0, 1.0)) if den > 0 else 0.0
    # std of w*L + (1-w)*R relative to one image, for images with correlation rho
    g = np.sqrt(w ** 2 + (1 - w) ** 2 + 2 * w * (1 - w) * rho)
    return base + (w * left_img + (1 - w) * right_img) / g


def _conditioned_noise(power, m, n, y_left, y_right, base_ext, clip, W, rng):
    """
    Gap fill minus the level bridge: Gaussian noise with variance ``power[k]`` per rfft bin
    of a length-m circular synthesis, **conditioned on the observed samples next to the gap**
    (conditional simulation / kriging of the Gaussian process with that spectrum).

    The noise ``z`` is synthesised on [left obs | gap | right obs | pad]; the fill is
    ``z + K (d - z_obs)`` with ``K = C_gap,obs C_obs,obs^-1`` from the autocovariance of the
    same spectrum, i.e. an exact draw from the process given the edge samples: it continues
    the data as far as the data's own autocorrelation reaches (a smooth signal keeps value
    and slope, white noise does not carry anything over) and has the right variance
    everywhere. ``d`` are the residuals of the observations from the level bridge,
    clipped to +-``clip`` so that an artifact or spike at the edge is not continued into
    the gap. The conditioning term is applied within ``W`` samples of each edge (the whole
    gap if it is shorter than 2 W) and faded out over the second half of W; ``W = 0``
    gives plain (unconditioned) noise.
    """
    pl, pr = y_left.size, y_right.size
    z = _synthesize(power, m, rng)
    r0 = float(power.sum())
    if W <= 0 or pl + pr == 0 or r0 <= 0:
        return z[pl:pl + n]
    Y = power.copy()
    if m % 2 == 0:
        Y[-1] *= 2.0
    R = np.fft.irfft(Y, m)                      # circular autocovariance of z / (m / 2)
    R *= m / 2.0
    obs = np.r_[np.arange(pl), np.arange(pl + n, pl + n + pr)]       # positions in z
    d = np.r_[y_left, y_right] - base_ext(obs - pl)
    if clip > 0:
        d = np.clip(d, -clip, clip)
    C = R[np.abs(obs[:, None] - obs[None, :])]
    C[np.diag_indices_from(C)] += _NUGGET * r0
    try:
        a = np.linalg.solve(C, d - z[obs])
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(C, d - z[obs], rcond=None)[0]
    out = z[pl:pl + n]
    full = (n <= 2 * W) if (pl and pr) else (n <= W)
    if full:                                    # correction R * a over the whole gap (FFT)
        del R
        spikes = np.zeros(m)
        spikes[obs] = a
        corr = np.fft.irfft(np.fft.rfft(spikes) * Y, m)[pl:pl + n]
        return out + corr * (m / 2.0)
    h = W // 2
    fade = np.r_[np.ones(h), 0.5 * (1.0 + np.cos(np.pi * np.arange(1, W - h + 1) / (W - h + 1.0)))]
    regions = ([np.arange(W)] if pl else []) + ([np.arange(n - W, n)] if pr else [])
    fades = ([fade] if pl else []) + ([fade[::-1]] if pr else [])
    for t, f in zip(regions, fades):            # long gap: only the edge regions (gather)
        out[t] += f * (R[np.abs((t + pl)[:, None] - obs[None, :])] @ a)
    return out


def _fill_long(y, ok, s, e, nxt, fs, p, root):
    n = e - s
    N = y.size
    method = p['method']
    if method == 'spectral':
        nps = _nperseg(n, fs, p['ctx'])
        # context per side: ~16 gap lengths (rounded up to 2^k) but at most 7 Welch segments
        # (4 nps) unless that is shorter than 3 segments (2 nps)
        L = int(min(p['ctx'], max(2 * nps, min(16 * (1 << int(np.ceil(np.log2(max(n, 2))))),
                                                4 * nps))))
    else:
        nps, L = 0, p['ctx']
    lo, hi = max(s - L, 0), min(e + L, N)
    ctx_l, ok_l = y[lo:s], ok[lo:s]
    ctx_r, ok_r = y[e:hi], ok[e:hi]
    left, right = ctx_l[ok_l].astype(np.float64), ctx_r[ok_r].astype(np.float64)
    if left.size + right.size == 0:          # only filled samples nearby (dense long gaps)
        ok_l, ok_r = np.isfinite(ctx_l), np.isfinite(ctx_r)
        left, right = ctx_l[ok_l].astype(np.float64), ctx_r[ok_r].astype(np.float64)
    near = max(min(int(round(0.5 * fs)), L), 1)
    lvl_a = float(np.median(left[-near:])) if left.size else float(np.median(right[:near]))
    lvl_b = float(np.median(right[:near])) if right.size else lvl_a

    def base_ext(t):
        return lvl_a + (lvl_b - lvl_a) * ((np.asarray(t) + 1.0) / (n + 1.0))

    base = base_ext(np.arange(n))
    two_sided = s > 0 and e < N
    t_cap = n // 2 if two_sided else n

    if method == 'mirror':
        fill = _mirror_core(y, s, e, nxt, p['ctx'], lvl_a, lvl_b, base)
        tk = min(max(int(round(_MIRROR_KINK_S * fs)), 2), t_cap)
        y[s:e] = _blend_edges(y, s, e, nxt, fill, base, tk, power_complementary=False)
        return

    rng = _gap_rng(root, n, left, right)
    # observations conditioning the edges: a few samples (~20 ms, 4..64) on each side
    q = int(min(max(round(_COND_S * fs), 4), 64))
    y_left = y[max(s - q, 0):s].astype(np.float64)
    y_right = y[e:min(e + q, nxt)].astype(np.float64)
    pad = max(nps, 2 * q, 16)
    m = int(_sfft.next_fast_len(n + y_left.size + y_right.size + pad, real=True))
    cum = None
    if method == 'spectral':
        cum = _context_psd([(ctx_l, ok_l), (ctx_r, ok_r)], nps, fs)
    if cum is not None:
        power = _bin_power(m, fs, cum)
    else:                                   # 'pink', or 'spectral' with too little context
        k = np.arange(m // 2 + 1, dtype=float)
        power = np.zeros(k.size)
        keep = k >= m / float(n)            # lowest frequency 1 / gap, as pink_noise
        power[keep] = k[keep] ** (-p['beta'])
        tot = power.sum()
        sd = _robust_sd(np.r_[left, right])
        power = power * (sd ** 2 / tot) if tot > 0 else power
    nl, nr = min(near, left.size), min(near, right.size)
    resid = np.r_[left[left.size - nl:] - lvl_a, right[:nr] - lvl_b]
    clip = _CLIP_SDS * _robust_sd(resid[::max(1, resid.size // 4096)])
    W = min(p['taper'], t_cap)
    noise = _conditioned_noise(power, m, n, y_left, y_right, base_ext, clip, W, rng)
    y[s:e] = base + noise


def _fill_row(y, bad, fs, p, root):
    """Fill one 1-D row in place; ``bad = ~isfinite(y)``, not all True."""
    gaps = _runs(bad)
    if p['method'] == 'linear':
        is_long = np.zeros(gaps.shape[0], dtype=bool)
    else:
        is_long = (gaps[:, 1] - gaps[:, 0]) > p['max_interp']
    _interp_short(y, gaps[~is_long])
    long_gaps = gaps[is_long]
    if long_gaps.shape[0] == 0:
        return
    ok = ~bad                       # ORIGINAL samples only (not the interpolated short gaps)
    nxt = np.r_[long_gaps[1:, 0], y.size]
    for (s, e), nx in zip(long_gaps, nxt):
        _fill_long(y, ok, int(s), int(e), int(nx), fs, p, root)


def fill_gaps(x, fs, *, max_interp_s=0.1, method='spectral', context_s=30.0, taper_s=0.5,
              beta=1.0, seed=0, axis=-1, all_nan='keep', copy=True):
    """
    Fill NaN/inf gaps so a detector can run on the signal (see the module docstring for
    the pipeline, the methods and the evidence behind the defaults).

    Parameters
    ----------
    x : array_like
        Signal, 1-D ``(n_samples,)`` or N-D with time along ``axis``. Non-finite samples
        (NaN, +inf, -inf) are gaps.
    fs : float
        Sampling frequency (Hz).
    max_interp_s : float
        Gaps up to this length (seconds) are linearly interpolated; longer gaps use
        ``method``. Default 0.1 s.
    method : {'spectral', 'pink', 'mirror', 'linear'}
        Fill for long gaps. ``'spectral'`` (default): noise with the band powers (robust
        Welch PSD) of the neighbouring data. ``'pink'``: 1/f^beta noise at the MAD
        amplitude of the neighbouring data. ``'mirror'``: neighbouring signal mirrored in
        from both sides (copies neighbouring events into the gap). ``'linear'``: straight
        line (not suitable for gaps longer than ~0.1 s before background-modelling
        detectors).
    context_s : float
        Maximum seconds of valid data used on each side of a long gap (spectrum, level,
        amplitude, mirror images). Default 30 s. ``'spectral'`` uses less for shorter
        gaps (about 16 gap lengths, at least three ~1 s Welch segments).
    taper_s : float
        ``'spectral'``/``'pink'``: how far (seconds) into the gap the fill is conditioned
        on the data at each edge (the whole gap if it is shorter than ``2 * taper_s``;
        faded out over the second half). Default 0.5 s. 0 disables the conditioning (the
        noise then starts with a step; not recommended).
    beta : float
        Spectral exponent for ``'pink'`` (and for the ``'spectral'`` fallback when fewer
        than 16 valid context samples exist).
    seed : None, int, sequence of int, SeedSequence or Generator
        Root of the per-gap random streams (see "Randomness" in the module docstring).
        Default 0: reproducible. A gap's noise depends only on the seed, the gap's
        length and the data next to it (quantised to 1/1000 of its robust SD), not on
        its position in the array.
    axis : int
        Time axis for N-D input; every other index is a channel, filled independently.
    all_nan : {'keep', 'zero', 'raise'}
        A channel without any finite sample: ``'keep'`` (default) leaves it unchanged
        (non-finite) and warns once, listing the channel indices; ``'zero'`` fills it with
        zeros and warns; ``'raise'`` raises ``ValueError``.
    copy : bool
        If False and ``x`` is a writable floating-point ndarray, it is filled **in place**
        and returned (no extra memory). Otherwise a filled copy is returned.

    Returns
    -------
    np.ndarray
        Same shape as ``x``; same dtype for floating-point input (float32 stays float32),
        float64 otherwise. Finite everywhere except all-NaN channels with
        ``all_nan='keep'``. Valid samples are never changed.

    Notes
    -----
    Always remove detections in gaps afterwards with :func:`drop_in_gaps` /
    :func:`mask_in_gaps`, using gaps found on the original signal.
    """
    fs = _check_fs(fs)
    if method not in _METHODS:
        raise ValueError(f'unknown method {method!r}; use one of {_METHODS}')
    if all_nan not in _ALL_NAN:
        raise ValueError(f'unknown all_nan {all_nan!r}; use one of {_ALL_NAN}')
    max_interp_s = _check_nonneg('max_interp_s', max_interp_s)
    context_s = _check_nonneg('context_s', context_s, strict=True)
    taper_s = _check_nonneg('taper_s', taper_s)
    try:
        beta = float(beta)
    except (TypeError, ValueError):
        raise ValueError(f'beta must be a finite number, got {beta!r}') from None
    if not np.isfinite(beta):
        raise ValueError(f'beta must be a finite number, got {beta!r}')
    root = _root_seed(seed)

    x_in = x
    x = np.asarray(x)
    if x.ndim == 0:
        raise ValueError('fill_gaps expects at least a 1-D signal')
    if not isinstance(axis, numbers.Integral) or not -x.ndim <= axis < x.ndim:
        raise ValueError(f'axis {axis!r} out of range for a {x.ndim}-D signal')
    if x.dtype != bool and (not np.issubdtype(x.dtype, np.number)
                            or np.issubdtype(x.dtype, np.complexfloating)):
        raise TypeError(f'fill_gaps expects a real numeric signal, got dtype {x.dtype}')
    if np.issubdtype(x.dtype, np.floating):
        in_place = (not copy) and isinstance(x_in, np.ndarray) and x.flags.writeable
        out = x if in_place else x.copy()
    else:
        out = x.astype(np.float64)

    bad = ~np.isfinite(out)
    if not bad.any():
        return out

    p = dict(method=method, max_interp=int(np.floor(max_interp_s * fs + 1e-9)),
             ctx=max(int(round(context_s * fs)), 2), taper=int(round(taper_s * fs)),
             beta=beta)
    view = np.moveaxis(out, axis, -1)
    bad_v = np.moveaxis(bad, axis, -1)
    all_bad = bad_v.all(axis=-1)
    if view.ndim == 1:
        rows = [] if all_bad else [()]
    else:
        rows = list(zip(*np.nonzero(bad_v.any(axis=-1) & ~all_bad)))

    if np.any(all_bad):
        if view.ndim == 1:
            what = 'the signal is'
        else:
            chans = [tuple(int(i) for i in idx) if len(idx) > 1 else int(idx[0])
                     for idx in zip(*np.nonzero(all_bad))]
            what = f'{len(chans)} channel(s) {chans} (index over the non-time axes) are'
        if all_nan == 'raise':
            raise ValueError(f'fill_gaps: {what} entirely NaN/inf and cannot be filled.')
        if all_nan == 'zero':
            view[all_bad] = 0
            msg = 'filled with zeros'
        else:
            msg = 'left unchanged (still non-finite)'
        warnings.warn(f'fill_gaps: {what} entirely NaN/inf; {msg}.', RuntimeWarning,
                      stacklevel=2)

    for idx in rows:
        _fill_row(view[idx], bad_v[idx].copy(), fs, p, root)
    return out


# ------------------------------------------------------------------ post-filtering
_INTEGRAL_TOL = 1e-6             # |v - round(v)| accepted as an integer sample index


def _to_seconds(name, v, units, fs, units_name):
    v = np.asarray(v)
    if not np.issubdtype(v.dtype, np.number) or np.issubdtype(v.dtype, np.complexfloating):
        raise TypeError(f'{name} must be real numbers, got dtype {v.dtype}')
    vf = v.astype(np.float64)
    if not np.isfinite(vf).all():
        raise ValueError(f'{name} contains NaN/inf')
    if units == 'seconds':
        if vf.size and np.issubdtype(v.dtype, np.integer):
            raise ValueError(
                f"{name} is integer-typed (sample indices?) but {units_name}='seconds'. "
                f"Use {units_name}='samples' for sample indices; if they really are whole "
                f"seconds, pass them as floats (np.asarray({name}, float)).")
        return vf
    if not np.issubdtype(v.dtype, np.integer):
        r = np.round(vf)
        if np.any(np.abs(vf - r) > _INTEGRAL_TOL):
            raise ValueError(
                f"{name} has non-integer values but {units_name}='samples'. Are they in "
                f"seconds? Then use {units_name}='seconds'. (Values within {_INTEGRAL_TOL:g} "
                "of an integer, e.g. t * fs, are accepted as sample indices.)")
        vf = r
    return vf / fs


def mask_in_gaps(det, gaps, fs, *, units, gap_units, margin_s=0.1, end=None):
    """
    Boolean mask of detections that are in or near a gap (gap widened by ``margin_s`` on
    each side).

    Parameters
    ----------
    det : array_like
        Detection positions, or interval starts, in ``units``.
    gaps : array_like, shape (n_gaps, 2)
        ``[start, stop)`` of each gap in ``gap_units``: :func:`find_gaps` gives samples,
        :func:`gap_intervals` gives seconds. Need not be sorted; may overlap.
    fs : float
        Sampling frequency (Hz). Always required (``margin_s`` is in seconds).
    units : {'seconds', 'samples'}
        Units of ``det`` (and ``end``). Required, no default.
    gap_units : {'seconds', 'samples'}
        Units of ``gaps``. Required, no default. The two are separate on purpose: a
        mix-up (e.g. sample-index detections against gaps converted to seconds) would
        silently mask nothing, so you state each one.
        Integer-typed values with ``'seconds'`` raise ``ValueError`` (pass whole seconds
        as floats); with ``'samples'``, float values must be integral within 1e-6
        (``t * fs`` is fine) or ``ValueError`` is raised.
    margin_s : float
        Exclusion margin around every gap, in **seconds**, >= 0. Default 0.1 s.
    end : array_like, optional
        Interval ends (same units and shape, ``end >= det``) for interval detections; an
        interval is masked if any part of ``[det, end]`` overlaps a widened gap.

    Returns
    -------
    np.ndarray of bool, shape of ``np.atleast_1d(det)``; True = drop the detection.
    """
    fs = _check_fs(fs)
    for nm, u in (('units', units), ('gap_units', gap_units)):
        if u not in _UNITS:
            raise ValueError(f"{nm} must be 'seconds' or 'samples', got {u!r}")
    margin_s = _check_nonneg('margin_s', margin_s)
    g_raw = np.asarray(gaps)
    if g_raw.size == 0:
        g_raw = g_raw.reshape(0, 2)
    if g_raw.ndim != 2 or g_raw.shape[1] != 2:
        raise ValueError(f'gaps must have shape (n_gaps, 2), got {g_raw.shape}')
    g = _to_seconds('gaps', g_raw, gap_units, fs, 'gap_units')
    if np.any(g[:, 1] < g[:, 0]):
        raise ValueError('gaps must have stop >= start')
    start = np.atleast_1d(_to_seconds('det', det, units, fs, 'units'))
    if end is None:
        stop = start
    else:
        stop = np.atleast_1d(_to_seconds('end', end, units, fs, 'units'))
        if stop.shape != start.shape:
            raise ValueError(f'end has shape {stop.shape}, det has shape {start.shape}')
        if np.any(stop < start):
            raise ValueError('end must be >= det for every detection')
    out = np.zeros(start.shape, dtype=bool)
    if g.shape[0] == 0 or start.size == 0:
        return out
    lo = g[:, 0] - margin_s
    hi = g[:, 1] + margin_s
    # interval [start, stop] overlaps [lo, hi) iff start < hi and stop >= lo
    order = np.argsort(lo, kind='stable')
    lo, hi = lo[order], np.maximum.accumulate(hi[order])
    k = np.searchsorted(lo, stop, side='right') - 1    # last widened gap starting <= stop
    valid = k >= 0
    out[valid] = start[valid] < hi[k[valid]]
    return out


def drop_in_gaps(det, gaps, fs, *, units, gap_units, margin_s=0.1, end=None):
    """
    Remove the detections flagged by :func:`mask_in_gaps` (same arguments).

    Returns
    -------
    np.ndarray, or tuple ``(det, end)`` when ``end`` is given
        The kept detections (and interval ends), with their original dtype.
    """
    m = ~mask_in_gaps(det, gaps, fs, units=units, gap_units=gap_units, margin_s=margin_s,
                      end=end)
    d = np.atleast_1d(np.asarray(det))
    if end is None:
        return d[m]
    return d[m], np.atleast_1d(np.asarray(end))[m]
