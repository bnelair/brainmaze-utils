# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Tools for digital signal processing: decimation/resampling, FFT filtering, low-frequency
filtering utilizing downsampling, buffering etc.

**Conventions**

- Multichannel arrays are ``(n_signals, n_samples)``: time runs along the **last** axis.
- Sampling frequencies and cutoffs are in Hz, durations in seconds.
- Functions do not modify their inputs.
- NaN marks missing data (gaps); each function documents how it treats NaN.

**Which functions filter?**

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Function
     - Anti-aliasing / filtering
     - NaN handling
   * - :func:`decimate`
     - **yes**: 16th-order Butterworth (SOS, zero-phase), default cutoff ``fs_new / 3``
     - interpolated for filtering, re-masked in the output
   * - :func:`nandecimate`
     - yes, weak: 30-tap FIR (legacy; prefer :func:`decimate`)
     - mean-filled for filtering, re-masked in the output
   * - :func:`resample`
     - **no** - interpolation only, by design; low-pass first when downsampling
     - never interpolated across; an output is NaN if a bracketing input sample is.
       When downsampling, a gap shorter than one output period can vanish
   * - :func:`unify_sampling_frequency`
     - yes (calls :func:`decimate`)
     - as :func:`decimate`
   * - :func:`downsample_min_max`
     - no (min/max envelope, for display)
     - NaN ignored within a window
   * - :class:`LowFrequencyFilter`
     - is itself a zero-phase low-/high-pass
     - short gaps bridged, long gaps split into separately filtered segments; NaN
       out exactly where NaN in (``inf`` raises)
   * - :func:`fft_filter`
     - is itself a brick-wall low-/high-pass
     - as :class:`LowFrequencyFilter`
"""

import warnings

import numpy as np
import scipy.signal as signal
import scipy.fft as fft

def get_datarate(x):
    """
    Fraction of valid (non-NaN) samples per signal.

    Parameters
    ----------
    x : numpy.ndarray or list of numpy.ndarray
        A single signal ``(n_samples,)``, a stack of signals ``(n_signals, n_samples)``
        (samples along the **last** axis), or a list of 1-D signals of possibly
        different lengths.

    Returns
    -------
    float, numpy.ndarray or list
        Values in ``[0, 1]``: ``1 - n_nan / n_samples``.

        - 1-D input -> ``float``
        - N-D input -> ``numpy.ndarray`` of shape ``x.shape[:-1]``
        - list input -> ``list`` of floats (one per element)
    """
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return float(1 - np.isnan(x).sum() / x.shape[0])
        return 1 - (np.isnan(x).sum(axis=-1) / x.shape[-1])
    return [1 - (np.isnan(x_).sum() / np.asarray(x_).size) for x_ in x]


def _fill_nans_linear(x):
    """
    Fill NaNs in each row of a 2-D float array (in place) by linear interpolation
    between the neighbouring valid samples; leading/trailing NaNs take the nearest
    valid value. All-NaN rows are set to 0 (the caller is expected to re-mask them).
    """
    idx = np.arange(x.shape[1])
    for row in x:
        nans = np.isnan(row)
        if not nans.any():
            continue
        if nans.all():
            row[:] = 0.0
            continue
        row[nans] = np.interp(idx[nans], idx[~nans], row[~nans])
    return x


def _decimated_nan_mask(nans, fs, fs_new, n_new):
    """
    Map an input NaN mask ``(n_signals, n_samples)`` onto an output grid of ``n_new``
    samples at ``fs_new`` (output sample ``k`` sits at time ``k / fs_new``).

    Output sample ``k`` is flagged if **any** input sample within half an output
    sample period (``+-0.5 / fs_new``) of its time stamp is NaN, or if one of the two
    input samples bracketing its time stamp is NaN.
    """
    n = nans.shape[1]
    out = np.zeros((nans.shape[0], n_new), dtype=bool)
    if not nans.any():
        return out
    # cumulative count -> O(1) "any NaN in [lo, hi)" per output sample
    csum = np.concatenate((np.zeros((nans.shape[0], 1), dtype=np.int64),
                           np.cumsum(nans, axis=1, dtype=np.int64)), axis=1)
    t = np.arange(n_new) / fs_new
    lo = np.clip(np.ceil((t - 0.5 / fs_new) * fs - 1e-9).astype(np.int64), 0, n)
    hi = np.clip(np.floor((t + 0.5 / fs_new) * fs + 1e-9).astype(np.int64) + 1, 0, n)
    hi = np.maximum(hi, np.minimum(lo + 1, n))
    out = (csum[:, hi] - csum[:, lo]) > 0
    # the two input samples bracketing each output time are always included (matters
    # when upsampling, where the +-0.5 / fs_new window may contain no input sample)
    pos = np.minimum(np.arange(n_new) * fs / fs_new, n - 1)
    i_lo = np.clip(np.floor(pos + 1e-9).astype(np.int64), 0, n - 1)
    i_hi = np.clip(np.ceil(pos - 1e-9).astype(np.int64), 0, n - 1)
    out |= nans[:, i_lo] | nans[:, i_hi]
    return out


# exact-time band-limited interpolation (used by decimate for non-integer ratios and upsampling)
_INTERP_ATTEN_DB = 120.0   # Kaiser design target: ~1e-6 relative error for content inside the band
_INTERP_MAX_BAND = 0.45    # widest band (relative to the input rate) the interpolator preserves
_INTERP_PHASES = 4096      # kernel table resolution (linear interpolation between phases)


def _interp_kernel(band):
    """
    Kaiser-windowed sinc reconstruction kernel for content up to ``band`` (relative to
    the input rate, ``<= _INTERP_MAX_BAND``).

    Returns ``(W, taps, table)``: half-width ``W`` in input samples, tap offsets
    ``-W+1 .. W`` and the (unnormalised) kernel sampled at ``_INTERP_PHASES + 1``
    fractional positions ``0 .. 1``. The transition band is ``[band, 1 - band]``
    (images of the content start at ``1 - band``), and the window is designed for
    ``_INTERP_ATTEN_DB`` attenuation (Kaiser's formula).
    """
    from scipy.special import i0
    band = min(max(band, 0.0), _INTERP_MAX_BAND)
    d_omega = 2 * np.pi * (1.0 - 2.0 * band)
    n_taps = (_INTERP_ATTEN_DB - 7.95) / (2.285 * d_omega) + 1
    W = max(2, int(np.ceil(n_taps / 2)))
    beta = 0.1102 * (_INTERP_ATTEN_DB - 8.7)
    taps = np.arange(-W + 1, W + 1)
    frac = np.arange(_INTERP_PHASES + 1) / _INTERP_PHASES
    t = frac[:, None] - taps[None, :]  # distance between output time and tap, in input samples
    u = np.clip(1.0 - (t / W) ** 2, 0.0, None)
    table = np.sinc(t) * i0(beta * np.sqrt(u)) / i0(beta)
    return W, taps, table


def _odd_extend(x, m):
    """Extend 2-D ``x`` by ``m`` samples at both ends of the last axis by odd (point) reflection."""
    n = x.shape[1]
    if n < 2:
        return np.repeat(x, 2 * m + 1, axis=1)
    r = min(m, n - 1)
    head = 2 * x[:, :1] - x[:, r:0:-1]
    tail = 2 * x[:, -1:] - x[:, -2:-r - 2:-1]
    if r < m:
        head = np.concatenate((np.repeat(head[:, :1], m - r, axis=1), head), axis=1)
        tail = np.concatenate((tail, np.repeat(tail[:, -1:], m - r, axis=1)), axis=1)
    return np.concatenate((head, x, tail), axis=1)


def _interp_exact_times(x, fs, fs_new, n_new, band):
    """
    Evaluate the band-limited signal ``x`` ``(n_signals, n_samples)`` sampled at ``fs``
    at the exact times ``k / fs_new``, ``k = 0 .. n_new - 1``.

    Each output value is a Kaiser-windowed sinc interpolation over ``2 * W`` input
    samples (see :func:`_interp_kernel`), weights normalised to sum to 1. Positions are
    computed per sample as ``k * fs / fs_new``, so there is no cumulative timing drift
    (float64: < 1e-6 samples even for days of data). The record is extended by odd
    reflection at both ends, which also covers output times up to one input period
    after the last input sample (upsampling).
    """
    n = x.shape[1]
    W, taps, table = _interp_kernel(band)
    pad = W + 1
    xp = _odd_extend(x, pad)
    out = np.empty((x.shape[0], n_new))
    block = max(1, (1 << 22) // (2 * W * max(1, x.shape[0])))
    for s in range(0, n_new, block):
        e = min(n_new, s + block)
        # output times may lie up to one input period past the last sample when
        # upsampling; the odd extension covers that (no clamping to the last value)
        p = np.minimum(np.arange(s, e, dtype=float) * fs / fs_new, float(n))
        ib = np.floor(p).astype(np.int64)
        q = (p - ib) * _INTERP_PHASES
        qi = np.minimum(np.floor(q).astype(np.int64), _INTERP_PHASES - 1)
        a = (q - qi)[:, None]
        w = table[qi] * (1.0 - a) + table[qi + 1] * a
        w /= w.sum(axis=1, keepdims=True)
        idx = ib[:, None] + taps[None, :] + pad
        out[:, s:e] = np.einsum('ckj,kj->ck', xp[:, idx], w)
    return out


def decimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Resample signal(s) to ``fs_new`` with a zero-phase anti-aliasing low-pass filter.
    NaN-aware. Despite the name it also upsamples (``fs_new > fs``).

    Processing steps, per signal:

    1. NaN samples are temporarily filled by linear interpolation between the
       neighbouring valid samples (leading/trailing NaNs take the nearest valid
       value) so the filter does not see steps at gap edges.
    2. Zero-phase low-pass at ``cutoff``: 16th-order Butterworth in second-order
       sections, applied forward and backward (:func:`scipy.signal.sosfiltfilt`); the
       magnitude response is therefore that of a 32nd-order filter with -6 dB at
       ``cutoff``. When upsampling it is skipped if ``cutoff >= 0.45 * fs``: the
       input holds nothing above ``fs / 2``, and the interpolator (step 3) already
       band-limits to ``0.45 * fs``.
    3. Output sample ``k`` is the filtered signal at time exactly ``k / fs_new`` s;
       the output has ``n_new = round(n_samples * fs_new / fs)`` samples.

       - Integer ratios ``fs / fs_new`` pick every ``q``-th sample (exact).
       - Any other ratio, including upsampling and rates with decimals such as
         30000.5 or 511.9999 Hz, evaluates the band-limited signal at the exact times
         with a Kaiser-windowed sinc interpolator (2 x 5 to 2 x 40 taps, designed for
         120 dB). The position of every output sample is computed directly from
         ``k * fs / fs_new``, so there is **no timing drift**, however long the record
         (timing error < 1e-6 samples). Interpolation error for content inside the
         band is ~1e-7 of its amplitude (measured on tones; the integer path is
         exact).
    4. The NaN mask is re-applied: an output sample is NaN if any input sample
       within ``+-0.5 / fs_new`` s of it, or one of the two input samples bracketing
       it, was NaN. Signals that are entirely NaN stay entirely NaN.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_samples,)`` or ``(n_signals, n_samples)`` - samples along the last axis.
        Not modified.
    fs : float
        Sampling frequency of ``x`` in Hz (``> 0``).
    fs_new : float
        Target sampling frequency in Hz (``> 0``). May be lower, equal or higher
        than ``fs``.
    cutoff : float, optional
        Anti-aliasing cutoff in Hz. Default ``fs_new / 3`` (two thirds of the new
        Nyquist frequency). When downsampling it must satisfy
        ``0 < cutoff < fs_new / 2``; a higher cutoff would let content above the new
        Nyquist frequency alias into the output. When upsampling it must be ``> 0``;
        if it is ``>= 0.45 * fs`` no filter is applied.
    datarate : bool
        If ``True``, also return :func:`get_datarate` of the **input**.

    Returns
    -------
    numpy.ndarray or tuple
        Resampled signal(s) with the same number of dimensions as ``x``
        (``float64``), or ``(resampled, datarate)`` if ``datarate=True``.

    Raises
    ------
    ValueError
        If ``fs`` or ``fs_new`` is not positive, if ``cutoff`` is outside the range
        given above, or if ``x`` has more than 2 dimensions.

    Notes
    -----
    - ``fs_new == fs`` still applies the low-pass at ``cutoff`` (as in v2.0.0).
    - Upsampling: with the default cutoff the low-pass is applied only for ratios
      ``fs_new / fs < 1.35`` (where ``fs_new / 3 < 0.45 * fs``); v2.0.0 applied it up
      to 1.5, returned NaN between ~1.45 and 1.5 and raised above. The
      interpolator preserves content up to ``0.45 * fs`` (or ``1.25 * cutoff``, if
      lower); content between ``0.45 * fs`` and ``fs / 2`` is attenuated. When the
      Butterworth is skipped (``cutoff >= 0.45 * fs``) such content is not removed but
      leaves an image at ``fs - f``: a tone at ``0.48 * fs`` comes out with gain ~0.93
      plus a ~7 % image.
    - When upsampling, ``round(n * fs_new / fs)`` output samples can include up to 3
      samples **after** the last input sample (e.g. 250->1000 Hz). They are extrapolated
      (odd extension) and are the least reliable part of the output: on a unit-amplitude
      tone their error is ~0.1 at 20 Hz and reaches ~1 for tones near 0.4 * fs. Discard
      them if the end of the record matters.
    - Samples close to (but outside) a NaN gap are computed from the interpolated
      fill and the filter's impulse response, so a few output samples next to a gap
      may carry a small transient; they are not masked.
    - The record edges are extended by odd reflection over ``~6 * fs / cutoff``
      samples before filtering, which keeps edge transients small (~1e-3 of the
      amplitude for an in-band tone) but not zero.

    .. note:: **Changed after v2.0.0:**
       Filter is now applied in second-order sections; the previous ``(b, a)``
       16th-order design was numerically unstable for ``fs / fs_new >= ~10`` and
       returned all-NaN output (e.g. 3000->250, 1000->50, 32000->1000 Hz). NaNs are
       now re-applied to the output instead of being silently filled. The final
       resampling step no longer uses the FFT (:func:`scipy.signal.resample`, which
       assumes a periodic signal and wraps the end of the record into its start); it
       picks samples (integer ratios) or interpolates at exact times. Upsampling by
       ratios ``>= 1.5`` used to raise in the filter design and now works. A
       ``cutoff`` at or above the new Nyquist frequency when downsampling now raises
       ``ValueError`` (it used to alias silently).
    """
    return_datarate = datarate is True
    x = np.array(x, dtype=float)  # copy, also promotes ints

    if not (fs > 0 and fs_new > 0):
        raise ValueError(f'fs and fs_new must be > 0, got fs={fs!r}, fs_new={fs_new!r}')

    dr = get_datarate(x) if return_datarate else None

    downsampling = fs_new <= fs
    if cutoff is None:
        cutoff = fs_new / 3  # two thirds of the new Nyquist
    if not cutoff > 0:
        raise ValueError(f'cutoff must be > 0, got {cutoff!r}')
    if downsampling and not cutoff < fs_new / 2:
        raise ValueError(
            f'cutoff={cutoff!r} Hz must be below the new Nyquist frequency fs_new/2 = {fs_new / 2!r} Hz; '
            'content above it would alias into the output')
    # when upsampling, a cutoff at/above the interpolator's band (0.45 fs) is pointless: a
    # Butterworth that close to Nyquist only adds error (v2.0.0 returned NaN there)
    apply_filter = cutoff < fs / 2 if downsampling else cutoff < _INTERP_MAX_BAND * fs

    b_multiple_signals = x.ndim != 1
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f'x must be 1-D or 2-D, got shape {x.shape}')

    n = x.shape[1]
    n_new = int(np.round((fs_new / fs) * n))
    if n == 0 or n_new == 0:
        y = np.zeros((x.shape[0], n_new))
        y[:] = np.nan
        y = y if b_multiple_signals else y[0]
        return (y, dr) if return_datarate else y

    nans = np.isnan(x)
    _fill_nans_linear(x)

    if apply_filter:
        sos = signal.butter(16, cutoff, 'lp', fs=fs, output='sos')
        # long odd-extension padding (~6 cutoff periods) keeps edge transients small
        padlen = int(min(n - 1, max(np.ceil(6 * fs / cutoff), 3 * (2 * sos.shape[0] + 1))))
        x = signal.sosfiltfilt(sos, x, axis=1, padlen=padlen)

    ratio = fs / fs_new
    if downsampling and abs(ratio - round(ratio)) < 1e-9:
        y = x[:, ::int(round(ratio))][:, :n_new]
        if y.shape[1] < n_new:  # cannot happen for an exact integer ratio; defensive
            y = np.concatenate((y, np.repeat(y[:, -1:], n_new - y.shape[1], axis=1)), axis=1)
    else:
        band = min(1.25 * cutoff, fs / 2) / fs if apply_filter else _INTERP_MAX_BAND
        y = _interp_exact_times(x, fs, fs_new, n_new, band)

    y[_decimated_nan_mask(nans, fs, fs_new, n_new)] = np.nan

    if not b_multiple_signals:
        y = y[0]

    if return_datarate:
        return y, dr
    return y


def nandecimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Downsample signal(s) with a short FIR anti-aliasing filter. NaN-aware.

    Legacy, lightweight variant of :func:`decimate`: NaNs are filled with the
    channel mean, the signal is low-passed with a 30-tap FIR (``firwin``, applied
    forward-backward), resampled with :func:`scipy.signal.resample` (FFT), and the
    NaN mask is re-applied (an output sample is NaN if any input sample within
    ``+-0.5 / fs_new`` s of it was NaN).

    .. warning::
       A 30-tap FIR has a wide transition band, so anti-aliasing is weak for large
       ratios ``fs / fs_new``. Prefer :func:`decimate` for analysis.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_samples,)`` or ``(n_signals, n_samples)`` - samples along the last axis.
        Not modified.
    fs : float
        Sampling frequency of ``x`` in Hz.
    fs_new : float
        Target sampling frequency in Hz.
    cutoff : float, optional
        FIR cutoff in Hz. Default ``fs_new / 3``.
    datarate : bool
        If ``True``, also return :func:`get_datarate` of the **input**.

    Returns
    -------
    numpy.ndarray or tuple
        Decimated signal(s) (``round(n_samples * fs_new / fs)`` samples), or
        ``(decimated, datarate)`` if ``datarate=True``.

    Notes
    -----
    .. note:: **Changed after v2.0.0:**
       NaNs were previously filled with the NaN *fraction* of the channel (a value
       in [0, 1]) instead of the channel mean, producing large spurious transients
       next to every gap for signals with a DC offset.
    """
    return_datarate = datarate is True
    x = np.array(x, dtype=float)
    dr = get_datarate(x) if return_datarate else None

    if cutoff is None:
        cutoff = fs_new / 3  # two 3rds of half-sampling

    b_multiple_signals = x.ndim != 1
    if x.ndim == 1:
        x = x.reshape(1, -1)

    nans = np.isnan(x)
    for idx in range(x.shape[0]):
        if nans[idx].all():
            x[idx, :] = 0.0
        elif nans[idx].any():
            x[idx, nans[idx, :]] = np.nanmean(x[idx, :])

    b = signal.firwin(30, cutoff / (0.5 * fs), pass_zero=True)
    b /= b.sum()
    x = signal.filtfilt(b, [1], x, axis=1)

    n_resampled = int(np.round((fs_new / fs) * x.shape[1]))
    x = signal.resample(x, n_resampled, axis=1)
    x[_decimated_nan_mask(nans, fs, fs_new, n_resampled)] = np.nan

    if not b_multiple_signals:
        x = x[0]

    if return_datarate:
        return x, dr
    return x


def unify_sampling_frequency(x : list, sampling_frequency: list, fs_new=None) -> tuple:
    """
    Bring a list of signals to a common sampling frequency using :func:`decimate`.

    - If all frequencies are equal and ``fs_new`` is ``None``: nothing is done.
    - If frequencies differ and ``fs_new`` is ``None``: all signals are decimated to
      the lowest frequency present.
    - If ``fs_new`` is given: every signal whose frequency differs from ``fs_new`` is
      resampled to ``fs_new`` with :func:`decimate`, which downsamples **or upsamples**
      (e.g. a 150 Hz channel brought to ``fs_new=200``).

    Signals already at the target frequency are passed through unchanged (they are
    **not** low-passed).

    Parameters
    ----------
    x : list of numpy.ndarray
        Signals (1-D, or 2-D with samples along the last axis).
    sampling_frequency : list or numpy.ndarray
        Sampling frequency in Hz of each signal.
    fs_new : float, optional
        Target sampling frequency in Hz. Signals below it are upsampled, signals
        above it are low-passed (``fs_new / 3``) and downsampled.

    Returns
    -------
    tuple
        ``(list of numpy.ndarray, fs_new)``. A **new** list is returned; the input list
        and its arrays are not modified.
    """

    b_process = False

    if not isinstance(x, list):
        raise TypeError('First variable must be list of numpy arrays')

    if not isinstance(sampling_frequency, (list, np.ndarray)):
        raise TypeError('Second parameter must be list or array of floats/integers')
    sampling_frequency = np.array(sampling_frequency)

    if x.__len__() != sampling_frequency.__len__():
        raise AssertionError('Length of a signal list must be same as length of sampling_frequency list')

    x = list(x)  # never modify the caller's list

    fs_in_set = np.unique(sampling_frequency)
    if isinstance(fs_new, type(None)):
        if fs_in_set.__len__() > 1:
            b_process = True
            fs_new = fs_in_set.min()
    else:
        if (sampling_frequency != fs_new).sum() > 0:
            b_process = True
        else:
            fs_new = fs_in_set.min()

    if b_process is True:
        for idx in range(x.__len__()):
            fs = sampling_frequency[idx]
            if fs == fs_new:
                continue
            x[idx] = decimate(x[idx], fs, fs_new)

    return x, fs_new

def _fft_brickwall(X, fs, cutoff, type):
    """Brick-wall FFT filter of a finite array along the last axis (periodic assumption)."""
    n_samples = X.shape[-1]
    Xs = fft.fft(X, axis=-1)
    freq = np.abs(np.fft.fftfreq(n_samples, d=1.0 / fs))
    keep = freq <= cutoff if type == 'lp' else freq > cutoff
    return np.real(fft.ifft(np.where(keep, Xs, 0), axis=-1))


def _fft_filter_extended(X, fs, cutoff, type, pad_periods=5):
    """
    Brick-wall filter of a finite array with the :class:`LowFrequencyFilter` edge
    handling: a least-squares line is removed (and added back to the low-pass part),
    both ends are extended by ``pad_periods / cutoff`` s (local line + mirrored
    residual), the extended signal is filtered, and the extension is cropped.
    ``'hp'`` is ``X`` minus the low-pass part, so ``lp + hp == X``.
    """
    n = X.shape[-1]
    slope, intercept = _line_fit(X)
    trend = slope[..., None] * np.arange(n) + intercept[..., None]
    n_pad = int(np.ceil(pad_periods * fs / cutoff))
    n_fit = int(round(2 * fs / cutoff))
    R = _extend_edges_linear(X - trend, n_pad, n_pad, n_fit)
    low = _fft_brickwall(R, fs, cutoff, 'lp')[..., n_pad:n_pad + n] + trend
    return low if type == 'lp' else X - low


def fft_filter(X:np.ndarray, fs:float, cutoff:float, type:str='lp', edges=None, max_gap_fill=None):
    """
    Ideal (brick-wall) FFT filter. NaN-aware.

    The signal is transformed with an FFT along the last axis, every frequency bin on
    the rejected side of ``cutoff`` is set to zero, and the result is transformed back.
    Bin frequencies are the exact DFT frequencies ``k * fs / n_samples``
    (:func:`numpy.fft.fftfreq`); positive and negative frequencies are treated
    symmetrically, so the output is real.

    - ``'lp'`` keeps bins with ``|f| <= cutoff`` (a bin exactly at ``cutoff`` is kept).
    - ``'hp'`` keeps bins with ``|f| > cutoff`` (DC is always removed by ``'hp'``).

    Parameters
    ----------
    X : numpy.ndarray
        Signal ``(n_samples,)`` or a stack of signals ``(..., n_samples)``; filtered
        along the **last** axis. NaN marks gaps (see Notes). Not modified.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Cutoff frequency in Hz (``>= 0``). A cutoff at or above Nyquist makes ``'lp'``
        the identity and ``'hp'`` return zeros.
    type : str
        ``'lp'`` or ``'hp'``.
    edges : {None, 'periodic', 'extend'}
        What is assumed beyond the ends of the record (and of each segment between
        gaps):

        - ``'periodic'``: nothing is added; the FFT treats the record as one period,
          so a mismatch between its start and end rings (Gibbs) into both edges.
          This is the v2.0.0 behaviour.
        - ``'extend'``: the :class:`LowFrequencyFilter` edge handling - a
          least-squares line is removed and added back to the low-pass part, and each
          end is extended by ``5 / cutoff`` s (a local line plus the mirrored
          residual), so offset, drift and gap edges produce no jump. Falls back to
          ``'periodic'`` for ``cutoff == 0`` (DC removal has no edge transient).
        - ``None`` (default): ``'periodic'`` if ``X`` contains no NaN (unchanged
          from v2.0.0), ``'extend'`` if it does.

        .. warning:: With ``None`` the edge handling depends on whether ``X`` contains
           NaN: a single NaN anywhere switches the **whole** output from ``'periodic'``
           to ``'extend'``, which changes samples far from the gap (e.g. up to ~1 for
           SD-172 1/f data, 1 Hz high-pass, 30-60 s from the NaN). Pass ``edges``
           explicitly (``'extend'`` is the better choice for EEG-like data) whenever
           the result must not depend on the presence of gaps.
    max_gap_fill : float, optional
        Gaps shorter than this many seconds are bridged by linear interpolation;
        longer gaps split the signal into separately filtered segments. Default
        ``0.5 / cutoff`` (``0`` for ``cutoff == 0``). A level jump across a bridged gap
        is filtered like a real step; use a smaller value to split instead.

    Returns
    -------
    numpy.ndarray
        Same shape as ``X`` (``float64``). NaN exactly where ``X`` is NaN.

    Raises
    ------
    ValueError
        If ``X`` contains ``inf``, or for an invalid ``type``, ``fs``, ``cutoff`` or
        ``edges``.

    Notes
    -----
    A brick-wall filter rings (Gibbs phenomenon) around sharp transients. With
    ``edges='periodic'`` no windowing or padding is applied.

    **Gaps (NaN).** As in :class:`LowFrequencyFilter`: interior gaps shorter than
    ``max_gap_fill`` are bridged linearly, filtered through and re-masked; at longer
    gaps the signal is split and each segment is filtered as if it were a complete
    record, with the chosen ``edges`` handling. Leading/trailing NaN are excluded.
    A level jump across a bridged gap is filtered as a real step, and a segment of only
    a few samples between long gaps gives finite but meaningless values (see
    :class:`LowFrequencyFilter`).
    (An FFT of the whole array would spread a single NaN over the entire output.)

    .. note:: **Changed after v2.0.0:**
       NaN input used to return an all-NaN output; gaps are now handled as above.
       ``edges`` and ``max_gap_fill`` are new; the default result for NaN-free input
       is that of v2.0.0 apart from the frequency-axis fix (see the Changes page).
    """
    if type not in ('lp', 'hp'):
        raise ValueError(f"type must be 'lp' or 'hp', got {type!r}")
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs!r}")
    if cutoff < 0:
        raise ValueError(f"cutoff must be >= 0, got {cutoff!r}")
    if edges not in (None, 'periodic', 'extend'):
        raise ValueError(f"edges must be None, 'periodic' or 'extend', got {edges!r}")
    X = np.asarray(X, dtype=float)
    if np.isinf(X).any():
        raise ValueError('fft_filter: X contains inf (NaN marks gaps and is handled; inf is not)')
    has_nan = bool(np.isnan(X).any())
    if edges is None:
        edges = 'extend' if has_nan else 'periodic'
    if edges == 'extend' and cutoff == 0:
        edges = 'periodic'
    if max_gap_fill is None:
        max_gap_fill = 0.5 / cutoff if cutoff > 0 else 0.0
    if not max_gap_fill >= 0:
        raise ValueError(f'max_gap_fill must be >= 0 (seconds), got {max_gap_fill!r}')
    if X.shape[-1] == 0:
        return X.copy()

    if edges == 'extend':
        func = lambda Z: _fft_filter_extended(Z, fs, cutoff, type)
    else:
        func = lambda Z: _fft_brickwall(Z, fs, cutoff, type)
    if has_nan:
        return _per_valid_segment(X, func, int(np.ceil(max_gap_fill * fs)))
    return func(X)

def buffer(x:np.ndarray, fs:float=1, segm_size:float=None, overlap:float = 0, drop:bool=True):
    """
    Cut a 1-D signal into (optionally overlapping) fixed-length segments.

    Segment ``k`` starts at sample ``round(k * (segm_size - overlap) * fs)`` - start
    positions are computed from exact times, so there is no cumulative drift for
    non-integer ``fs * (segm_size - overlap)``. Every segment has
    ``round(segm_size * fs)`` samples.

    Parameters
    ----------
    x : numpy.ndarray
        1-D signal ``(n_samples,)``. For multichannel data call per channel.
    fs : float
        Sampling frequency in Hz.
    segm_size : float, optional
        Segment length in seconds. If ``None``, ``x`` is returned unchanged.
    overlap : float
        Overlap between consecutive segments in seconds, ``0 <= overlap < segm_size``.
    drop : bool
        If ``True`` (default) a trailing incomplete segment is dropped; otherwise one
        final segment covering the remaining samples is zero-padded (``False`` for
        boolean input) to full length.

    Returns
    -------
    numpy.ndarray
        ``(n_segments, round(segm_size * fs))`` with the dtype of ``x``. NaNs are
        copied as-is. If no segment fits, the result has shape ``(0, n_segm)``.

    Raises
    ------
    ValueError
        If ``x`` is not 1-D, or ``overlap`` is negative or ``>= segm_size``
        (which previously caused an infinite loop), or the hop
        ``segm_size - overlap`` is shorter than one sample.
    """
    x = np.asarray(x)
    if isinstance(segm_size, type(None)):
        return x
    if x.ndim != 1:
        raise ValueError(f'buffer expects a 1-D signal, got shape {x.shape}; buffer each channel separately')
    if segm_size <= 0:
        raise ValueError(f'segm_size must be > 0, got {segm_size!r}')
    if overlap < 0 or overlap >= segm_size:
        raise ValueError(f'overlap must satisfy 0 <= overlap < segm_size, got overlap={overlap!r}, segm_size={segm_size!r}')

    n = x.shape[0]
    n_segm = int(round(fs * segm_size))
    if n_segm < 1:
        raise ValueError(f'segm_size * fs must be >= 1 sample, got {segm_size * fs!r}')
    step_s = segm_size - overlap
    if step_s * fs < 1:
        raise ValueError('segm_size - overlap must correspond to at least one sample')

    # candidate starts from exact times (no cumulative rounding drift)
    k_max = int(np.floor(max(n - n_segm, 0) / (step_s * fs))) + 3
    st = np.round(np.arange(k_max) * step_s * fs).astype(np.int64)
    full = st + n_segm <= n
    starts = st[full]
    out = x[starts[:, None] + np.arange(n_segm)[None, :]] if starts.size else np.zeros((0, n_segm), dtype=x.dtype)

    if not drop:
        last_end = int(starts[-1]) + n_segm if starts.size else 0
        partial = st[~full]
        if last_end < n and partial.size and partial[0] < n:
            seg = np.zeros((1, n_segm), dtype=x.dtype)
            tail = x[partial[0]:]
            seg[0, :tail.shape[0]] = tail
            out = np.concatenate((out, seg), axis=0)
    return out

def PSD(x:np.ndarray, fs:float, nperseg=None, noverlap=0, nfft=None):
    """
    Estimates PSD of an input signal or signals using Welch's method.
    If nperseg is None, the spectrum is estimated from the whole signal in a single window.

    Parameters
    ----------
    x : np.ndarray
        A single signal with a shape (n_samples) or set of signals with a shape (n_signals, n_shapes)
    fs : float
        Sampling frequency
    nperseg : int
        Number of samples for a segment
    noverlap : int
        Number of overlap samples.

    Returns
    -------
    freq : np.ndarray
        Frequency axis for estimated PSD
    psd : np.ndarray
        Power spectral density estimate

    """
    axis = x.ndim-1
    if isinstance(nperseg, type(None)):
        nperseg = x.shape[axis]

    freq, psd = signal.welch(
        x,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend='constant',
        return_onesided=True,
        scaling='density',
        axis=axis,
        average='mean'
    )

    return freq, psd



    #N = xbuffered.shape[1]
    #psdx = fft.fft(xbuffered, axis=1)
    #psdx = psdx[:, 1:int(np.round(N / 2)) + 1]

    #psdx = (1 / (fs * N)) * np.abs(psdx) ** 2
    #psdx[np.isinf(psdx)] = np.nan
    return #psdx

def _line_fit(x):
    """
    Least-squares line through each row of ``x`` (last axis), sample index as abscissa.

    Returns ``(slope, intercept)`` with shape ``x.shape[:-1]``; the line is
    ``slope * k + intercept`` for ``k = 0 .. n-1``. A single sample gives slope 0.
    """
    n = x.shape[-1]
    if n < 2:
        return np.zeros(x.shape[:-1]), x[..., 0].astype(float)
    k = np.arange(n, dtype=float)
    k_c = k - k.mean()
    y_m = x.mean(axis=-1)
    slope = (x * k_c).sum(axis=-1) / (k_c ** 2).sum()
    return slope, y_m - slope * k.mean()


def _extend_edges_linear(x, n_front, n_back, n_fit):
    """
    Extend ``x`` (last axis) by ``n_front`` / ``n_back`` samples for edge-transient-free
    filtering.

    At each edge a least-squares line is fitted to the ``n_fit`` outermost samples.
    The extension is that line extrapolated outwards **plus** the even (mirror)
    reflection of the residual about the line. The extension is therefore continuous
    in value with the data and continues its local slope (offset and drift do not
    produce a step or a kink), while the faster content is mirrored.
    """
    n = x.shape[-1]
    n_fit = int(max(1, min(n, n_fit)))
    pad_cfg = [(0, 0)] * (x.ndim - 1)
    k = np.arange(n_fit, dtype=float)

    seg = x[..., :n_fit]
    a, b = _line_fit(seg)
    res = seg - (a[..., None] * k + b[..., None])
    head_res = np.pad(res, pad_cfg + [(n_front, 0)], mode='reflect' if n_fit > 1 else 'edge')[..., :n_front]
    kh = np.arange(-n_front, 0, dtype=float)
    head = a[..., None] * kh + b[..., None] + head_res

    seg = x[..., n - n_fit:]
    a, b = _line_fit(seg)
    res = seg - (a[..., None] * k + b[..., None])
    tail_res = np.pad(res, pad_cfg + [(0, n_back)], mode='reflect' if n_fit > 1 else 'edge')[..., n_fit:]
    kt = np.arange(n_fit, n_fit + n_back, dtype=float)
    tail = a[..., None] * kt + b[..., None] + tail_res

    return np.concatenate((head, x, tail), axis=-1)


def _zero_phase(flt, x):
    """Forward-backward filtering along the last axis; ``flt`` is ``('sos', sos)`` or ``('ba', b, a)``."""
    n = x.shape[-1]
    if flt[0] == 'sos':
        sos = flt[1]
        padlen = min(3 * (2 * sos.shape[0] + 1), n - 1)
        return signal.sosfiltfilt(sos, x, axis=-1, padlen=padlen)
    b, a = flt[1], flt[2]
    padlen = min(3 * max(len(np.atleast_1d(a)), len(b)), n - 1)
    return signal.filtfilt(b, a, x, axis=-1, padlen=padlen)


def _valid_runs(nan_row):
    """``(start, stop)`` index pairs of the runs of non-NaN samples in a 1-D NaN mask."""
    d = np.diff(np.concatenate(([0], (~nan_row).astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _per_valid_segment(X, func, max_fill=0):
    """
    Gap-aware application of ``func`` (finite N-D array -> array of the same shape,
    time on the last axis) to ``X`` that contains NaN gaps.

    1. Interior NaN runs shorter than ``max_fill`` samples (dropouts) are bridged by
       linear interpolation between their neighbouring valid samples.
    2. Every remaining run of valid (or bridged) samples is passed to ``func`` on its
       own, so it is treated exactly like a complete record (``func``'s own edge
       handling applies at the gap edges).
    3. The output is NaN exactly where ``X`` was NaN.

    Rows that share the same NaN mask (the usual case for multichannel recordings
    with common gaps) are processed together.
    """
    n = X.shape[-1]
    X2 = X.reshape(-1, n)
    nans = np.isnan(X2)
    out = np.full(X2.shape, np.nan)
    if (nans == nans[:1]).all():
        groups = [np.arange(X2.shape[0])]
    else:
        groups = [np.array([r]) for r in range(X2.shape[0])]
    idx = np.arange(n)
    for rows in groups:
        m = nans[rows[0]].copy()
        Xr = X2[rows].copy()
        if max_fill > 0:
            for a, b in _valid_runs(~m):  # NaN runs
                if a > 0 and b < n and b - a < max_fill:
                    for i in range(Xr.shape[0]):
                        Xr[i, a:b] = np.interp(idx[a:b], [a - 1, b], [Xr[i, a - 1], Xr[i, b]])
                    m[a:b] = False
        for a, b in _valid_runs(m):
            out[rows, a:b] = func(Xr[:, a:b])
    out[nans] = np.nan
    return out.reshape(X.shape)


class LowFrequencyFilter:
    """
    Zero-phase low-pass or high-pass filter for very low cutoff frequencies relative to
    the sampling rate (e.g. a 0.5 Hz high-pass on 8 kHz data), implemented as a
    decimate -> filter -> upsample cascade.

    A cutoff of ``1e-4 * fs`` cannot be realised well by a direct filter (an FIR would
    need ~1e5 taps, an IIR is ill-conditioned). Instead the signal is

    1. halved in rate ``n_decimate`` times: each step applies a zero-phase
       anti-aliasing low-pass (normalised cutoff ``dec_cutoff`` x Nyquist of that
       stage) and keeps every 2nd sample;
    2. low-passed at ``cutoff`` at the low rate ``fs / 2**n_decimate`` (zero-phase);
    3. brought back to ``fs`` by ``n_decimate`` steps of zero-insertion followed by
       the same anti-aliasing low-pass (gain 2).

    The result of 1-3 is the low-frequency content of the signal. ``filter_type='lp'``
    returns it; ``filter_type='hp'`` returns ``x`` minus it.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signals that will be filtered, in Hz.
    cutoff : float
        Cutoff frequency in Hz. Must lie below the Nyquist frequency of the low rate,
        ``cutoff < fs / 2**n_decimate / 2``, and should lie well inside the band kept
        by the anti-aliasing stages (``< dec_cutoff * fs / 2**(n_decimate + 1)``).
    n_decimate : int
        Number of halving steps (``>= 0``). The cutoff filter runs at
        ``fs / 2**n_decimate``. Choose it so that this rate is ~20-500 x ``cutoff``,
        e.g. ``n_decimate=5`` (250 Hz) for 0.5 Hz on 8 kHz data.
    n_order : int, optional
        Filter order: number of taps for ``ftype='fir'`` (default 101), Butterworth
        order for ``ftype='iir'`` (default 3). Used for both the anti-aliasing and the
        cutoff filter. An FIR needs about ``n_order >= 1.5 * fs_low / cutoff`` taps to
        resolve the cutoff (``fs_low = fs / 2**n_decimate``), otherwise its transition
        band is much wider than ``cutoff`` and the effective cutoff is higher: the
        default 101 taps for 0.5 Hz at ``fs_low = 250`` Hz pass 0.94 at 0.5 Hz and reach
        -6 dB only at ~1.7 Hz. A ``UserWarning`` is issued when the FIR's low-frequency
        gain at ``cutoff`` exceeds 0.3 (nominal 0.25).
    dec_cutoff : float
        Normalised (to the Nyquist of each stage, ``0 < dec_cutoff < 1``) cutoff of the
        anti-aliasing low-pass used by the halving/upsampling steps. Default 0.3.
    filter_type : {'lp', 'hp'}
        Which side of ``cutoff`` is returned:

        - ``'lp'`` returns the low-frequency content **below** ``cutoff``.
        - ``'hp'`` returns ``x`` minus that low-frequency content, i.e. the
          content **above** ``cutoff`` (use this to remove slow drift / DC).
    max_gap_fill : float, optional
        NaN gaps (interior runs of NaN) shorter than this many **seconds** are bridged
        by linear interpolation before filtering; longer gaps split the signal into
        segments that are filtered separately. Default ``0.5 / cutoff`` (1 s for
        0.5 Hz). ``0`` always splits. See *Gaps* in the notes.
    ftype : {'fir', 'iir'}
        ``'fir'``: :func:`scipy.signal.firwin` windowed-sinc filters (``cutoff`` is the
        design -6 dB point). ``'iir'``: Butterworth filters in second-order sections
        (``cutoff`` is the design -3 dB point). All filters are applied forward and
        backward, so the phase is zero and the magnitude response is squared: at
        ``cutoff`` the low-frequency gain is ~0.5 (IIR) or ~0.25 (FIR, **only if** it
        has enough taps, see ``n_order``).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from brainmaze_utils.signal import LowFrequencyFilter

        fs = 8000
        x = np.random.randn(60 * fs) + 1500.0 + np.linspace(0, 300, 60 * fs)  # DC + drift

        # remove DC and slow drift below 0.5 Hz (high-pass), keep everything above
        hp = LowFrequencyFilter(fs=fs, cutoff=0.5, n_decimate=5, ftype='iir', n_order=3, filter_type='hp')
        x_hp = hp(x)

        # keep only the slow content below 0.5 Hz
        lp = LowFrequencyFilter(fs=fs, cutoff=0.5, n_decimate=5, ftype='iir', n_order=3, filter_type='lp')
        x_slow = lp(x)

    Notes
    -----
    **Input.** ``x`` may be 1-D ``(n_samples,)`` or N-D with time along the **last**
    axis, e.g. ``(n_channels, n_samples)``. NaN marks gaps (see below); ``inf`` raises
    ``ValueError``. The input is not modified.

    **Gaps (NaN).** The output is NaN exactly where the input is NaN, and finite
    everywhere else, so the filter can follow a NaN-aware :func:`decimate` in a
    downsampling cascade.

    - Gaps shorter than ``max_gap_fill`` (default half a cutoff period, e.g. dropped
      samples) are bridged by linear interpolation, filtered through, and re-masked.
      The bridge does not check the two sides for a level jump: a jump across a
      bridged gap (amplifier re-zero, reconnection after a dropout) is filtered as a
      genuine step. E.g. a 500 uV DC step across a 0.1 s gap at 0.5 Hz high-pass gives
      a transient above 10 uV for about +-1.5 s around the gap (the same as the response
      to a real step). If such jumps are expected, lower ``max_gap_fill`` below the gap
      length (``0`` always splits) so that each side is filtered on its own.
    - Longer gaps split the signal: each valid segment is filtered on its own, with
      the same edge handling as the record edges (below), so there is no jump at a
      gap edge, and nothing on one side of a long gap affects the other side.
      A segment shorter than about ``1 / cutoff`` s (e.g. a few samples between two
      long gaps) carries no information at the cutoff frequency: its output is finite
      but is not a meaningful low-/high-pass estimate (a 3-sample segment can be off by
      about 10 for SD-20 data). Treat such segments as unusable.

    Measured on 1/f noise (SD 20) with offset, drift and a 10 Hz tone, 0.5 Hz IIR
    high-pass, maximum error within 10 s of a gap compared with the same signal
    without the gap: 1 ms gap 0.008, 10 ms 0.14, 100 ms 0.95 (bridged); 1 s 5.4,
    10 s 7.5 (split; bridging would give 6.0 and 8.1). The error next to a split
    gap is the record-edge error described below. Leading and trailing NaN are
    simply excluded. A multichannel array whose channels share the same gaps is
    processed in one pass. Every split segment is extended by ``3 / cutoff`` s at both
    ends, so many long gaps cost time: 10 min at 8 kHz (0.5 Hz IIR) takes 1.3 s
    without gaps, 1.5 s with 500 dropouts, 2.8 s with 50 two-second gaps and 11 s
    with 300 split segments.

    **Record edges.** A filter with a 0.5 Hz cutoff has a transient lasting
    seconds, so what is assumed about the signal beyond the two ends of the record
    determines the first and last few seconds of the output. This class

    1. removes a least-squares straight line (offset + drift) from each signal and
       adds it back to the low-frequency output (a zero-phase low-pass passes a
       straight line unchanged, so for ``'hp'`` offset and linear drift are removed
       exactly, edges included);
    2. extends each end by ``max(3 / cutoff s, the legacy pad)`` samples: a line fitted
       to the outermost ``2 / cutoff`` s is extrapolated and the residual is mirrored
       about it, so the extension is continuous in value and slope with the data;
    3. runs the cascade on the extended signal and crops the extension.

    For a signal with a 2000 (uV) offset, a 40 uV/s drift and three 10 uV tones
    (3, 10, 40 Hz) at 8 kHz, a 0.5 Hz ``'hp'`` (``n_decimate=5``, IIR order 3) gives
    a maximum edge error of ~0.75 uV over the first/last second (previously ~1900 uV);
    offset and drift no longer contribute at all (the output equals that for the
    zero-mean signal). The remaining edge error is the irreducible effect of not
    knowing the signal beyond the record, of the order of ``A * cutoff / (pi * f)``
    for a component of amplitude ``A`` at frequency ``f``; on 1/f-like EEG it scales
    with the signal power near and below ``cutoff``. Far from the edges (more than
    ~3 / cutoff s), the output equals that of the cascade applied to an infinitely
    long signal.

    Design choice: extensions were compared on synthetic data (scripts in the PR). On
    1/f-like signals the mirrored extension gives the smallest edge error (zero
    extension of the detrended signal is ~18 % worse); for a signal whose content lies
    entirely well above ``cutoff`` (pure tones), zero extension would be ~2x better
    (0.36 vs 0.77 uV in the example above) but only without any offset or drift.

    .. note:: **Changed after v2.0.0:**
       The signal used to be zero-padded by only ``2 * n_order * 2**n_decimate``
       samples (e.g. 24 ms for a 0.5 Hz filter at 8 kHz). Any offset therefore became a
       step at both edges (output off by about half the offset, e.g. ~1000 uV for a
       2000 uV offset, over the first/last seconds), and the filter transient reached far
       into the record. The frequency response in the interior is unchanged. IIR filters
       are now applied in second-order sections (numerically identical response; the
       ``b_*``/``a_*`` attributes are kept for reference). N-D input is supported.
       NaN gaps are handled (see *Gaps*); v2.0.0 returned an **all-NaN** output for
       any input containing a NaN. ``inf`` raises ``ValueError``. Short FIRs that
       cannot resolve the cutoff now emit a ``UserWarning``.
    """

    __version__ = '0.1.0'

    def __init__(self, fs=None, cutoff=None, n_decimate=1, n_order=None, dec_cutoff=0.3, filter_type='lp', ftype='fir',
                 max_gap_fill=None):
        if fs is None or cutoff is None:
            raise ValueError('LowFrequencyFilter: fs and cutoff are required')
        if not fs > 0:
            raise ValueError(f'fs must be > 0, got {fs!r}')
        if not cutoff > 0:
            raise ValueError(f'cutoff must be > 0, got {cutoff!r}')
        if int(n_decimate) != n_decimate or n_decimate < 0:
            raise ValueError(f'n_decimate must be an integer >= 0, got {n_decimate!r}')
        if not 0 < dec_cutoff < 1:
            raise ValueError(f'dec_cutoff must be in (0, 1), got {dec_cutoff!r}')
        if filter_type not in ('lp', 'hp'):
            raise ValueError(f"filter_type must be 'lp' or 'hp', got {filter_type!r}")
        if ftype not in ('fir', 'iir'):
            raise ValueError(f"ftype must be 'fir' or 'iir', got {ftype!r}")
        if max_gap_fill is None:
            max_gap_fill = 0.5 / cutoff
        if not max_gap_fill >= 0:
            raise ValueError(f'max_gap_fill must be >= 0 (seconds), got {max_gap_fill!r}')
        self.max_gap_fill = float(max_gap_fill)

        self.fs = fs
        self.cutoff = cutoff
        self.n_decimate = int(n_decimate)
        self.dec_cutoff = dec_cutoff
        self.filter_type = filter_type

        self.n_order = n_order
        self.ftype = ftype

        self.n_append = None
        self.n_pad = None

        self.design_filters()

    def design_filters(self):
        """
        Design the anti-aliasing (``*_dec``) and cutoff (``*_filt``) filters and the
        edge-extension length ``n_pad`` (samples at ``fs``).
        """
        fs_low = self.fs / 2 ** self.n_decimate
        wn = 2 * self.cutoff / fs_low
        if not 0 < wn < 1:
            raise ValueError(
                f'cutoff={self.cutoff!r} Hz must be below the Nyquist frequency of the decimated rate '
                f'fs / 2**n_decimate / 2 = {fs_low / 2!r} Hz; reduce n_decimate')

        if self.ftype == 'fir':
            if isinstance(self.n_order, type(None)): self.n_order = 101
            self.a_dec = [1]
            self.b_dec = signal.firwin(self.n_order, self.dec_cutoff, pass_zero=True)
            self.b_dec /= self.b_dec.sum()

            self.a_filt = [1]
            self.b_filt = signal.firwin(self.n_order, wn, pass_zero=True)
            self.b_filt /= self.b_filt.sum()
            self._dec = ('ba', self.b_dec, self.a_dec)
            self._filt = ('ba', self.b_filt, self.a_filt)

            # forward-backward gain at the cutoff is ~0.25 for an FIR long enough to resolve
            # it; a short FIR has a transition band much wider than the cutoff
            _, h = signal.freqz(self.b_filt, worN=[self.cutoff], fs=fs_low)
            n_needed = int(np.ceil(1.5 * fs_low / self.cutoff)) | 1
            if abs(h[0]) ** 2 > 0.3:
                warnings.warn(
                    f'LowFrequencyFilter: a {self.n_order}-tap FIR cannot resolve cutoff={self.cutoff!r} Hz at the '
                    f'decimated rate {fs_low!r} Hz; its low-frequency gain at the cutoff is {abs(h[0]) ** 2:.2f} '
                    f'instead of ~0.25, i.e. the effective cutoff is much higher. Use n_order >= {n_needed}, '
                    f"ftype='iir', or a larger n_decimate.", UserWarning, stacklevel=3)

        elif self.ftype == 'iir':
            if isinstance(self.n_order, type(None)): self.n_order = 3
            self.b_dec, self.a_dec = signal.butter(self.n_order, self.dec_cutoff, btype='low')
            self.b_filt, self.a_filt = signal.butter(self.n_order, wn, btype='low')
            self.sos_dec = signal.butter(self.n_order, self.dec_cutoff, btype='low', output='sos')
            self.sos_filt = signal.butter(self.n_order, wn, btype='low', output='sos')
            self._dec = ('sos', self.sos_dec)
            self._filt = ('sos', self.sos_filt)

        # legacy pad (kept as a lower bound) and the edge extension actually used:
        # >= 3 cutoff periods, enough for the cutoff filter's transient to die out.
        self.n_append = (2 * self.n_order) * (2 ** self.n_decimate)
        self.n_pad = int(max(self.n_append, np.ceil(3 * self.fs / self.cutoff)))

    def decimate(self, X):
        """
        Anti-aliasing low-pass (zero-phase) and downsampling by 2 along the last axis.
        """
        return _zero_phase(self._dec, X)[..., ::2]

    def upsample(self, X):
        """
        Upsample by 2 along the last axis: zero-insertion, then the anti-aliasing
        low-pass (zero-phase) with gain 2.
        """
        X_up = np.zeros(X.shape[:-1] + (X.shape[-1] * 2,))
        X_up[..., ::2] = X
        return _zero_phase(self._dec, X_up) * 2

    def filter_signal(self, X):
        """
        Low-frequency content of ``X`` (below ``cutoff``), same shape as ``X``.

        See the class notes for the edge and NaN handling: every run of valid samples
        between NaN gaps is filtered on its own, and NaN samples stay NaN. Raises
        ``ValueError`` if ``X`` contains inf.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 0:
            raise ValueError('LowFrequencyFilter expects at least a 1-D signal')
        if np.isinf(X).any():
            raise ValueError('LowFrequencyFilter: X contains inf (NaN marks gaps and is handled; inf is not)')
        if X.shape[-1] == 0:
            return X.copy()
        if np.isnan(X).any():
            return _per_valid_segment(X, self._filter_finite, int(np.ceil(self.max_gap_fill * self.fs)))
        return self._filter_finite(X)

    def _filter_finite(self, X):
        """Low-frequency content of a NaN-free ``X`` (record-edge handling, see class notes)."""
        n = X.shape[-1]

        slope, intercept = _line_fit(X)
        trend = slope[..., None] * np.arange(n) + intercept[..., None]
        R = X - trend

        q = 2 ** self.n_decimate
        C = (-(n + 2 * self.n_pad)) % q
        pad_front = self.n_pad + C
        pad_back = self.n_pad
        n_fit = int(round(2 * self.fs / self.cutoff))  # two cutoff periods
        R = _extend_edges_linear(R, pad_front, pad_back, n_fit)

        for _ in range(self.n_decimate):
            R = self.decimate(R)
        R = _zero_phase(self._filt, R)
        for _ in range(self.n_decimate):
            R = self.upsample(R)

        return R[..., pad_front: pad_front + n] + trend

    def __call__(self, X):
        """
        Apply the filter along the last axis.

        Returns the low-frequency content (``filter_type='lp'``) or ``X`` minus the
        low-frequency content (``filter_type='hp'``). ``X`` is not modified.
        """
        X = np.asarray(X, dtype=float)
        low = self.filter_signal(X)
        if self.filter_type == 'lp':
            return low
        return X - low


def resample(x, fsamp_orig, fsamp_new):
    """
    Resample a signal to a new sampling frequency by linear interpolation. NaN-aware.

    .. warning::
       **No anti-aliasing filter is applied - by design.** ``resample`` only
       interpolates. When downsampling, any content above the new Nyquist frequency
       ``fsamp_new / 2`` folds back (aliases) into the output: e.g. a 400 Hz tone
       resampled 1000 -> 250 Hz appears as a full-amplitude 100 Hz tone. Band-limiting
       the signal is deliberately left to the caller, who knows which band matters.
       Before downsampling, low-pass below ``fsamp_new / 2`` (with margin), or use
       :func:`decimate`, which filters and downsamples in one step.

       .. code-block:: python

           import scipy.signal as ss
           from brainmaze_utils.signal import resample, decimate

           fs, fs_new = 1000, 250
           # option 1: your own zero-phase low-pass, then resample
           sos = ss.butter(8, 0.8 * fs_new / 2, 'lp', fs=fs, output='sos')
           y = resample(ss.sosfiltfilt(sos, x), fs, fs_new)  # x must be NaN-free here
           # option 2: decimate (anti-aliasing filter included, NaN-aware)
           y = decimate(x, fs, fs_new)

       Upsampling (``fsamp_new > fsamp_orig``) does not need a filter; linear
       interpolation slightly attenuates content close to the original Nyquist.

    Sample ``i`` of the input is at time ``i / fsamp_orig`` and sample ``k`` of the
    output at ``k / fsamp_new`` (both starting at 0); the output has
    ``round(n_samples * fsamp_new / fsamp_orig)`` samples. Each output value is the
    linear interpolation between the two input samples that bracket its time.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_samples,)`` or ``(..., n_samples)`` - resampled along the last axis.
        Not modified.
    fsamp_orig : float
        Sampling frequency of ``x`` in Hz.
    fsamp_new : float
        Target sampling frequency in Hz.

    Returns
    -------
    numpy.ndarray
        Resampled signal (``float64``). If ``fsamp_orig == fsamp_new`` a float copy
        of ``x`` is returned.

    Notes
    -----
    - NaN handling: an output sample is NaN if either bracketing input sample is NaN
      (an output time that coincides with an input sample uses only that sample).
      Values are never interpolated across a gap. **When downsampling, short gaps can
      disappear:** a gap that falls entirely between two output instants marks no
      output sample (e.g. a 3-sample gap, 1000 -> 250 Hz, gives no NaN at all). Use
      :func:`decimate` if the gap mask must be preserved (it flags every output
      sample within half an output period of a NaN).
    - Empty input (or an output of 0 samples) returns an empty array of shape
      ``x.shape[:-1] + (n_new,)``.
    - Output times beyond the last input sample (possible when upsampling) take the
      last input value.

    .. note:: **Changed after v2.0.0:**
       Time axes were ``linspace(0, 1, N)`` / ``linspace(0, 1, N_new)``, which pins
       both endpoints and stretched time: the effective output rate was
       ``(N_new - 1) / (N - 1) * fs`` rather than ``fsamp_new``. Also fixed
       ``np.NaN`` (removed in NumPy 2) and division by zero for constant signals.
    """
    x = np.array(x, dtype=float)
    if fsamp_orig <= 0 or fsamp_new <= 0:
        raise ValueError('sampling frequencies must be > 0')
    if fsamp_orig == fsamp_new:
        return x

    n = x.shape[-1]
    n_new = int(np.round(n * fsamp_new / fsamp_orig))
    lead_shape = x.shape[:-1]
    if n == 0 or n_new == 0:  # before reshape(-1, n): -1 cannot be inferred when n == 0
        return np.full(lead_shape + (n_new,), np.nan)
    x2 = x.reshape(-1, n)
    out = np.full((x2.shape[0], n_new), np.nan)

    pos = np.arange(n_new) * (fsamp_orig / fsamp_new)  # output times in input-sample units
    pos = np.minimum(pos, n - 1)
    pos_r = np.round(pos)
    pos = np.where(np.abs(pos - pos_r) < 1e-9, pos_r, pos)  # snap float noise onto exact samples
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = pos - i0
    exact = w == 0
    i1 = np.where(exact, i0, i1)

    for r in range(x2.shape[0]):
        row = x2[r]
        a = row[i0]
        b = row[i1]
        out[r] = a + (b - a) * w  # NaN in either neighbour -> NaN
        if exact.any():
            out[r, exact] = a[exact]
    return out.reshape(lead_shape + (n_new,))

def detrend(y, x=None, y2=None, method='lstsq'):
    """
    Remove a linear trend from a 1-D signal.

    Parameters
    ----------
    y : numpy.ndarray
        1-D signal to detrend.
    x : numpy.ndarray, optional
        Abscissa of ``y`` (same length). Default ``numpy.linspace(0, 1, len(y))``.
    y2 : numpy.ndarray, optional
        Second signal (same length) from which the **trend fitted on** ``y`` is
        subtracted as well.
    method : {'lstsq', 'endpoints'}
        - ``'lstsq'`` (default): ordinary least-squares line fitted to all finite
          samples of ``y``.
        - ``'endpoints'``: the line through the first and last sample (the behaviour
          of this function before the fix); sensitive to noise on the two endpoints.

    Returns
    -------
    numpy.ndarray or tuple of numpy.ndarray
        ``y - trend``, or ``(y - trend, y2 - trend)`` if ``y2`` is given. NaNs in the
        inputs stay NaN; they are ignored when fitting.

    Notes
    -----
    .. note:: **Changed after v2.0.0:**
       Default changed from the endpoint line to a least-squares fit. With the
       endpoint line a single noisy first/last sample tilted the whole trend (e.g.
       a +10 outlier at ``y[0]`` left a residual slope of ~9.7 on a slope-5 ramp).
       Use ``method='endpoints'`` to reproduce the old results.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(x, type(None)):
        x = np.linspace(0, 1, y.shape[0])
    x = np.asarray(x, dtype=float)

    if method == 'lstsq':
        ok = np.isfinite(y) & np.isfinite(x)
        if ok.sum() < 2:
            raise ValueError('detrend needs at least two finite samples')
        a, b = np.polyfit(x[ok], y[ok], 1)
    elif method == 'endpoints':
        a = (y[0] - y[-1]) / (x[0] - x[-1])
        b = y[0] - (x[0] * a)
    else:
        raise ValueError(f"method must be 'lstsq' or 'endpoints', got {method!r}")

    trend = x * a + b
    if isinstance(y2, type(None)):
        return y - trend
    return y - trend, np.asarray(y2, dtype=float) - trend

def find_peaks(y):
    """
    Finds the peaks in a given signal.

    Parameters
    ----------
    y : numpy.ndarray
        The input signal in which to find peaks.

    Returns
    -------
    position : numpy.ndarray
        The positions of the peaks in the input signal.
    value : numpy.ndarray
        The values of the peaks in the input signal.
    """
    position = []
    value = []
    for k in range(1, y.__len__() -1):
        if y[k-1] < y[k] and y[k+1] < y[k]:
            position += [k]
            value += [y[k]]
    return np.array(position), np.array(value)


def downsample_min_max(signal: np.ndarray, original_fs: float, final_fs: float) -> tuple[np.ndarray, float]:
    """
    Downsamples an iEEG signal using the min-max method, preserving the temporal
    order of min and max values within each downsampling window.

    The method processes the input signal in non-overlapping windows. For each
    window, it finds the minimum and maximum values and their original temporal
    order. These two values (min and max) are then placed in the output signal
    in their temporal order of appearance within the window.

    Args:
        signal (np.ndarray): The input iEEG signal. Can be 1D (samples,)
                             or 2D (channels, samples).
        original_fs (float): The original sampling rate of the signal in Hz.
        final_fs (float): The desired final sampling rate of the *output points* in Hz.
                          Since each original window produces two points (a min and a max),
                          the number of original signal windows processed per second is
                          `final_fs / 2`.

    Returns:
        tuple[np.ndarray, float]:
            - np.ndarray: The downsampled iEEG signal. If the input was 1D,
                          the output is 1D. If 2D, output is 2D.
            - float: The actual final sampling rate of the output signal in Hz.
                     This will be close to the requested `final_fs` but may differ
                     slightly due to integer window sizes.

    Raises:
        TypeError: If the input signal is not a NumPy array.
        ValueError: If signal dimensions are incorrect, sampling rates are not positive,
                    or if `final_fs` implies a `window_size` less than 1.

    Note:
        The signal is processed in full windows. Any remaining samples at the end
        of the signal that do not form a complete window are ignored.

        NaN samples are ignored when searching for the min/max of a window; a window
        that is entirely NaN produces two NaN output points. (Previously a single NaN
        made both points of its window NaN.)
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Input signal must be a NumPy array.")
    if signal.ndim not in [1, 2]:
        raise ValueError("Input signal must be 1D (samples,) or 2D (channels, samples).")
    if original_fs <= 0:
        raise ValueError("Original sampling rate must be positive.")
    if final_fs <= 0:
        raise ValueError("Final sampling rate must be positive.")

    # Each window in the original signal will produce two points (min and max).
    # So, the number of original signal segments (windows) processed per second is `final_fs / 2`.
    effective_segment_fs = final_fs / 2.0

    if effective_segment_fs <= 0:  # Should be caught by final_fs <= 0 already
        raise ValueError("Calculated effective_segment_fs is not positive. Check final_fs.")

    # window_size is the number of original samples per min-max pair output
    window_size = int(round(original_fs / effective_segment_fs))

    if window_size < 1:
        raise ValueError(
            f"Calculated window_size is {window_size}, which is less than 1. "
            f"This typically means final_fs ({final_fs} Hz) is too high compared to original_fs ({original_fs} Hz) "
            "for meaningful min-max downsampling that produces 2 points per window."
        )
    if window_size == 1:
        print(
            f"Warning: window_size is 1. Each original sample will effectively be duplicated "
            f"in the output (as min and max of a single point are the point itself). "
            f"The output sampling rate will be 2 * original_fs. "
            f"Requested final_fs ({final_fs} Hz) might lead to this behavior."
        )

    input_signal_is_1d = False
    if signal.ndim == 1:
        input_signal_is_1d = True
        # Convert 1D signal to 2D for consistent processing
        signal_2d = signal.reshape(1, -1)
    else:
        signal_2d = signal

    n_channels, n_samples = signal_2d.shape

    if n_samples == 0:
        actual_output_fs = 0.0
        if input_signal_is_1d:
            return np.array([], dtype=signal.dtype), actual_output_fs
        else:
            return np.array([[] for _ in range(n_channels)], dtype=signal.dtype), actual_output_fs

    # Calculate the number of full windows that can be formed
    num_windows = n_samples // window_size

    if num_windows == 0:
        # Not enough data for even one full window
        actual_output_fs = 0.0  # No output points generated from full windows
        if input_signal_is_1d:
            return np.array([], dtype=signal.dtype), actual_output_fs
        else:
            # Return an empty array with the correct number of channels
            return np.empty((n_channels, 0), dtype=signal.dtype), actual_output_fs

    # Trim signal to the length that is a multiple of window_size
    trimmed_length = num_windows * window_size
    trimmed_signal = signal_2d[:, :trimmed_length]

    # Reshape to (n_channels, num_windows, window_size) to process windows
    reshaped_signal = trimmed_signal.reshape(n_channels, num_windows, window_size)

    # Find min and max values within each window, ignoring NaN. A window that is
    # entirely NaN yields NaN for both points.
    # np.argmin / np.argmax return the index of the *first* occurrence of min/max.
    if np.issubdtype(reshaped_signal.dtype, np.floating):
        nan_mask = np.isnan(reshaped_signal)
        all_nan = nan_mask.all(axis=2)
        argmin_in_windows = np.argmin(np.where(nan_mask, np.inf, reshaped_signal), axis=2)
        argmax_in_windows = np.argmax(np.where(nan_mask, -np.inf, reshaped_signal), axis=2)
    else:
        all_nan = np.zeros(reshaped_signal.shape[:2], dtype=bool)
        argmin_in_windows = np.argmin(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)
        argmax_in_windows = np.argmax(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)
    min_vals_in_windows = np.take_along_axis(reshaped_signal, argmin_in_windows[..., None], axis=2)[..., 0]
    max_vals_in_windows = np.take_along_axis(reshaped_signal, argmax_in_windows[..., None], axis=2)[..., 0]
    if all_nan.any():
        min_vals_in_windows = np.where(all_nan, np.nan, min_vals_in_windows)
        max_vals_in_windows = np.where(all_nan, np.nan, max_vals_in_windows)

    # Initialize the output array: each window produces 2 points
    downsampled_signal_2d = np.empty((n_channels, num_windows * 2), dtype=signal.dtype)

    # Determine the temporal order of min and max within each window
    # mask_min_first[ch, win_idx] is True if min occurred before max in that window
    mask_min_occurred_first = argmin_in_windows < argmax_in_windows
    # mask_max_occurred_first covers cases where max is strictly first,
    # or if min_idx == max_idx (then argmin_in_windows < argmax_in_windows is False)
    mask_max_occurred_first_or_simultaneously = argmin_in_windows >= argmax_in_windows

    # Populate the downsampled signal
    for ch in range(n_channels):
        # Slicing for current channel
        ch_min_vals = min_vals_in_windows[ch, :]
        ch_max_vals = max_vals_in_windows[ch, :]
        ch_mask_min_first = mask_min_occurred_first[ch, :]
        ch_mask_max_first_or_simult = mask_max_occurred_first_or_simultaneously[ch, :]

        # Get indices in the output array for min/max pairs
        # Example: if num_windows = 3, output length = 6
        # even_indices = [0, 2, 4], odd_indices = [1, 3, 5]
        even_output_indices = np.arange(num_windows) * 2
        odd_output_indices = even_output_indices + 1

        # Case 1: Min value appeared first in the window
        if np.any(ch_mask_min_first):
            output_indices_for_min = even_output_indices[ch_mask_min_first]
            output_indices_for_max = odd_output_indices[ch_mask_min_first]
            downsampled_signal_2d[ch, output_indices_for_min] = ch_min_vals[ch_mask_min_first]
            downsampled_signal_2d[ch, output_indices_for_max] = ch_max_vals[ch_mask_min_first]

        # Case 2: Max value appeared first or simultaneously with min in the window
        if np.any(ch_mask_max_first_or_simult):
            output_indices_for_max = even_output_indices[ch_mask_max_first_or_simult]
            output_indices_for_min = odd_output_indices[ch_mask_max_first_or_simult]
            downsampled_signal_2d[ch, output_indices_for_max] = ch_max_vals[ch_mask_max_first_or_simult]
            downsampled_signal_2d[ch, output_indices_for_min] = ch_min_vals[ch_mask_max_first_or_simult]

    # Calculate the actual sampling rate of the output signal
    # Each original window of `window_size` samples produces 2 output samples.
    # The duration of `window_size` original samples is `window_size / original_fs`.
    # So, 2 output samples span a time of `window_size / original_fs`.
    # The time per output sample is `(window_size / original_fs) / 2`.
    # The actual output sampling rate is `1 / ((window_size / original_fs) / 2) = (2 * original_fs) / window_size`.
    actual_output_fs = (2.0 * original_fs) / window_size

    if input_signal_is_1d:
        return downsampled_signal_2d.ravel(), actual_output_fs
    else:
        return downsampled_signal_2d, actual_output_fs