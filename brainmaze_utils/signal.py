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
"""

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
    sample period (``+-0.5 / fs_new``) of its time stamp is NaN.
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
    return out


def _downsample_filtered(x, fs, fs_new, n_new):
    """
    Pick samples at times ``k / fs_new`` from an already low-passed signal
    ``(n_signals, n_samples)``. Integer ratios use exact sample picking; non-integer
    ratios use polyphase resampling (:func:`scipy.signal.resample_poly`).
    """
    ratio = fs / fs_new
    if abs(ratio - round(ratio)) < 1e-9:
        q = int(round(ratio))
        y = x[:, ::q]
    else:
        from fractions import Fraction
        frac = Fraction(fs_new / fs).limit_denominator(1000)
        y = signal.resample_poly(x, frac.numerator, frac.denominator, axis=1, padtype='line')
    if y.shape[1] >= n_new:
        return y[:, :n_new]
    # can only happen for non-integer ratios when rounding gives one more sample
    pad = np.repeat(y[:, -1:], n_new - y.shape[1], axis=1)
    return np.concatenate((y, pad), axis=1)


def decimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Downsample signal(s) with a zero-phase anti-aliasing low-pass filter. NaN-aware.

    Processing steps, per signal:

    1. NaN samples are temporarily filled by linear interpolation between the
       neighbouring valid samples (leading/trailing NaNs take the nearest valid
       value) so the filter does not see steps at gap edges.
    2. Zero-phase low-pass: 16th-order Butterworth in second-order sections,
       applied forward and backward (:func:`scipy.signal.sosfiltfilt`); the
       magnitude response is therefore that of a 32nd-order filter with
       -6 dB at ``cutoff``.
    3. Downsampling to ``n_new = round(n_samples * fs_new / fs)`` samples placed at
       ``k / fs_new`` s. Integer ratios ``fs / fs_new`` pick every ``q``-th sample
       exactly; non-integer ratios use :func:`scipy.signal.resample_poly`.
    4. The NaN mask is re-applied: an output sample is NaN if any input sample
       within ``+-0.5 / fs_new`` s of it was NaN. Signals that are entirely NaN
       stay entirely NaN.

    Parameters
    ----------
    x : numpy.ndarray
        ``(n_samples,)`` or ``(n_signals, n_samples)`` - samples along the last axis.
        Not modified.
    fs : float
        Sampling frequency of ``x`` in Hz.
    fs_new : float
        Target sampling frequency in Hz. Must satisfy ``0 < fs_new <= fs``.
    cutoff : float, optional
        Anti-aliasing cutoff in Hz. Default ``fs_new / 3`` (two thirds of the new
        Nyquist frequency). Must be ``< fs / 2``.
    datarate : bool
        If ``True``, also return :func:`get_datarate` of the **input**.

    Returns
    -------
    numpy.ndarray or tuple
        Decimated signal(s) with the same number of dimensions as ``x`` (``float64``),
        or ``(decimated, datarate)`` if ``datarate=True``.

    Notes
    -----
    - If ``fs_new == fs`` the input is returned unchanged (as a float copy); no
      filtering is applied.
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
       now re-applied to the output instead of being silently filled.
    """
    return_datarate = datarate is True
    x = np.array(x, dtype=float)  # copy, also promotes ints

    if fs_new <= 0 or fs <= 0:
        raise ValueError(f'fs and fs_new must be > 0, got fs={fs!r}, fs_new={fs_new!r}')
    if fs_new > fs:
        raise ValueError(f'decimate only downsamples (fs_new={fs_new!r} > fs={fs!r}); use resample() to upsample')

    dr = get_datarate(x) if return_datarate else None

    if fs_new == fs:
        return (x, dr) if return_datarate else x

    if cutoff is None:
        cutoff = fs_new / 3  # two thirds of the new Nyquist
    if not 0 < cutoff < fs / 2:
        raise ValueError(f'cutoff must be in (0, fs/2), got {cutoff!r} for fs={fs!r}')

    b_multiple_signals = x.ndim != 1
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise ValueError(f'x must be 1-D or 2-D, got shape {x.shape}')

    nans = np.isnan(x)
    _fill_nans_linear(x)

    sos = signal.butter(16, cutoff, 'lp', fs=fs, output='sos')
    # long odd-extension padding (~6 cutoff periods) keeps edge transients small
    padlen = int(min(x.shape[1] - 1, np.ceil(6 * fs / cutoff)))
    x = signal.sosfiltfilt(sos, x, axis=1, padlen=padlen)

    n_new = int(np.round((fs_new / fs) * x.shape[1]))
    x = _downsample_filtered(x, fs, fs_new, n_new)
    x[_decimated_nan_mask(nans, fs, fs_new, n_new)] = np.nan

    if not b_multiple_signals:
        x = x[0]

    if return_datarate:
        return x, dr
    return x


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
      decimated to ``fs_new``.

    Signals already at the target frequency are passed through unchanged (they are
    **not** low-passed).

    Parameters
    ----------
    x : list of numpy.ndarray
        Signals (1-D, or 2-D with samples along the last axis).
    sampling_frequency : list or numpy.ndarray
        Sampling frequency in Hz of each signal.
    fs_new : float, optional
        Target sampling frequency in Hz; must not exceed any input frequency.

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

def fft_filter(X:np.ndarray, fs:float, cutoff:float, type:str='lp'):
    """
    Ideal (brick-wall) FFT filter.

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
        along the **last** axis. Not modified.
    fs : float
        Sampling frequency in Hz.
    cutoff : float
        Cutoff frequency in Hz (``>= 0``). A cutoff at or above Nyquist makes ``'lp'``
        the identity and ``'hp'`` return zeros.
    type : str
        ``'lp'`` or ``'hp'``.

    Returns
    -------
    numpy.ndarray
        Same shape as ``X``.

    Raises
    ------
    ValueError
        If ``X`` contains NaN or inf. An FFT spreads a single NaN over the whole
        signal, so gaps must be handled (filled or cut out) before calling.

    Notes
    -----
    A brick-wall filter assumes the signal is periodic and rings (Gibbs phenomenon)
    around sharp transients and at the record edges. No windowing or padding is applied.
    """
    if type not in ('lp', 'hp'):
        raise ValueError(f"type must be 'lp' or 'hp', got {type!r}")
    if fs <= 0:
        raise ValueError(f"fs must be > 0, got {fs!r}")
    if cutoff < 0:
        raise ValueError(f"cutoff must be >= 0, got {cutoff!r}")
    X = np.asarray(X)
    if not np.all(np.isfinite(X)):
        raise ValueError('fft_filter: X contains NaN/inf; fill or remove gaps before filtering')

    n_samples = X.shape[-1]
    Xs = fft.fft(X, axis=-1)
    freq = np.abs(np.fft.fftfreq(n_samples, d=1.0 / fs))
    keep = freq <= cutoff if type == 'lp' else freq > cutoff
    X_new = np.where(keep, Xs, 0)
    return np.real(fft.ifft(X_new, axis=-1))

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

class LowFrequencyFilter:
    """
        Parameters
        ----------
        fs : float
            sampling frequency
        cutoff : float
            frequency cutoff
        n_decimate : int
            how many times the signal will be downsampled before the low frequency filtering
        n_order : int
            n-th order filter used for filtration
        dec_cutoff : float
            relative frequency at which the signal will be filtered when downsampled
        filter_type : str
            Which side of ``cutoff`` is returned:

            - ``'lp'`` returns the low-frequency content **below** ``cutoff``.
            - ``'hp'`` returns ``x`` minus that low-frequency content, i.e. the
              high-frequency content **above** ``cutoff`` (use this to remove slow drift).

            Pick the one that matches the content you want to keep -- see the examples.
        ftype : str
            'fir' or 'iir'


        .. code-block:: python

            x = np.random.randn(10000)

            # keep slow content below the cutoff (e.g. isolate drift / a slow oscillation)
            lowpass = LowFrequencyFilter(fs=fs, cutoff=cutoff, n_decimate=2, n_order=101, filter_type='lp')
            x_slow = lowpass(x)

            # remove slow drift, keeping content above the cutoff
            highpass = LowFrequencyFilter(fs=fs, cutoff=cutoff, n_decimate=2, n_order=101, filter_type='hp')
            x_detrended = highpass(x)
    """

    __version__ = '0.0.2'

    def __init__(self, fs=None, cutoff=None, n_decimate=1, n_order=None, dec_cutoff=0.3, filter_type='lp', ftype='fir'):
        self.fs = fs
        self.cutoff = cutoff
        self.n_decimate = n_decimate
        self.dec_cutoff = dec_cutoff
        self.filter_type = filter_type

        self.n_order = n_order
        self.ftype = ftype

        self.n_append = None

        self.design_filters()

    def design_filters(self):
        """
        Design decimation and filtering coefficients based on filter type (FIR or IIR).
        """
        if self.ftype == 'fir':
            if isinstance(self.n_order, type(None)): self.n_order = 101
            self.n_append = (2 * self.n_order) * (2**self.n_decimate)

            self.a_dec = [1]
            self.b_dec = signal.firwin(self.n_order, self.dec_cutoff, pass_zero=True)
            self.b_dec /= self.b_dec.sum()

            self.a_filt = [1]
            self.b_filt = signal.firwin(self.n_order, 2 * self.cutoff / (self.fs/2**self.n_decimate), pass_zero=True)
            self.b_filt /= self.b_filt.sum()

        elif self.ftype == 'iir':
            if isinstance(self.n_order, type(None)): self.n_order = 3
            self.n_append = (2 * self.n_order) * (2**self.n_decimate)

            self.b_dec, self.a_dec = signal.butter(self.n_order, self.dec_cutoff, btype='low')
            self.b_filt, self.a_filt = signal.butter(self.n_order,  2 * self.cutoff / (self.fs/2**self.n_decimate), btype='low')

        else: raise AssertionError(f'[INPUT ERROR]: ftype must be \'iir\' or \'fir\'')

    def decimate(self, X):
        """
        Apply anti-aliasing filter and downsample signal by factor of 2.

        Parameters
        ----------
        X : numpy.ndarray
            Input signal

        Returns
        -------
        numpy.ndarray
            Downsampled signal
        """
        X = signal.filtfilt(self.b_dec, self.a_dec, X)
        return X[::2]

    def upsample(self, X):
        """
        Upsample signal by factor of 2 using zero-insertion and filtering.

        Parameters
        ----------
        X : numpy.ndarray
            Input signal

        Returns
        -------
        numpy.ndarray
            Upsampled signal
        """
        X_up = np.zeros(X.shape[0] * 2)
        X_up[::2] = X
        X_up = signal.filtfilt(self.b_dec, self.a_dec, X_up) * 2
        return X_up

    def filter_signal(self, X):
        """
        Low-pass ``X`` below ``cutoff`` using the decimate -> filter -> upsample scheme.

        Parameters
        ----------
        X : numpy.ndarray
            1-D signal. Must not contain NaN (NaN propagates through the filters and
            the output becomes NaN).

        Returns
        -------
        numpy.ndarray
            Low-frequency content of ``X``, same length.

        Notes
        -----
        The signal mean is removed before filtering and added back afterwards, and
        the signal is extended at both ends by even (mirror) reflection. The
        reflection is continuous in value, so a DC offset does not create artificial
        steps at the record edges.

        .. note:: **Changed after v2.0.0:**
           Previously the signal was zero-padded, which turned any DC offset into a
           step at both edges: for a signal at 100 (e.g. uV) with ``filter_type='hp'``
           the output deviated by up to ~50 over the first/last ~1.5-2.3 s.
        """
        X = np.asarray(X, dtype=float)
        mu = X.mean() if X.size else 0.0
        X = X - mu
        n = X.shape[0]

        # extend for filter transients + make length divisible by 2**n_decimate
        total = n + 2 * self.n_append
        C = (-total) % (2 ** self.n_decimate)
        pad_front = self.n_append + C
        pad_back = self.n_append
        if n > 1:
            X = np.pad(X, (pad_front, pad_back), mode='reflect')
        else:
            X = np.pad(X, (pad_front, pad_back), mode='edge')

        for k in range(self.n_decimate):
            X = self.decimate(X)

        X = signal.filtfilt(self.b_filt, self.a_filt, X)

        for k in range(self.n_decimate):
            X = self.upsample(X)

        X = X[pad_front: pad_front + n]
        return X + mu


    def __call__(self, X):
        """
        Apply the filter.

        Returns the low-frequency content (``filter_type='lp'``) or ``X`` minus the
        low-frequency content (``filter_type='hp'``). ``X`` is not modified.
        """
        X_orig = np.asarray(X, dtype=float).copy()
        X = self.filter_signal(X_orig)
        if self.filter_type == 'lp': return X
        if self.filter_type == 'hp': return X_orig - X
        raise ValueError(f"filter_type must be 'lp' or 'hp', got {self.filter_type!r}")

def resample(x, fsamp_orig, fsamp_new):
    """
    Resample a signal to a new sampling frequency by linear interpolation. NaN-aware.

    .. warning::
       **No anti-aliasing filter is applied.** When downsampling, any content above
       the new Nyquist frequency ``fsamp_new / 2`` aliases into the output. Low-pass
       the signal below the new Nyquist first (e.g. use :func:`decimate`, which
       filters and downsamples) - this is deliberately left to the caller.

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
      Values are never interpolated across a gap.
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
    x2 = x.reshape(-1, n)
    out = np.full((x2.shape[0], n_new), np.nan)
    if n == 0 or n_new == 0:
        return out.reshape(lead_shape + (n_new,))

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