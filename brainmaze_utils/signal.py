# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Tools for digital signal processing: fft filtering, low-frequency filtering utilizing downsampling etc.
"""

import numpy as np
import scipy.signal as signal
import scipy.fft as fft

def get_datarate(x):
    """
    Calculates datarate based on NaN values

    Parameters
    ----------
    x : numpy ndarray / list

    Returns
    -------
    numpy ndarray

    flaot values 0-1 with relative NaN count per signal.
    """
    if isinstance(x, np.ndarray):
        return 1 - (np.isnan(x).sum(axis=1)  / (x.shape[1]))
    else:
        return [1-(np.isnan(x_).sum()/x_.squeeze().shape[0]) for x_ in x]

def decimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Downsample signal with anti-aliasing filter (Butterworth - 16th order).
    Works for signals with NaN values - replaced by zero.
    Can return also data-rate. Can take matrix where signals are stacked in 0th dim.

    Parameters
    ----------
    x : np ndarray
        shape[n_signal, n_sample], shape[n_sample] - can process multiple signals.
    fs : float
        Signals fs
    fs_new : float
        Downsample to fs_new
    cutoff : float
        Elective cutoff freq. of anti-alias. filter. By default 2/3 of 0.5*fs_new
    datarate : bool
        If return datarate of signals

    Returns
    -------
    numpy ndarray / tuple

    """

    x = x.copy()
    if datarate is True:
        datarate = get_datarate(x)

    if isinstance(cutoff, type(None)):
        cutoff = fs_new / 3 # two 3rds of half-sampling

    b_multiple_signals = True
    if x.ndim == 1:
        b_multiple_signals = False
        x = x.reshape(1, -1)

    for idx in range(x.shape[0]):
        x[idx, np.isnan(x[idx, :])] = np.nanmean(x[idx, :])

    b, a = signal.butter(16, cutoff/(0.5*fs), 'lp', analog=False)
    #b, a = signal.butter(3, cutoff/(0.5*fs), 'lp', analog=False)
    #a = [1]
    #b = signal.firwin(100, cutoff/(0.5*fs), pass_zero=True)
    #b /= b.sum()




    x = signal.filtfilt(b, a, x, axis=1)

    n_resampled = int(np.round((fs_new / fs) * x.shape[1]))
    x = signal.resample(x, n_resampled, axis=1)

    if b_multiple_signals is False:
        x = x.squeeze()

    if datarate:
        return x, datarate
    return x
    # PLOT AMPLITUDE CHAR
    #w, h = signal.freqs(b, a)
    #w = 0.5 * fs * w / w.max()
    #plt.semilogx(w, 20 * np.log10(abs(h) / (abs(h)).max()))
    #plt.title('Butterworth filter frequency response')
    #plt.xlabel('Frequency [radians / second]')
    #plt.ylabel('Amplitude [dB]')
    #plt.margins(0, 0.1)
    #plt.grid(which='both', axis='both')
    #plt.axvline(125, color='green') # cutoff frequency
    #plt.show()

def nandecimate(x, fs, fs_new, cutoff=None, datarate=False):
    """
    Downsample signal with anti-aliasing filter (Butterworth - 16th order).
    Works for signals with NaN values - replaced by zero.
    Can return also data-rate. Can take matrix where signals are stacked in 0th dim.

    Parameters
    ----------
    x : np ndarray
        shape[n_signal, n_sample], shape[n_sample] - can process multiple signals.
    fs : float
        Signals fs
    fs_new : float
        Downsample to fs_new
    cutoff : float
        Elective cutoff freq. of anti-alias. filter. By default 2/3 of 0.5*fs_new
    datarate : bool
        If return datarate of signals

    Returns
    -------
    numpy ndarray / tuple

    """

    x = x.copy()
    if datarate is True:
        datarate = get_datarate(x)

    if isinstance(cutoff, type(None)):
        cutoff = fs_new / 3 # two 3rds of half-sampling

    b_multiple_signals = True
    if x.ndim == 1:
        b_multiple_signals = False
        x = x.reshape(1, -1)


    nans = np.isnan(x)
    for idx in range(x.shape[0]):
        chmean = np.nanmean(x[idx, :])
        x[idx, nans[idx, :]] = np.nanmean(nans[idx, :])

   #for idx in range(x.shape[0]):
    #    idxes = np.where(nans[idx, :])[0]
    #    chmean = np.nanmean(x[idx, :])
    #    for i in tqdm(list(idxes)):
    #        loc_mean = np.nanmean(x[idx, i-100:i+100])
    #        if np.isnan(loc_mean): loc_mean = chmean
    #        x[idx, i] = loc_mean


    #b, a = signal.butter(16, cutoff/(0.5*fs), 'lp', analog=False)
    #b, a = signal.butter(3, cutoff/(0.5*fs), 'lp', analog=False)
    a = [1]
    b = signal.firwin(30, cutoff/(0.5*fs), pass_zero=True)
    b /= b.sum()

    x = signal.filtfilt(b, a, x, axis=1)

    n_resampled = int(np.round((fs_new / fs) * x.shape[1]))
    x = signal.resample(x, n_resampled, axis=1)
    nans = signal.resample(nans, n_resampled, axis=1)
    x[nans > 0.5] = np.nan

    if b_multiple_signals is False:
        x = x.squeeze()

    if datarate:
        return x, datarate
    return x
    # PLOT AMPLITUDE CHAR
    #w, h = signal.freqs(b, a)
    #w = 0.5 * fs * w / w.max()
    #plt.semilogx(w, 20 * np.log10(abs(h) / (abs(h)).max()))
    #plt.title('Butterworth filter frequency response')
    #plt.xlabel('Frequency [radians / second]')
    #plt.ylabel('Amplitude [dB]')
    #plt.margins(0, 0.1)
    #plt.grid(which='both', axis='both')
    #plt.axvline(125, color='green') # cutoff frequency
    #plt.show()

def unify_sampling_frequency(x : list, sampling_frequency: list, fs_new=None) -> tuple:
    """
    Takes list of signals and list of frequencies and downsamples to the same sampling frequency.
    If all frequencies are same and fs_new is not specified, no operation performed. If not all frequencies are the same
    and fs_new is not specified, downsamples all signals on the lowest fs present in the list. If fs_new is specified,
    signals will be processed and downsampled on that frequency. If all sampling frequencies == fs_new, nothing is performed.

    Parameters
    ----------
    x : list
        list of numpy signals for downsampling
    sampling_frequency : list
        for each signal
    fs_new : float
        new sampling frequency

    Returns
    -------
    tuple - (numpy ndarray, new_freq)
    """

    b_process = False

    if not isinstance(x, list):
        raise TypeError('First variable must be list of numpy arrays')

    if not isinstance(sampling_frequency, (list, np.ndarray)):
        raise TypeError('Second parameter must be list or array of floats/integers')
    sampling_frequency = np.array(sampling_frequency)

    if x.__len__() != sampling_frequency.__len__():
        raise AssertionError('Length of a signal list must be same as length of sampling_frequency list')

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
            sig = x[idx]
            sig = decimate(sig, fs, fs_new)
            x[idx] = sig
            sampling_frequency[idx] = fs_new

    return x, fs_new

def fft_filter(X:np.ndarray, fs:float, cutoff:float, type:str='lp'):
    """
    FFT filter

    Filters along the last axis, so a 2-D array of shape ``(n_signals, n_samples)`` is
    filtered per-signal.

    Parameters
    ----------
    X : numpy.ndarray
        Signal ``(n_samples,)`` or a stack of signals ``(..., n_samples)``.
    fs : float
    cutoff : float
    type : str
        'lp' or 'hp'

    Returns
    -------
    numpy.ndarray
        Same shape as ``X``.

    """
    if type not in ('lp', 'hp'):
        raise ValueError(f"type must be 'lp' or 'hp', got {type!r}")

    n_samples = X.shape[-1]
    Xs = fft.fft(X, axis=-1)
    freq = np.linspace(0, fs, n_samples)
    above = np.where(freq > cutoff)[0]
    if above.size == 0:  # cutoff at or above the top bin -> nothing above it to remove
        stop = pos = n_samples
    else:
        pos = int(above[0])
        stop = n_samples - pos  # symmetric upper bound == original [pos:-pos]

    if type == 'lp':
        X_new = Xs.copy()
        X_new[..., pos:stop] = 0
    else:  # 'hp'
        X_new = np.zeros_like(Xs)
        X_new[..., pos:stop] = Xs[..., pos:stop]
    return np.real(fft.ifft(X_new, axis=-1))

def buffer(x:np.ndarray, fs:float=1, segm_size:float=None, overlap:float = 0, drop:bool=True):
    """
    Buffer signal into matrix

    Parameters
    ----------
    x : np.ndarray
        Signal to be
    fs : float
        Sampling frequency
    segm_size : float
        Segment size in seconds
    overlap : float
        Overlap size in seconds
    drop : bool
        Drop last segment if True, else append zeros
    Returns
    -------
    buffered_signal : np.ndarray
    """

    if not isinstance(x, np.ndarray):
        pass
    if x.ndim != 1:
        pass

    if isinstance(segm_size, type(None)):
        return x

    n_segm = int(round(fs * segm_size))
    #n_overlap = int(round(fs * overlap))
    n_shift = int(round(fs * (segm_size - overlap)))
    idx = 0

    buffered_signal = []
    while idx+n_segm-n_shift < x.shape[0]:
        app = x[idx:idx+n_segm]
        if app.__len__() < n_segm:
            if drop == False:
                app = np.append(app, np.zeros(n_segm - app.__len__()))
            else:
                idx = 2**30
                app = None

        if not isinstance(app, type(None)):
            buffered_signal.append(app)
        idx += n_shift
    return np.array(buffered_signal)

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
        Apply low-frequency filter using decimation/upsampling strategy.

        Parameters
        ----------
        X : numpy.ndarray
            Input signal

        Returns
        -------
        numpy.ndarray
            Filtered signal
        """
        # append for filter
        X = np.concatenate((np.zeros(self.n_append), X, np.zeros(self.n_append)), axis=0)

        # append to divisible by 2
        C = int(2**np.ceil(np.log2(X.shape[0] + 2*self.n_append))) - X.shape[0]
        X = np.append(np.zeros(C), X)

        for k in range(self.n_decimate):
            X = self.decimate(X)

        X = signal.filtfilt(self.b_filt, self.a_filt, X)

        for k in range(self.n_decimate):
            X = self.upsample(X)
        #X = self.upsample(X)

        X = X[self.n_append + C : -self.n_append]
        return X


    def __call__(self, X):
        # append for filter
        X_orig = X.copy()
        X = self.filter_signal(X)
        if self.filter_type == 'lp': return X
        if self.filter_type == 'hp': return X_orig - X

def resample(x, fsamp_orig, fsamp_new):
    """
    Resample a signal from an old sampling frequency to a new sampling frequency. Works with nans.

    Parameters
    ----------
    x : numpy.ndarray
        The input signal to be resampled.
    fsamp_orig : float
        The original sampling frequency of the signal.
    fsamp_new : float
        The new sampling frequency to resample the signal to.

    Returns
    -------
    numpy.ndarray
        The resampled signal.
    """

    if fsamp_orig == fsamp_new:
        return x

    nans = np.isnan(x)
    var = np.nanstd(x)
    mu = np.nanmean(x)
    x = (x - mu) / var
    x[np.isnan(x)] = 0

    xp = np.linspace(0, 1, len(x))
    xi = np.linspace(0, 1, int(np.round(len(x) * fsamp_new / fsamp_orig)))

    x = np.interp(xi, xp, x)
    nans = np.interp(xi, xp, nans)


    x[nans >= 0.5] = np.NaN
    x = (x * var) + mu
    return x

def detrend(y, x=None, y2=None):
    """
        Detrends the input signal by removing the linear trend.

        Parameters
        ----------
        y : numpy.ndarray
            The input signal to be detrended.
        x : numpy.ndarray, optional
            The x-values corresponding to the y-values. If None, a linear space is used.
        y2 : numpy.ndarray, optional
            An additional signal to be detrended using the same trend as y.

        Returns
        -------
        numpy.ndarray
            The detrended signal.
        tuple of numpy.ndarray
            The detrended signals if y2 is provided.
    """

    if isinstance(x, type(None)):
        x = np.linspace(0, 1, y.shape[0])

    a = (y[0] - y[-1]) / (x[0] - x[-1])
    b = y[0] - (x[0]*a)
    if isinstance(y2, type(None)):
        return y - (x*a+b)

    return y - (x * a + b), y2 - (x*a+b)

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

    # Find min and max values within each window
    min_vals_in_windows = np.min(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)
    max_vals_in_windows = np.max(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)

    # Find indices of min and max values within each window (relative to window start)
    # np.argmin and np.argmax return the index of the *first* occurrence of min/max
    argmin_in_windows = np.argmin(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)
    argmax_in_windows = np.argmax(reshaped_signal, axis=2)  # Shape: (n_channels, num_windows)

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