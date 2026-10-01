Changes
-------------------

Unreleased (after v2.0.0): correctness fixes
"""""""""""""""""""""""""""""""""""""""""""""

These are bug fixes, but several of them **change numerical results**. If you need to compare
with results computed by v2.0.0 or earlier, check the items below.

Signal (:mod:`brainmaze_utils.signal`)
''''''''''''''''''''''''''''''''''''''

- ``decimate``: the 16th-order Butterworth filter was designed in ``(b, a)`` form and became
  unstable for ratios ``fs / fs_new >= ~10``. The output was then **entirely NaN, with no error**
  (e.g. 3000->250, 4000->250, 1000->50, 32000->1000 Hz) or overflowed. It now uses second-order
  sections. Other changes:

  - Integer ratios pick every q-th filtered sample. Non-integer ratios use polyphase resampling
    with the exact rational ratio, so long recordings do not drift in time.
  - NaN gaps are interpolated for filtering and then **re-applied** to the output. Before, they
    were silently filled with the channel mean.
  - ``datarate=True`` works for multichannel input.
- ``nandecimate``: NaNs were filled with the NaN *fraction* (0-1) instead of the channel mean. Next
  to every gap this caused errors of the order of the DC offset (47 uV for a 100 uV offset). Fixed.
- ``resample``: crashed on NumPy 2 (``np.NaN``). The time axes pinned both endpoints, so the
  effective output rate was ``(N_new - 1) / (N - 1) * fs`` (1000->300 Hz: 299.3 Hz). Output sample
  ``k`` is now exactly at ``k / fs_new``. It still applies **no anti-aliasing filter, by design**;
  this is now documented prominently.
- ``LowFrequencyFilter``: removed the large jumps at the start and end of the output.

  - Root causes: zero padding turned any DC offset into a step, and the padding was only
    ``2 * n_order * 2**n_decimate`` samples (24 ms for a 0.5 Hz filter at 8 kHz) instead of
    several cutoff periods.
  - Now a least-squares line (offset and drift) is removed and added back. Each edge is
    extended by ``3 / cutoff`` seconds: a local line is extrapolated and the residual is mirrored
    about it.
  - Example: 8 kHz, 0.5 Hz high-pass, 2000 uV offset with 40 uV/s drift. The edge error fell
    from ~1000-1400 uV to ~0.7 uV, which is the same as for the signal without offset and drift.
  - The interior frequency response is unchanged. IIR filters now use second-order sections.
    N-D input is supported. NaN input raises ``ValueError``.
- ``fft_filter``: the frequency axis was ``linspace(0, fs, n)`` and the cutoff bin was removed on
  one side only. A 10 Hz tone with a 9.5 Hz cutoff came out at half amplitude from both ``'lp'``
  and ``'hp'``. Bins are now exact DFT frequencies, applied symmetrically. NaN/inf raise
  ``ValueError`` (an FFT spreads one NaN over the whole signal).
- ``buffer``: ``overlap >= segm_size`` used to hang in an infinite loop; it now raises.
  Non-integer ``fs`` no longer accumulates drift (0.24 s per hour at 499.9 Hz before). 2-D input
  raises instead of being buffered along the channel axis.
- ``detrend``: the default is now a least-squares line. It used to be the line through the two
  endpoints, which a single noisy endpoint tilted. ``method='endpoints'`` restores the old
  behaviour.
- ``get_datarate`` accepts 1-D input. ``unify_sampling_frequency`` no longer replaces the arrays in
  the caller's list. ``downsample_min_max`` ignores NaN inside a window.

Statistics (:mod:`brainmaze_utils.stat`)
''''''''''''''''''''''''''''''''''''''''

- ``combine_mvgauss_distributions``: the between-group term used an element-wise square instead
  of the outer product, so the cross-covariances were wrong (+1.0 instead of -1.0 in the test
  case). Fixed; the result now equals the pooled (``bias=True``) covariance exactly.
- ``kl_divergence_nonparametric``: inputs are normalised. Bins with ``q == 0, p > 0`` now give
  ``inf`` (they were silently dropped and gave 0). An optional ``eps`` adds smoothing.

Annotations (:mod:`brainmaze_utils.annotations`)
''''''''''''''''''''''''''''''''''''''''''''''''

- ``create_day_indexes``:

  - On pandas 3, every epoch got day 0 (a chained assignment that did nothing). On pandas 2 the
    last day could be missed.
  - The ``tzinfo`` argument was ignored.
  - Rewritten: it returns a sorted copy with a ``day`` column computed on the wall clock of the
    chosen timezone.
- ``merge_annotations`` sorts by start first, and it and ``tile_annotations`` preserve extra
  columns such as ``channel``. These used to raise. ``load_CyberPSG(..., tile=...)`` therefore
  works on files with channel annotations.
- ``time_to_utc``, ``time_to_local``, ``time_to_timezone``, ``time_to_timestamp`` and
  ``create_duration`` return copies instead of modifying the input frame.
- CyberPSG:

  - Annotation types that share a name are no longer dropped (they used to load as
    ``error_unknown``).
  - Labels survive a save/load round trip: ``'N2'`` no longer comes back as ``'N2_best'``. The
    standard UUIDs are kept.
  - 7-digit (.NET) fractional seconds are parsed.
- NSRR: R&K stages 3 and 4 both map to ``N3``. Before, any file containing Stage 3 raised
  ``KeyError``.

Other
'''''

- ``vector.get_mutual_vectors(x)`` without labels raised ``UnboundLocalError``.
- ``vector.translate`` returns a new array and promotes integer input to float. It used to work
  in place and truncate.
- ``vector.rotate``: the direction convention is documented (positive = clockwise, about the
  points' mean).
- ``types.ObjDict``: reading a missing attribute raises ``AttributeError`` instead of creating
  the key.
- ``files.get_files`` only skips AppleDouble files (basename starting with ``._``), not every
  path that contains ``._``.
- Packaging:

  - ``tqdm`` is a runtime dependency. It was imported but only listed under the test extra, so
    a clean install failed to import ``annotations``.
  - ``pytz`` was removed (unused).
  - Lower bounds: ``numpy>=1.24``, ``scipy>=1.10``, ``pandas>=2.0``.
  - The stale committed ``egg-info`` was removed.
