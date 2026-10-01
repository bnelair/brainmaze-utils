Changes
-------------------

Unreleased (after v2.0.0): correctness fixes
"""""""""""""""""""""""""""""""""""""""""""""

These are bug fixes, but several of them **change numerical results**, and some calls that used
to return (often wrong) results now raise. If you compare with results computed by v2.0.0 or
earlier, or upgrade a pipeline, check the items below. Every behaviour change is listed; the
first table collects every new exception and warning.

New exceptions and warnings
'''''''''''''''''''''''''''

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Function
     - Raises when
     - v2.0.0 did
   * - ``signal.decimate``
     - ``ValueError``: ``fs <= 0`` or ``fs_new <= 0``; ``cutoff <= 0``; ``cutoff >= fs_new / 2``
       when downsampling (would alias); ``x`` with more than 2 dimensions
     - aliased silently for a high ``cutoff``; crashed or returned garbage otherwise
   * - ``signal.resample``
     - ``ValueError``: a sampling frequency ``<= 0``
     - divided by zero / returned garbage
   * - ``signal.LowFrequencyFilter``
     - ``ValueError`` at construction: ``fs`` or ``cutoff`` missing or ``<= 0``; ``n_decimate`` not
       an integer ``>= 0``; ``dec_cutoff`` outside ``(0, 1)``; ``filter_type`` not ``'lp'/'hp'``;
       ``ftype`` not ``'fir'/'iir'``; ``cutoff`` above the decimated Nyquist frequency;
       ``max_gap_fill < 0``. ``ValueError`` when filtering: input contains ``inf`` or is 0-D.
       **UserWarning** at construction: the FIR is too short to resolve ``cutoff`` (low-frequency
       gain at ``cutoff`` above 0.3 instead of ~0.25), e.g. the default 101 taps for 0.5 Hz at
       ``fs / 2**n_decimate = 250`` Hz
     - no validation: an invalid ``filter_type`` returned ``None``, an invalid ``ftype`` failed
       later with ``AttributeError``, a too-high cutoff failed in the filter design; ``inf`` gave
       an all-NaN output
   * - ``signal.fft_filter``
     - ``ValueError``: ``type`` not ``'lp'/'hp'``, ``fs <= 0``, ``cutoff < 0``, invalid ``edges``,
       ``max_gap_fill < 0``, input contains ``inf``
     - wrong or all-NaN output
   * - ``signal.buffer``
     - ``ValueError``: ``overlap < 0`` or ``overlap >= segm_size``; ``segm_size <= 0``; segment or hop
       shorter than one sample; ``x`` not 1-D
     - ``overlap >= segm_size`` **hung forever**; 2-D input was buffered along the channel axis
   * - ``signal.detrend``
     - ``ValueError``: fewer than two finite samples (``'lstsq'``); unknown ``method``
     - n/a (endpoint line only)
   * - ``stat.kl_divergence_nonparametric``
     - ``ValueError``: shapes differ; negative, NaN or inf values; a distribution (row) with zero
       mass; ``eps`` not a finite number ``> 0``
     - returned a number (invalid bins dropped)
   * - ``annotations.create_day_indexes``
     - ``ValueError``: ``start`` holds timezone-naive datetimes
     - accepted naive datetimes
   * - ``annotations.load_NSRR``
     - ``KeyError`` listing **all** unmapped ``EventConcept`` values
     - ``KeyError`` on the first unmapped one (and on every Stage 3 epoch, see below)
   * - CyberPSG loader
     - ``ValueError`` for an unparseable time string (message names the string)
     - ``ValueError`` from ``strptime``, also for valid 7-digit (.NET) fractions
   * - ``types.ObjDict``
     - ``AttributeError`` when reading a missing attribute whose name starts with ``_``
     - created the key, which broke ``copy``/``deepcopy``/``pickle``

Signal (:mod:`brainmaze_utils.signal`)
''''''''''''''''''''''''''''''''''''''

- ``decimate``:

  - The 16th-order Butterworth filter was designed in ``(b, a)`` form and became unstable for
    ratios ``fs / fs_new >= ~10``. The output was then **entirely NaN, with no error** (e.g.
    3000->250, 4000->250, 1000->50, 32000->1000 Hz) or overflowed. It now uses second-order
    sections.
  - The final resampling step no longer uses the FFT (``scipy.signal.resample``, which treats the
    record as periodic and wraps its end into its start). Integer ratios pick every q-th filtered
    sample (exact). **Every other ratio**, including rates with decimals (30000.5, 511.9999,
    24414.0625 Hz) and upsampling, evaluates the filtered signal at exactly ``k / fs_new`` with a
    Kaiser-windowed sinc interpolator (120 dB design): no timing drift however long the record,
    and ~1e-7 interpolation error for in-band content (measured on a 7 Hz tone: 32556->1000 Hz
    1.2e-8, 30000->256 Hz 1.2e-8, 499.907->200 Hz 8e-7; 511.9999->256 Hz over 8 h ends within
    5e-8 of the analytic tone). Results therefore differ from v2.0.0 slightly everywhere and
    noticeably near the record edges, where v2.0.0's FFT wrap-around was wrong.
  - Upsampling (``fs_new > fs``) works for any ratio. v2.0.0 upsampled only for ratios below 1.5
    (and returned NaN between ~1.45 and 1.5). With the default cutoff ``fs_new / 3`` the
    Butterworth is still applied when it lies below ``0.45 * fs`` (ratios below ~1.35, e.g.
    150->200 Hz, as in v2.0.0) and skipped otherwise; the interpolator keeps content up to
    ``0.45 * fs``. Accuracy 150->200 Hz: 4e-7 (v2.0.0: 7e-5). When upsampling, the last output
    sample can lie up to one input period after the last input sample (up to 3 output samples,
    e.g. 250->1000 Hz); these are extrapolated and their error on a unit tone is 0.1 (20 Hz) to ~1
    (near ``0.4 * fs``). When the Butterworth is skipped, content between ``0.45 * fs`` and
    ``fs / 2`` leaves an image (~7 % at ``0.48 * fs``).
  - A user ``cutoff >= fs_new / 2`` raises when downsampling. v2.0.0 accepted it and let content
    above the new Nyquist frequency alias (70 Hz passed at full amplitude for 1000->100 Hz with
    ``cutoff=80``).
  - ``fs_new == fs`` low-pass filters at ``cutoff`` as in v2.0.0.
  - NaN gaps are interpolated for filtering and then **re-applied** to the output: an output
    sample is NaN if any input sample within ``+-0.5 / fs_new`` of it, or one of the two input
    samples bracketing it, was NaN. Before, gaps were silently filled with the channel mean and
    returned as data.
  - ``datarate=True`` works for 1-D and multichannel input. Empty input returns an empty array.
- ``nandecimate``: NaNs were filled with the NaN *fraction* (0-1) instead of the channel mean. Next
  to every gap this caused errors of the order of the DC offset (47 uV for a 100 uV offset). The
  output NaN mask uses the same window rule as ``decimate`` (it used to threshold a resampled mask
  at 0.5).
- ``unify_sampling_frequency``: uses ``decimate``, so everything above applies; in particular
  channels below ``fs_new`` are upsampled for any ratio (brainmaze-eeg calls it with
  ``fs_new=200``). It no longer replaces the arrays in the caller's list.
- ``resample``:

  - Crashed on NumPy 2 (``np.NaN``) and divided by zero for constant signals.
  - The time axes pinned both endpoints, so the effective output rate was
    ``(N_new - 1) / (N - 1) * fs`` (1000->300 Hz: 299.3 Hz). Output sample ``k`` is now exactly at
    ``k / fs_new``.
  - It still applies **no anti-aliasing filter, by design**; this is now documented prominently.
  - NaN: an output sample is NaN if one of its two bracketing input samples is NaN. When
    downsampling, a gap that falls entirely between two output instants marks nothing (a
    3-sample gap at 1000->250 Hz gives no NaN); use ``decimate`` to keep gap masks.
  - Empty input returns an empty array (it raised a reshape error).
- ``LowFrequencyFilter``:

  - Removed the large jumps at the start and end of the output. Root causes: zero padding turned
    any DC offset into a step, and the padding was only ``2 * n_order * 2**n_decimate`` samples
    (24 ms for a 0.5 Hz filter at 8 kHz) instead of several cutoff periods. Now a least-squares
    line (offset and drift) is removed and added back, and each edge is extended by ``3 / cutoff``
    seconds: a local line is extrapolated and the residual is mirrored about it. Example: 8 kHz,
    0.5 Hz high-pass, 2000 uV offset with 40 uV/s drift: the edge error fell from ~1000-1400 uV to
    ~0.7 uV, the same as for the signal without offset and drift. The interior frequency response
    is unchanged.
  - **NaN gaps are handled** (v2.0.0 returned an all-NaN output for any input containing a NaN).
    The output is NaN exactly where the input is NaN. Interior gaps shorter than ``max_gap_fill``
    (new parameter, default ``0.5 / cutoff`` s) are bridged by linear interpolation; longer gaps
    split the signal and every segment is filtered with the record-edge handling above, so there
    is no jump at gap edges. The ``decimate -> LowFrequencyFilter`` cascade therefore works on
    gappy recordings.
    Limits: a level jump across a *bridged* gap is filtered like a real step (500 uV step, 0.1 s
    gap, 0.5 Hz high-pass: >10 uV for about +-1.5 s); lower ``max_gap_fill`` to split instead.
    Segments of a few samples between long gaps give finite but meaningless values. The default
    ``0.5 / cutoff`` is slightly above the measured bridge-versus-split crossover (~0.9 s at 0.5 Hz,
    i.e. ~0.45 / cutoff; the difference near the crossover is small), and is kept.
  - IIR filters use second-order sections (numerically identical response). N-D input is
    supported (time on the last axis).
  - New ``UserWarning`` for FIRs too short to resolve the cutoff (see the table). The docstring
    used to claim a gain of ~0.25 at the cutoff for FIR filters; the default 101 taps for 0.5 Hz
    at 8 kHz with ``n_decimate=5`` actually pass 0.94 at 0.5 Hz (-6 dB at ~1.7 Hz). Use about
    ``n_order >= 1.5 * fs / 2**n_decimate / cutoff`` taps or ``ftype='iir'``. Defaults are
    unchanged.
- ``fft_filter``:

  - The frequency axis was ``linspace(0, fs, n)`` and the cutoff bin was removed on one side only.
    A 10 Hz tone with a 9.5 Hz cutoff came out at half amplitude from both ``'lp'`` and ``'hp'``.
    Bins are now exact DFT frequencies, applied symmetrically.
  - **NaN gaps are handled** like in ``LowFrequencyFilter`` (v2.0.0: all-NaN output).
  - New ``edges`` parameter: ``'periodic'`` (v2.0.0 behaviour, the default for NaN-free input) or
    ``'extend'`` (the ``LowFrequencyFilter`` edge handling, the default when the input has NaN).
    **Caution:** with the default ``edges=None`` one NaN anywhere switches the whole output from
    ``'periodic'`` to ``'extend'`` (differences up to ~1 for SD-172 1/f data far from the gap);
    pass ``edges`` explicitly when the result must not depend on the presence of gaps.
    For a 1 Hz high-pass on a 60 s cut of 1/f data, ``'extend'`` reduces the edge error from
    ~80-100 to ~6-14.
- ``buffer``: ``overlap >= segm_size`` used to hang in an infinite loop; it now raises.
  Non-integer ``fs`` no longer accumulates drift (0.24 s per hour at 499.9 Hz before). 2-D input
  raises instead of being buffered along the channel axis.
- ``detrend``: the default is now a least-squares line fitted to the finite samples. It used to be
  the line through the two endpoints, which a single noisy endpoint tilted. ``method='endpoints'``
  restores the old behaviour.
- ``get_datarate`` accepts 1-D input (it raised). ``downsample_min_max`` ignores NaN inside a
  window (a single NaN used to make both points of its window NaN).

Statistics (:mod:`brainmaze_utils.stat`)
''''''''''''''''''''''''''''''''''''''''

- ``combine_mvgauss_distributions``: the between-group term used an element-wise square instead
  of the outer product, so the cross-covariances were wrong (+1.0 instead of -1.0 in the test
  case). Fixed; the result now equals the pooled (``bias=True``) covariance exactly.
- ``kl_divergence_nonparametric``:

  - The last axis holds the bins. Each distribution (row) is normalised on its own and N-D input
    returns the **sum of the per-row divergences**. For normalised histograms - including the
    ``(n_features, n_bins)`` stacks brainmaze-eeg passes - the value is identical to v2.0.0.
  - **Breaking for raw counts:** v2.0.0 did not normalise and returned
    ``sum(c_p * log(c_p / c_q))``, which grows with the number of samples and is not a divergence.
    Counts now give the KL divergence of the normalised histograms.
  - Bins with ``q == 0, p > 0`` give ``inf`` (they were silently dropped, returning e.g. 0).
  - Optional ``eps`` adds smoothing. Invalid input raises (see the table).

Annotations (:mod:`brainmaze_utils.annotations`)
''''''''''''''''''''''''''''''''''''''''''''''''

- ``create_day_indexes``:

  - On pandas 3, every epoch got day 0 (a chained assignment that did nothing). On pandas 2 the
    last day could be missed (epochs at +0, +6, +20, +30, +44 h from 22:00 UTC with
    ``hour=12`` gave ``[0, 0, 1, 1, 1]`` instead of ``[0, 0, 1, 1, 2]``).
  - The ``tzinfo`` argument was ignored.
  - Rewritten: it returns a sorted copy with a ``day`` column computed on the wall clock of the
    chosen timezone (DST-safe). Numeric timestamps and fractional ``hour`` are accepted;
    timezone-naive datetimes raise.
  - Unchanged but now documented: for UTC data (e.g. from ``load_CyberPSG``) the default cuts days
    at ``hour`` **UTC**; pass ``tzinfo`` for the local wall clock.
- ``merge_annotations``: epochs are grouped by ``annotation`` and by every extra column (e.g.
  ``channel``; missing values incl. ``pd.NA`` form one group), and touching epochs of a group are
  merged. Interleaved channels and overlapping labels therefore merge correctly; unsorted input
  is handled. Extra columns are kept with their dtypes. v2.0.0 merged only rows consecutive in the
  input order and raised on extra columns. For sorted, non-overlapping single-channel input the
  result is unchanged.
- ``tile_annotations`` keeps extra columns (it raised) and, for empty input, the ``duration``
  column. ``load_CyberPSG(..., tile=...)`` therefore works on files with channel annotations.
- ``time_to_utc``, ``time_to_local``, ``time_to_timezone``, ``time_to_timestamp`` and
  ``create_duration`` return copies instead of modifying the input frame, and accept NumPy scalar
  timestamps.
- CyberPSG:

  - **Writer** (``save_CyberPSG``): the 13 standard labels (``AWAKE, N1, N2, N3, REM, UNKNOWN,
    Arousal, N, SLP, IED, seizure, seizure_05, seizure_08``) are written with the suffix ``_bm``
    (``N2`` -> ``N2_bm``); v2.0.0 wrote ``N2_best``. The annotation-type **UUIDs are exactly those
    v2.0.0 wrote** (e.g. ``IED`` -> ``...000000000011``, ``seizure`` -> ``...000000000013``), so
    CyberPSG sees the same types. Input labels ``N2``, ``N2_best`` and ``N2_bm`` are one type (v2.0.0
    raised ``KeyError`` for ``N2`` together with ``N2_best``). Other labels are written as given.
  - **Loader** (``load_CyberPSG``): strips one of the suffixes ``_bm``, ``_best``, ``_aisc``,
    ``_PiesPro`` from every label (v2.0.0: ``_aisc`` and ``_PiesPro`` only). Files written by
    v2.0.0 (``N2_best``), by this version (``N2_bm``) and mixed collections load to the same labels
    (``N2``); a save/load round trip returns the original labels. New ``strip_suffixes=False``
    returns the stored names, e.g. to tell a scorer's ``N2`` from a model's ``N2_best`` in one
    file (both load as ``N2`` by default). The annotation group is still named ``Import_best``.
    Compatibility: the v2.0.0 loader strips only ``_aisc``/``_PiesPro``, so it reads files written
    by this version as ``N2_bm``, ``IED_bm`` etc.; upgrade readers when mixing versions.
  - Annotation types that share a name are no longer dropped (they used to load as
    ``error_unknown``).
  - 7-digit (.NET) fractional seconds are parsed (truncated to microseconds).
- NSRR: R&K stages 3 and 4 both map to ``N3``. Before, any file containing Stage 3 raised
  ``KeyError`` (the two-way dictionary lost the Stage 3 entry). Custom many-to-one mappings work.

Other
'''''

- ``vector.get_mutual_vectors(x)`` without labels raised ``UnboundLocalError``.
- ``vector.translate`` returns a new array and promotes integer input to float. It used to work
  in place and truncate.
- ``vector.rotate``: the direction convention is documented (positive = clockwise, about the
  points' mean).
- ``types.ObjDict``: attribute auto-creation works as in v2.0.0 (``d.cfg.sub = 3`` creates
  ``cfg``; reading a missing public attribute creates it, so ``hasattr`` is always true for public
  names). Names starting with ``_`` (dunders included) are no longer auto-created: this fixes
  ``copy.deepcopy``, ``copy.copy`` and ``pickle`` of an ``ObjDict`` (they failed with
  ``'ObjDict' object is not callable``) and stops protocol probes such as ``__array__`` or
  ``_repr_html_`` from adding keys.
- ``files.get_files`` only skips AppleDouble files (basename starting with ``._``), not every
  path that contains ``._``.
- Packaging:

  - ``tqdm`` is a runtime dependency. It was imported but only listed under the test extra, so
    a clean install failed to import ``annotations``.
  - ``pytz`` was removed (unused).
  - Lower bounds: ``numpy>=1.24``, ``scipy>=1.10``, ``pandas>=2.0``.
  - The stale committed ``egg-info`` was removed.
