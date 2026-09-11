"""
Phase of peak firing per electrode, and the theta metrics that go with it.

The question this answers: for each electrode, if we take EVERY spike in the session (all
good units pooled together, not one unit at a time) and look up the theta phase *of that
electrode* at each spike time, what phase do spikes pile up at?

That single number per electrode, plotted on the probe geometry, is a map of where the
theta phase reference sits relative to the cell layer. In CA1 the pyramidal layer flips
firing phase by ~180 degrees across the layer, so this map is a way to find the layer
without having to eyeball raw traces.

PHASE CONVENTION: phases here are PEAK-REFERENCED -- 0/360 degrees is the theta PEAK and
180 is the TROUGH -- matching the hippocampal literature (Mizuseki et al. 2009 Neuron
64:267-280 and 2011 Nat Neurosci 14:1174-1181, both "peak of theta = 0, 360 deg, trough =
180 deg"), in which CA1 pyramidal cells fire near 180.

HexMazeThetaV1 stores only the analytic signal z, and phase comes from np.mod(np.angle(z),
2*pi), which is 0 at the LFP peak -- so the convention is a property of the derivation and
there is nothing to rotate. Expect pooled spikes on a pyramidal-layer electrode to peak
near 180.

Note the spikes are the SAME for every electrode (all units pooled). What changes from
electrode to electrode is the theta phase used to look them up. So the map shows how theta
phase itself varies over the probe, measured through the spikes.

Everything here works for both labs:
    Berke lab (IM-*) sessions   -> spikesorting v1, one run epoch ("00_r1")
    Frank lab sessions          -> spikesorting v0, 4-5 run epochs ("01_r1", "03_r2", ...)
The spikesorting_helpers module hides that split, and we take the FIRST run epoch of each
session so every session contributes exactly one figure.

The two labs also need different probe layouts, which `probe_layout` sorts out. Berke
sessions use one probe model whose rel_x / rel_y are distinct for every electrode in the
session, so they are drawn true to scale. Frank sessions are implanted with 3-4 separate
probes, and rel_x / rel_y are coordinates WITHIN a probe -- four identical 128-channel
probes put 512 electrodes on 128 positions -- so those sessions are drawn as one column per
(probe, shank), with true depth down each column.

Reading the theta tables is the slow part. One epoch of a 512 channel session is ~17 GB of
theta phase on /stelmo, so the per-electrode reads are farmed out to a process pool
(`n_workers`), and each worker reduces its columns down to a handful of numbers instead of
handing whole timeseries back to the parent.

Typical usage:
    from spyglass_hexmaze.spike_theta_phase import (
        session_jobs, compute_session_metrics, plot_session_metrics,
    )

    jobs = session_jobs()                      # one (session, first run epoch) per session
    metrics, info = compute_session_metrics(**jobs[0])
    plot_session_metrics(metrics, info)
"""

import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import spyglass.common as sgc
from spyglass.common import AnalysisNwbfile

from spyglass_hexmaze.hex_maze_theta import HexMazeThetaV1, HexMazeThetaReference
from spyglass_hexmaze.hex_maze_decoding import HexMazeDecodedPosition
from spyglass_hexmaze.spikesorting_helpers import (
    fetch_good_units,
    electrodes_with_units,
    fetch_electrode_geometry,
)

# Defaults shared by the computation and the plots. These match the values already used in
# the Select_Theta_Reference_Electrodes notebook so the panels are comparable to that figure.
SPEED_THRESH = 5.0        # cm/s, "running" for the running-only spike phase and decode metrics
SPATIAL_COV_MAX = 200.0   # only keep confident decodes for the decode-distance metrics
SUBSAMPLE = 50            # keep every Nth sample for the correlation-style metrics
N_PHASE_BINS = 24         # 15 degree bins for the spike phase histograms

# Where per-session results are cached. CSVs are gitignored, so this is safe to leave around.
CACHE_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "spike_theta_phase_cache"


## Picking which (session, epoch) to run

def first_run_interval(nwb_file_name, theta_intervals=None):
    """The first run epoch of a session that has theta computed for it.

    Berke lab sessions only have one run epoch ("00_r1"). Frank lab sessions have 4-5
    ("01_r1", "03_r2", ...) and we just take the first, so every session contributes one
    figure. Intervals with a "noPreTrialTimes" suffix are alternate versions of an epoch we
    already have, so they are ignored.

    Parameters:
        nwb_file_name (str): NWB file to look up (e.g. "Toby20250319_.nwb")
        theta_intervals (list): Optional pre-fetched interval names for this session, to
            avoid re-querying HexMazeThetaV1 inside a loop. Default None = query.

    Returns:
        str: The first run interval name, or None if the session has no theta entries.
    """
    if theta_intervals is None:
        theta_intervals = (HexMazeThetaV1 & {"nwb_file_name": nwb_file_name}).fetch(
            "target_interval_list_name")
    intervals = sorted(iv for iv in theta_intervals if "noPreTrial" not in iv)
    return intervals[0] if intervals else None


def session_jobs():
    """Every session with theta computed, paired with its first run epoch.

    Returns:
        list[dict]: One dict per session with keys ``nwb_file_name`` and
            ``target_interval_list_name``, sorted by session name.
    """
    entries = pd.DataFrame(
        HexMazeThetaV1().fetch("nwb_file_name", "target_interval_list_name", as_dict=True))
    jobs = []
    for nwb_file_name, group in entries.groupby("nwb_file_name"):
        interval = first_run_interval(nwb_file_name, group["target_interval_list_name"].tolist())
        if interval is not None:
            jobs.append({"nwb_file_name": nwb_file_name,
                         "target_interval_list_name": interval})
    return sorted(jobs, key=lambda j: j["nwb_file_name"])


def session_id(nwb_file_name, target_interval_list_name):
    """Short label for a (session, epoch), used for figure folders and cache file names."""
    return f"{nwb_file_name.replace('_.nwb', '')}_{target_interval_list_name}"


## Circular statistics

def circular_stats(resultant, n):
    """Mean angle, resultant length and Rayleigh p from a summed unit vector.

    Parameters:
        resultant (complex): Sum of exp(i * phase) over all observations (spikes). Weighting
            by spike count is fine, in which case `n` is the total spike count.
        n (int): Number of observations that went into `resultant`.

    Returns:
        tuple: (mean_angle_deg, R, rayleigh_p) where mean_angle_deg is in [0, 360),
            R is the mean resultant length in [0, 1] (1 = perfectly locked), and
            rayleigh_p is the Rayleigh test p value (Zar's approximation).
            All NaN / 1.0 if there were no observations.
    """
    if n <= 0:
        return np.nan, np.nan, 1.0
    R = np.abs(resultant) / n
    mean_angle_deg = np.degrees(np.angle(resultant)) % 360
    # Rayleigh test for non-uniformity (Zar, Biostatistical Analysis)
    Z = n * R ** 2
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n))
    return mean_angle_deg, R, float(np.clip(p, 0, 1))


def circular_spread(phases_deg):
    """Circular standard deviation of a set of angles, in degrees.

    Used to say how much the phase of peak firing varies across a probe: a small spread
    means every electrode sees theta at about the same phase, a large one means theta
    rotates a lot over the probe (which is what happens when we cross the cell layer).

    Parameters:
        phases_deg (np.ndarray): Angles in degrees.

    Returns:
        float: Circular SD in degrees, sqrt(-2 ln R). NaN if there are no angles.
    """
    phases_deg = np.asarray(phases_deg, dtype=float)
    phases_deg = phases_deg[np.isfinite(phases_deg)]
    if phases_deg.size == 0:
        return np.nan
    R = np.abs(np.mean(np.exp(1j * np.radians(phases_deg))))
    R = np.clip(R, 1e-12, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def circular_linear_corr(cos_phase, sin_phase, linear):
    """Mardia's circular-linear correlation between a phase and a linear variable.

    Combines the correlation of the linear variable with cos(phase) and with sin(phase) into
    one number in [0, 1] (0 = no relationship to phase, 1 = phase fully predicts it).

    Parameters:
        cos_phase (np.ndarray): cos of the phase, shape (n_samples,) or (n_samples, n_elec)
        sin_phase (np.ndarray): sin of the phase, same shape as cos_phase
        linear (np.ndarray): The linear variable, shape (n_samples,)

    Returns:
        tuple: (corr, preferred_phase_deg), each a scalar or one value per column.
            preferred_phase_deg is where the linear variable peaks, in [-180, 180].
    """
    rxc = _columnwise_pearson(cos_phase, linear)
    rxs = _columnwise_pearson(sin_phase, linear)
    rcs = _columnwise_pearson(cos_phase, sin_phase, paired=True)
    corr = np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))
    # First-harmonic fit linear ~ a*cos(phase) + b*sin(phase) peaks at atan2(b, a)
    preferred_phase_deg = np.degrees(np.arctan2(rxs, rxc))
    return corr, preferred_phase_deg


def _columnwise_pearson(matrix, other, paired=False):
    """Pearson correlation of each column of `matrix` against `other`.

    Parameters:
        matrix (np.ndarray): Shape (n_samples, n_columns)
        other (np.ndarray): Shape (n_samples,) to correlate every column against, or the
            same shape as `matrix` when `paired` is True (column i vs column i).
        paired (bool): Correlate matching columns instead of every column against a vector.

    Returns:
        np.ndarray: One correlation per column of `matrix`.
    """
    matrix_centered = matrix - matrix.mean(0)
    if paired:
        other_centered = other - other.mean(0)
        return (matrix_centered * other_centered).sum(0) / np.sqrt(
            (matrix_centered ** 2).sum(0) * (other_centered ** 2).sum(0))
    other_centered = other - other.mean()
    return (matrix_centered * other_centered[:, None]).sum(0) / np.sqrt(
        (matrix_centered ** 2).sum(0) * (other_centered ** 2).sum())


## Reading theta columns in parallel
#
# The stored theta tables are one HDF5 dataset per electrode, so we can read just the
# electrodes we want. But a whole epoch is still ~33 MB per electrode and there are up to
# 512 of them, so the reads are spread over a process pool. Each worker reduces its columns
# to a few numbers plus a subsampled copy, which is what comes back to the parent.
#
# The worker globals below are set in the parent before the Pool is created, so they are
# inherited by fork (no pickling of big arrays).

_WORKER = {}


def _find_stored_group(h5_file, object_id):
    """Find the HDF5 group holding a stored DataFrame, by the object id we recorded for it."""
    found = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and obj.attrs.get("object_id") == object_id:
            found["group"] = obj

    h5_file.visititems(visit)
    if "group" not in found:
        raise ValueError(f"No object {object_id} in {h5_file.filename}")
    return found["group"]


def _theta_file_and_object(theta_entry):
    """Absolute path + object id of the stored analytic signal.

    The table stores only the analytic signal; phase and power are derived from it here,
    the same way HexMazeThetaV1.fetch_theta_phase / fetch_theta_power do.

    Parameters:
        theta_entry (HexMazeThetaV1): Restriction matching exactly one entry

    Returns:
        tuple: (path, object_id)
    """
    analysis_file_name, object_id = theta_entry.fetch1(
        "analysis_file_name", "analytic_signal_object_id")
    return AnalysisNwbfile.get_abs_path(analysis_file_name), object_id


def _theta_worker(columns):
    """Reduce a batch of electrodes to everything the metrics need, in ONE pass.

    Reads each electrode's stored real and imaginary columns, rebuilds the complex analytic
    signal, and derives phase and power from it -- so the file is read once per electrode
    rather than once for phase and again for power.

    Phase is np.mod(np.angle(z), 2*pi), i.e. peak-referenced: 0/2pi is the LFP peak and pi
    the trough, matching HexMazeThetaV1's accessors and the hippocampal literature.

    For each electrode this computes, against that electrode's own theta:
      - median theta power, and a subsampled power trace
      - the resultant vector of all pooled spikes (sum of exp(i*phase) at spike times)
      - the same for running spikes only
      - the spike count histogram over phase bins (all spikes, and running only)
      - a subsampled phase trace
    Spike times were already binned onto the theta sample grid by the parent, so "look up
    the phase at every spike" is a weighted sum over the whole column here.

    Parameters:
        columns (list[str]): Electrode column stems, e.g. ["electrode 12", ...]

    Returns:
        list[tuple]: (column, median_power, subsampled_power, resultant_all, resultant_run,
            hist_all, hist_run, subsampled_phase) per electrode.
    """
    counts_all = _WORKER["counts_all"]
    counts_run = _WORKER["counts_run"]
    n_bins = _WORKER["n_phase_bins"]
    subsample = _WORKER["subsample"]

    out = []
    with h5py.File(_WORKER["analytic_path"], "r") as f:
        group = _find_stored_group(f, _WORKER["analytic_object"])
        for column in columns:
            z = group[f"{column}_real"][:] + 1j * group[f"{column}_imag"][:]

            power = np.abs(z) ** 2
            phase = np.mod(np.angle(z), 2 * np.pi)

            # Resultant vector = sum of the unit vector at every spike. Because counts_all
            # holds the number of spikes in each theta sample, this dot product is exactly
            # "add up exp(i*phase) once per spike", just done in one pass. The unit vector
            # is z/|z|, which is exp(i*phase) without recomputing any trig.
            unit_vector = z / np.abs(z)
            resultant_all = complex(np.dot(counts_all, unit_vector))
            resultant_run = (complex(np.dot(counts_run, unit_vector))
                             if counts_run is not None else complex(np.nan, np.nan))

            # Spike count per phase bin, for the "phase of peak firing" histogram
            bin_index = np.minimum((phase * (n_bins / (2 * np.pi))).astype(np.int64), n_bins - 1)
            hist_all = np.bincount(bin_index, weights=counts_all, minlength=n_bins)
            hist_run = (np.bincount(bin_index, weights=counts_run, minlength=n_bins)
                        if counts_run is not None else np.full(n_bins, np.nan))

            out.append((column,
                        float(np.median(power)),
                        power[::subsample].astype(np.float32),
                        resultant_all, resultant_run, hist_all, hist_run,
                        phase[::subsample].astype(np.float32)))
    return out


def _run_pool(worker, columns, n_workers):
    """Run `worker` over `columns` in a process pool and concatenate the results.

    Parameters:
        worker (callable): _theta_worker
        columns (list[str]): All column names to process
        n_workers (int): Number of worker processes. 1 runs in-process (easier to debug).

    Returns:
        list: The per-column tuples the worker returned, in no particular order.
    """
    # Deal the columns out round-robin so every worker gets a similar amount of reading
    batches = [columns[i::n_workers] for i in range(n_workers)]
    batches = [b for b in batches if b]
    if n_workers == 1:
        return worker(batches[0])
    with Pool(n_workers) as pool:
        return [item for batch in pool.map(worker, batches) for item in batch]


## Spikes and speed

def pooled_spike_times(nwb_file_name, curation_id=None, verbose=True):
    """Every spike from every good unit in a session, pooled into one sorted array.

    "All spikes" here means all spikes from all well-isolated units (units labeled
    noise / reject / mua are already dropped by fetch_good_units). Units are pooled rather
    than treated separately because we want one phase per electrode, not one per unit.

    Parameters:
        nwb_file_name (str): NWB file to fetch units for
        curation_id (int): Passed through to fetch_good_units. Default None = its default.
        verbose (bool): Print how many units / spikes were found. Default True.

    Returns:
        tuple: (spike_times, n_units, unit_electrodes) where spike_times is a sorted float
            array of spike times in seconds on the session clock, n_units is how many units
            contributed, and unit_electrodes is the list of electrodes carrying a unit
            (empty for Frank lab v0 sessions, which have no peak channel).
    """
    units = fetch_good_units(nwb_file_name, curation_id=curation_id, verbose=verbose)
    if len(units) == 0:
        return np.empty(0), 0, []
    spike_times = np.sort(np.concatenate(
        [np.asarray(st, dtype=float) for st in units["spike_times"]]))
    if verbose:
        print(f"  pooled {len(spike_times)} spikes from {len(units)} good units")
    return spike_times, len(units), electrodes_with_units(units)


def fetch_decode_dataframe(nwb_file_name, target_interval_list_name):
    """Decoded position for the epoch matching a theta interval, if it exists.

    Theta is keyed on the interval list name but decoding is keyed on the epoch NUMBER, so
    the epoch is looked up from TaskEpoch rather than assumed -- restricting by
    nwb_file_name alone can silently land the two on different epochs.

    Parameters:
        nwb_file_name (str): NWB file to look up
        target_interval_list_name (str): Run interval used for theta, e.g. "01_r1"

    Returns:
        pd.DataFrame: Time-indexed decode dataframe (speed, decode_distance, spatial_cov,
            ...), or None if this session/epoch has no decoding populated.
    """
    epoch = (sgc.TaskEpoch & {"nwb_file_name": nwb_file_name,
                              "interval_list_name": target_interval_list_name}).fetch("epoch")
    if len(epoch) == 0:
        return None
    decode = HexMazeDecodedPosition & {"nwb_file_name": nwb_file_name, "epoch": int(epoch[0])}
    if len(decode) == 0:
        return None
    # A session can have more than one decoding parameter set; take the first one so this
    # never errors out mid-batch
    first_key = decode.fetch("KEY")[0]
    return (HexMazeDecodedPosition & first_key).fetch1_dataframe()


## The main computation

def compute_session_metrics(
    nwb_file_name,
    target_interval_list_name,
    n_workers=6,
    subsample=SUBSAMPLE,
    speed_thresh=SPEED_THRESH,
    spatial_cov_max=SPATIAL_COV_MAX,
    n_phase_bins=N_PHASE_BINS,
    curation_id=None,
    cache_dir=CACHE_DIR,
    use_cache=True,
    verbose=True,
):
    """Per-electrode spike phase + theta metrics for one (session, run epoch).

    The expensive part is reading the stored theta phase and theta power tables off
    /stelmo, so results are cached to `cache_dir` and reused unless `use_cache=False`.

    Parameters:
        nwb_file_name (str): NWB file, e.g. "Toby20250319_.nwb"
        target_interval_list_name (str): Run interval used for theta, e.g. "01_r1"
        n_workers (int): Processes used to read theta columns. Default 6.
        subsample (int): Keep every Nth sample for the correlation-style metrics. Default 50.
        speed_thresh (float): cm/s above which the rat counts as running. Default 5.
        spatial_cov_max (float): Max spatial coverage for a decode to count as confident.
        n_phase_bins (int): Phase bins for the spike phase histograms. Default 24 (15 deg).
        curation_id (int): Passed through to fetch_good_units. Default None = its default.
        cache_dir (Path): Where to read/write cached results.
        use_cache (bool): Reuse a cached result if one exists. Default True.
        verbose (bool): Print progress. Default True.

    Returns:
        tuple: (metrics, info).
            metrics (pd.DataFrame): One row per electrode, indexed by electrode name (str),
                with probe geometry (rel_x, rel_y, is_bad) and these columns:
                    peak_firing_phase_deg: circular mean phase of all pooled spikes, in the
                        electrode's own theta. THE metric this module is about. Peak-
                        referenced (0/360 = theta peak, 180 = trough), so a pyramidal-layer
                        electrode should land near 180. This is rotated by 180 deg from the
                        raw stored phase -- see the module docstring.
                    peak_firing_R / peak_firing_p: locking strength and Rayleigh p
                    peak_firing_hist_phase_deg: center of the phase bin with the most spikes
                        (the literal "phase of peak firing", quantized to the bin width)
                    peak_firing_phase_deg_run / _R_run / _p_run / hist_phase_deg_run: the
                        same four, using running spikes only (NaN if no speed available)
                    theta_power: median theta power
                    phase_offset_deg: mean theta phase offset from the reference electrode
                    power_corr: correlation of the theta power envelope with the reference
                    power_speed_corr: correlation of theta power with running speed
                    dist_phase_corr: circular-linear corr of decode distance with theta phase
                    dist_pref_phase_deg: theta phase where decode distance is most positive
            info (dict): Session-level context for the plots -- session_id, reference
                electrode, spike/unit counts, phase histograms, whether speed and decode
                were available, and the saved theta reference electrodes if any.
    """
    label = session_id(nwb_file_name, target_interval_list_name)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = cache_dir / f"{label}_metrics.csv"
    info_path = cache_dir / f"{label}_info.npz"

    if use_cache and metrics_path.exists() and info_path.exists():
        if verbose:
            print(f"{label}: loading cached results from {metrics_path.name}")
        metrics = pd.read_csv(metrics_path, index_col=0)
        metrics.index = metrics.index.astype(str)
        with np.load(info_path, allow_pickle=True) as stored:
            info = {k: stored[k] for k in stored.files}
        info = {k: (v.item() if getattr(v, "ndim", 1) == 0 else v) for k, v in info.items()}
        return metrics, info

    started = time.time()
    theta_key = {"nwb_file_name": nwb_file_name,
                 "target_interval_list_name": target_interval_list_name}
    theta_entry = HexMazeThetaV1 & theta_key
    if len(theta_entry) != 1:
        raise ValueError(f"{label}: expected 1 HexMazeThetaV1 entry, found {len(theta_entry)}")

    if verbose:
        print(f"\n=== {label} ===")

    # ---- theta sample times, and which electrodes were actually stored --------------------
    # The electrode list comes from the stored table itself rather than from
    # LFPBandSelection, because a session can have several LFP band selections that all
    # match this restriction and those would give us duplicate electrode ids.
    analytic_path, analytic_object = _theta_file_and_object(theta_entry)
    with h5py.File(analytic_path, "r") as f:
        analytic_group = _find_stored_group(f, analytic_object)
        theta_time = analytic_group["time"][:]
        # Stored columns are "electrode N_real" / "electrode N_imag"; take the stems
        electrode_ids = sorted(int(c.split()[-1].rsplit("_", 1)[0])
                               for c in analytic_group if c.endswith("_real"))

    # ---- geometry ------------------------------------------------------------------------
    geometry = fetch_electrode_geometry(nwb_file_name)
    # Electrode names are strings in the geometry table and ints in the theta tables
    theta_names = [str(e) for e in electrode_ids]
    columns = [f"electrode {e}" for e in electrode_ids]
    if verbose:
        print(f"  {len(electrode_ids)} electrodes with theta")

    # ---- pooled spikes -------------------------------------------------------------------
    spike_times, n_units, unit_electrodes = pooled_spike_times(
        nwb_file_name, curation_id=curation_id, verbose=verbose)

    if verbose:
        print(f"  theta: {len(theta_time)} samples, "
              f"{(theta_time[-1] - theta_time[0]) / 60:.1f} min")

    # ---- speed, from the decode dataframe (the only place we have it for these sessions) --
    decode_df = fetch_decode_dataframe(nwb_file_name, target_interval_list_name)
    has_decode = decode_df is not None
    speed_at_theta = None
    if has_decode:
        speed_at_theta = np.asarray(
            decode_df["speed"].reindex(pd.Index(theta_time), method="nearest").values, dtype=float)
    if verbose:
        print(f"  decode: {'yes' if has_decode else 'NO (no speed / decode panels)'}")

    # ---- bin spikes onto the theta sample grid ---------------------------------------------
    # Every spike is assigned to its nearest theta sample, then counted. Doing it this way
    # turns "phase at each spike time" into a weighted sum over the column, which is what
    # lets a worker reduce a 33 MB column without ever materializing per-spike phases.
    counts_all = _bin_spikes(spike_times, theta_time)
    n_spikes_all = int(counts_all.sum())
    if speed_at_theta is not None:
        running = speed_at_theta >= speed_thresh
        counts_run = np.where(running, counts_all, 0.0)
        n_spikes_run = int(counts_run.sum())
    else:
        counts_run, n_spikes_run = None, 0
    if verbose:
        print(f"  spikes inside this epoch: {n_spikes_all}"
              + (f" ({n_spikes_run} while running)" if counts_run is not None else ""))

    # ---- one pass over the analytic signal, giving both power and phase -------------------
    _WORKER.clear()
    _WORKER.update(analytic_path=analytic_path, analytic_object=analytic_object,
                   subsample=subsample, counts_all=counts_all, counts_run=counts_run,
                   n_phase_bins=n_phase_bins)
    t0 = time.time()
    results = _run_pool(_theta_worker, columns, n_workers)
    if verbose:
        print(f"  read theta (power + phase) in {time.time() - t0:.0f}s")

    key_of = lambda c: c.split()[-1]
    median_power = pd.Series({key_of(r[0]): r[1] for r in results})
    power_sub = pd.DataFrame({key_of(r[0]): r[2] for r in results})[theta_names]
    resultant_all = {key_of(r[0]): r[3] for r in results}
    resultant_run = {key_of(r[0]): r[4] for r in results}
    hist_all = {key_of(r[0]): r[5] for r in results}
    hist_run = {key_of(r[0]): r[6] for r in results}
    phase_sub = pd.DataFrame({key_of(r[0]): r[7] for r in results})[theta_names]

    # Reference electrode = the good electrode with the strongest theta, same rule the
    # Select_Theta_Reference_Electrodes notebook uses to pick its automatic reference
    is_bad = geometry["is_bad"].reindex(theta_names).fillna(False).values
    good_names = [n for n, bad in zip(theta_names, is_bad) if not bad]
    reference_name = median_power[good_names].idxmax() if good_names else theta_names[0]

    # ---- per-electrode metrics --------------------------------------------------------------
    bin_centers = np.degrees((np.arange(n_phase_bins) + 0.5) * 2 * np.pi / n_phase_bins)

    rows = {}
    for name in theta_names:
        mean_deg, R, p = circular_stats(resultant_all[name], n_spikes_all)
        row = {
            "peak_firing_phase_deg": mean_deg,
            "peak_firing_R": R,
            "peak_firing_p": p,
            "peak_firing_hist_phase_deg": (bin_centers[int(np.argmax(hist_all[name]))]
                                           if n_spikes_all > 0 else np.nan),
        }
        if counts_run is not None:
            mean_deg_r, R_r, p_r = circular_stats(resultant_run[name], n_spikes_run)
            row.update({
                "peak_firing_phase_deg_run": mean_deg_r,
                "peak_firing_R_run": R_r,
                "peak_firing_p_run": p_r,
                "peak_firing_hist_phase_deg_run": (bin_centers[int(np.argmax(hist_run[name]))]
                                                   if n_spikes_run > 0 else np.nan),
            })
        rows[name] = row
    metrics = pd.DataFrame.from_dict(rows, orient="index")

    # Theta power, phase offset from the reference, and power correlation with the reference
    metrics["theta_power"] = median_power.reindex(metrics.index)
    reference_phase = phase_sub[reference_name].values.astype(float)
    offsets = np.degrees(np.angle(np.mean(
        np.exp(1j * (phase_sub.values.astype(float) - reference_phase[:, None])), axis=0)))
    metrics["phase_offset_deg"] = pd.Series(offsets, index=phase_sub.columns)
    metrics["power_corr"] = power_sub.corrwith(power_sub[reference_name])

    # Speed / decode based metrics, when this session has decoding populated
    metrics["power_speed_corr"] = np.nan
    metrics["dist_phase_corr"] = np.nan
    metrics["dist_pref_phase_deg"] = np.nan
    if has_decode:
        speed_sub = speed_at_theta[::subsample][: len(power_sub)]
        # Correlate over the samples where speed is actually known. The decode dataframe can
        # carry a few hundred NaN speed samples out of a million, and Pearson over a vector
        # holding even one NaN comes back NaN for EVERY electrode -- which blanks the whole
        # panel and reads as "no relationship to speed" rather than "not computed".
        finite_speed = np.isfinite(speed_sub)
        if finite_speed.sum() > 100:
            metrics["power_speed_corr"] = pd.Series(
                _columnwise_pearson(power_sub.values.astype(float)[finite_speed],
                                    speed_sub[finite_speed]),
                index=power_sub.columns)

        decode_distance = np.asarray(decode_df["decode_distance"].reindex(
            pd.Index(theta_time), method="nearest").values, dtype=float)[::subsample][: len(phase_sub)]
        spatial_cov = np.asarray(decode_df["spatial_cov"].reindex(
            pd.Index(theta_time), method="nearest").values, dtype=float)[::subsample][: len(phase_sub)]
        keep = ((speed_sub >= speed_thresh) & (spatial_cov < spatial_cov_max)
                & np.isfinite(decode_distance))
        if keep.sum() > 100:
            kept_phase = phase_sub.values[keep].astype(float)
            corr, preferred = circular_linear_corr(
                np.cos(kept_phase), np.sin(kept_phase), decode_distance[keep])
            metrics["dist_phase_corr"] = pd.Series(corr, index=phase_sub.columns)
            metrics["dist_pref_phase_deg"] = pd.Series(preferred, index=phase_sub.columns)

    # Attach geometry, and blank out bad channels the same way the existing figure does
    metrics = metrics.join(geometry[["rel_x", "rel_y", "is_bad"]])
    metrics["is_bad"] = metrics["is_bad"].fillna(False)
    metric_columns = [c for c in metrics.columns if c not in ("rel_x", "rel_y", "is_bad")]
    metrics.loc[metrics["is_bad"], metric_columns] = np.nan

    # Saved reference electrode set for this epoch, if one was picked by hand
    saved = HexMazeThetaReference & theta_key
    saved_reference = sorted({int(e) for ids in saved.fetch("electrode_ids") for e in ids})

    info = {
        "session_id": label,
        "nwb_file_name": nwb_file_name,
        "target_interval_list_name": target_interval_list_name,
        "reference_name": reference_name,
        "saved_reference_electrodes": np.array(saved_reference, dtype=int),
        "unit_electrodes": np.array(unit_electrodes, dtype=int),
        "n_units": n_units,
        "n_spikes_all": n_spikes_all,
        "n_spikes_run": n_spikes_run,
        "has_decode": bool(has_decode),
        "epoch_minutes": float((theta_time[-1] - theta_time[0]) / 60),
        "bin_centers_deg": bin_centers,
        "hist_all": np.array([hist_all[n] for n in theta_names]),
        "hist_run": np.array([hist_run[n] for n in theta_names]),
        "electrode_names": np.array(theta_names),
        "speed_thresh": speed_thresh,
        "n_phase_bins": n_phase_bins,
    }

    metrics.to_csv(metrics_path)
    np.savez_compressed(info_path, **info)
    if verbose:
        print(f"  done in {time.time() - started:.0f}s -> {metrics_path.name}")

    return metrics, info


def _bin_spikes(spike_times, theta_time):
    """Count spikes into their nearest theta sample.

    Parameters:
        spike_times (np.ndarray): Sorted spike times in seconds (whole session)
        theta_time (np.ndarray): Theta sample times in seconds (this epoch only)

    Returns:
        np.ndarray: float64 array the same length as theta_time, holding the number of
            spikes assigned to each theta sample. Spikes outside the epoch are dropped.
    """
    counts = np.zeros(len(theta_time), dtype=float)
    if len(spike_times) == 0:
        return counts
    inside = spike_times[(spike_times >= theta_time[0]) & (spike_times <= theta_time[-1])]
    if len(inside) == 0:
        return counts
    # searchsorted gives the sample just after each spike; pick whichever neighbour is closer
    right = np.searchsorted(theta_time, inside)
    right = np.clip(right, 1, len(theta_time) - 1)
    left = right - 1
    nearest = np.where(inside - theta_time[left] <= theta_time[right] - inside, left, right)
    np.add.at(counts, nearest, 1.0)
    return counts


## Plotting -- same probe-geometry style as the Select_Theta_Reference_Electrodes notebook

def _min_spacing(values):
    """Smallest nonzero gap between sorted unique values, used to size the electrode boxes."""
    values = values[np.isfinite(values)]
    gaps = np.diff(np.unique(values))
    gaps = gaps[gaps > 0]
    return gaps.min() if len(gaps) else 1.0


def probe_layout(metrics, nwb_file_name):
    """Work out where to draw each electrode, and how big the figure has to be.

    There are two cases, and which one a session falls into is decided from the data rather
    than from the lab:

    1. Unique positions (Berke lab). rel_x / rel_y already place every electrode of the
       session distinctly, so they are drawn as-is, true to scale, on one wide flat panel.

    2. Colliding positions (Frank lab). rel_x / rel_y are coordinates within ONE probe (see
       fetch_electrode_geometry), so a session with 4 identical 128-channel probes puts its
       512 electrodes on 128 positions and all but one electrode per position is hidden.
       These sessions are laid out as one column per (probe, shank) instead: x counts shank
       columns, y stays the true depth along the shank. The spacing BETWEEN columns is
       therefore schematic -- we have no coordinates that would place the probes relative to
       each other -- while depth within a shank stays real, which is the axis that matters
       for finding the cell layer.

    Parameters:
        metrics (pd.DataFrame): Output of compute_session_metrics, indexed by electrode name
        nwb_file_name (str): Session the metrics came from, used to look up which probe and
            shank each electrode sits on. Read at plot time rather than stored in the cached
            metrics, so old cache files keep working.

    Returns:
        tuple: (plot_metrics, layout).
            plot_metrics (pd.DataFrame): Copy of `metrics` whose rel_x / rel_y are the
                coordinates to draw at (unchanged in case 1, re-laid-out in case 2).
            layout (dict): How to set the panels up -- box_width, box_height (scalar or one
                value per electrode), aspect, ylabel, xlabel, xticks, xticklabels,
                panel_width and panel_height in inches, and n_columns.
    """
    plot_metrics = metrics.copy()

    if not plot_metrics[["rel_x", "rel_y"]].duplicated().any():
        # True-to-scale geometry: keep the wide flat panel the Berke figures already use
        return plot_metrics, {
            "box_width": _min_spacing(plot_metrics["rel_x"].values),
            "box_height": _min_spacing(plot_metrics["rel_y"].values),
            "aspect": "true",
            "ylabel": "rel_y (µm)",
            "xlabel": "rel_x (µm)",
            "xticks": None,
            "xticklabels": None,
            "panel_width": 16.0,
            "panel_height": 3.0,
            "n_columns": len(np.unique(plot_metrics["rel_x"].values)),
        }

    # ---- shank-column layout -------------------------------------------------------------
    geometry = fetch_electrode_geometry(nwb_file_name)
    shanks = geometry[["electrode_group_name", "probe_shank"]].reindex(plot_metrics.index)

    # One column per (probe, shank), ordered by probe then shank. Group names are strings but
    # number-like, so sort them numerically when we can to keep probes in their natural order.
    def sort_key(pair):
        group, shank = pair
        try:
            return (0, float(group), float(shank))
        except (TypeError, ValueError):
            return (1, str(group), float(shank))

    pairs = sorted({tuple(p) for p in shanks.dropna().itertuples(index=False)}, key=sort_key)
    column_of = {pair: i for i, pair in enumerate(pairs)}
    columns = np.array([column_of.get(tuple(p), np.nan) for p in shanks.itertuples(index=False)],
                       dtype=float)

    # Site pitch down each column, per electrode: probes of different models have different
    # pitches (e.g. 40 µm vs 26 µm), and a single global box height would make boxes on the
    # finer probe overlap while leaving gaps on the coarser one.
    depth = plot_metrics["rel_y"].values.astype(float)
    heights = np.full(len(plot_metrics), np.nan)
    for pair, column in column_of.items():
        in_column = columns == column
        # A column with only one electrode left in it has no spacing to measure, so leave it
        # NaN and let the median pitch below fill it in
        if in_column.sum() >= 2:
            heights[in_column] = _min_spacing(depth[in_column])
    # Anything with no measurable pitch (no shank recorded, or a column down to one
    # electrode) falls back to the median pitch, or to a slice of the depth range if no
    # column had a pitch at all -- either way the box stays big enough to see
    depth_span = np.ptp(depth[np.isfinite(depth)]) if np.isfinite(depth).any() else 1.0
    fallback = (np.nanmedian(heights) if np.isfinite(heights).any()
                else max(depth_span / 32.0, 1.0))
    heights[~np.isfinite(heights)] = fallback

    plot_metrics["rel_x"] = columns
    plot_metrics["rel_y"] = depth

    return plot_metrics, {
        "box_width": 0.8,                       # in column units, so columns stay separated
        "box_height": heights,
        "aspect": None,                         # x counts shanks; let the axes fill the panel
        "ylabel": "depth along shank (µm)",
        "xlabel": "probe / shank",
        "xticks": list(range(len(pairs))),
        "xticklabels": [f"p{group}s{int(shank)}" for group, shank in pairs],
        # Wide enough that 32 electrode numbers per column stay legible, and tall enough that
        # a whole shank of sites is readable rather than a hairline
        "panel_width": max(9.0, 0.75 * len(pairs) + 2.5),
        "panel_height": 5.0,
        "n_columns": len(pairs),
    }


def geometry_boxes(ax, metrics, values, cmap_name, label, clim=None, reference_name=None,
                   unit_electrodes=(), highlight=None, box_width=None, box_height=None,
                   label_numbers=True, aspect="true", ylabel="rel_y (µm)",
                   xticks=None, xticklabels=None):
    """Draw the probe as one box per electrode, colored by `values`.

    Bad channels come out white (their values are NaN). Electrodes carrying a good unit get
    a star, the automatic reference electrode is outlined in yellow, and anything in
    `highlight` (e.g. a saved reference set) is outlined in red.

    Parameters:
        ax (matplotlib.axes.Axes): Axes to draw on
        metrics (pd.DataFrame): Output of compute_session_metrics, or the re-laid-out copy
            probe_layout returns. rel_x / rel_y are read as the position to draw each box at.
        values (np.ndarray): One value per row of `metrics`, in the same order
        cmap_name (str): Matplotlib colormap name
        label (str): Colorbar label
        clim (tuple): (vmin, vmax) for the colormap. Default None = autoscale.
        reference_name (str): Electrode to outline in yellow. Default None = none.
        unit_electrodes (list[int]): Electrodes carrying a good unit, starred.
        highlight (list[int]): Electrodes to outline in red.
        box_width (float or array): Box width, scalar or one per row of `metrics`. Default
            None = smallest rel_x spacing.
        box_height (float or array): Box height, scalar or one per row. Default None =
            smallest rel_y spacing.
        label_numbers (bool): Print each good electrode's number on its box. Default True.
        aspect (str or float): "true" forces the axes to the probe's own x:y proportions
            (right when rel_x / rel_y are real distances). A number forces that height:width
            ratio. None lets the axes fill its panel, which is what the shank-column layout
            wants -- there the x axis counts shanks and is not a distance.
        ylabel (str): Y axis label. Default "rel_y (µm)".
        xticks (list): X tick positions. Default None = leave matplotlib's ticks.
        xticklabels (list): X tick labels to go with `xticks`.

    Returns:
        None
    """
    if box_width is None:
        box_width = _min_spacing(metrics["rel_x"].values)
    if box_height is None:
        box_height = _min_spacing(metrics["rel_y"].values)
    # Box sizes may be per-electrode (probes of different models in one session have
    # different site pitches), so broadcast whatever we were given up to one value per row
    widths = np.broadcast_to(np.asarray(box_width, dtype=float), (len(metrics),))
    heights = np.broadcast_to(np.asarray(box_height, dtype=float), (len(metrics),))

    cmap = plt.colormaps[cmap_name].copy()
    cmap.set_bad("white")
    boxes = [Rectangle((x - w / 2, y - h / 2), w, h)
             for x, y, w, h in zip(metrics["rel_x"], metrics["rel_y"], widths, heights)]
    collection = PatchCollection(boxes, cmap=cmap)
    collection.set_array(np.ma.masked_invalid(np.asarray(values, dtype=float)))
    if clim:
        collection.set_clim(*clim)
    ax.add_collection(collection)

    # Star electrodes that carry a good unit (empty for Frank lab v0 sessions)
    has_unit = metrics.index.astype(int).isin(set(int(e) for e in unit_electrodes))
    if has_unit.any():
        ax.scatter(metrics.loc[has_unit, "rel_x"], metrics.loc[has_unit, "rel_y"],
                   marker="*", s=70, c="white", edgecolors="black", linewidths=0.6,
                   zorder=5, label="Has good unit(s)")

    # Position -> box size lookup, so the outlines below match the box they sit on even when
    # the session mixes probe models with different site pitches
    size_by_name = {name: (w, h) for name, w, h in zip(metrics.index, widths, heights)}

    # Outline the automatic (max theta power) reference electrode in yellow
    if reference_name is not None and reference_name in metrics.index:
        row = metrics.loc[reference_name]
        w, h = size_by_name[reference_name]
        ax.add_patch(Rectangle((row["rel_x"] - w / 2, row["rel_y"] - h / 2), w, h,
                               fill=False, edgecolor="yellow",
                               linewidth=2, zorder=6, label="Max-power electrode"))

    # Outline a hand-picked reference set in red
    for i, electrode in enumerate(highlight or []):
        name = str(int(electrode))
        if name not in metrics.index:
            continue
        row = metrics.loc[name]
        w, h = size_by_name[name]
        ax.add_patch(Rectangle((row["rel_x"] - w / 2, row["rel_y"] - h / 2), w, h,
                               fill=False, edgecolor="red",
                               linewidth=2, zorder=7,
                               label="Saved reference" if i == 0 else None))

    if label_numbers:
        # Numbers sit up-and-right of center so they clear the unit stars. White text with a
        # black outline stays readable on any box color.
        for name, row in metrics.iterrows():
            if row["is_bad"] or not np.isfinite(row["rel_x"]):
                continue
            w, h = size_by_name[name]
            ax.text(row["rel_x"] + 0.22 * w, row["rel_y"] + 0.22 * h, name,
                    ha="center", va="center", fontsize=4, color="white", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=1.0, foreground="black")], zorder=8)

    pad_x, pad_y = widths.max(), heights.max()
    ax.set_xlim(metrics["rel_x"].min() - pad_x, metrics["rel_x"].max() + pad_x)
    ax.set_ylim(metrics["rel_y"].min() - pad_y, metrics["rel_y"].max() + pad_y)
    if aspect == "true":
        x_span = (metrics["rel_x"].max() - metrics["rel_x"].min()) + 2 * pad_x
        y_span = (metrics["rel_y"].max() - metrics["rel_y"].min()) + 2 * pad_y
        ax.set_box_aspect(y_span / x_span)
    elif aspect is not None:
        ax.set_box_aspect(aspect)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels if xticklabels is not None else xticks,
                           fontsize=7, rotation=90)
    ax.set_ylabel(ylabel)
    plt.colorbar(collection, ax=ax, label=label, fraction=0.02, pad=0.01)


def plot_session_metrics(metrics, info, save_dir=None, label_numbers=True):
    """The theta-metrics probe figure for one session, with phase of peak firing on it.

    Panels, top to bottom:
        1. Phase of peak firing, all spikes (circular mean)  <- the new metric
        2. Phase of peak firing, all spikes (histogram peak bin)
        3. Spike-theta locking strength R, all spikes
        4. Phase of peak firing, running spikes only (skipped if no speed)
        5. Median theta power
        6. Theta phase offset from the max-power electrode
        7. Theta power correlation with the max-power electrode
        8. Corr(theta power, speed)                          (skipped if no speed)
        9. Circ-lin corr(decode distance, theta phase)       (skipped if no decode)
       10. Theta phase where decode distance is most positive (skipped if no decode)

    Parameters:
        metrics (pd.DataFrame): First return value of compute_session_metrics
        info (dict): Second return value of compute_session_metrics
        save_dir (Path): Folder to save a PDF into. Default None = don't save.
        label_numbers (bool): Print electrode numbers on every box. Default True.

    Returns:
        matplotlib.figure.Figure: The figure, so callers can save or tweak it.
    """
    has_run = "peak_firing_phase_deg_run" in metrics.columns
    has_decode = bool(np.asarray(info["has_decode"]).item()) if "has_decode" in info else False

    # (column, colormap, colorbar label, clim, title)
    panels = [
        ("peak_firing_phase_deg", "twilight", "Phase (deg)", (0, 360),
         "PHASE OF PEAK FIRING: all spikes, circular mean (0/360 = theta peak, 180 = trough)"),
        ("peak_firing_hist_phase_deg", "twilight", "Phase (deg)", (0, 360),
         "Phase of peak firing: all spikes, peak histogram bin"),
        ("peak_firing_R", "magma", "R", (0, None),
         "Spike-theta locking strength R (all spikes, pooled over units)"),
    ]
    if has_run:
        panels.append(
            ("peak_firing_phase_deg_run", "twilight", "Phase (deg)", (0, 360),
             f"Phase of peak firing: running spikes only "
             f"(speed > {float(np.asarray(info['speed_thresh'])):.0f} cm/s)"))
    panels += [
        ("theta_power_log10", "viridis", "Theta power (log10)", None,
         "Median theta power"),
        ("phase_offset_deg", "twilight", "Phase offset (deg)", (-180, 180),
         f"Theta phase offset from max-power electrode {info['reference_name']}"),
        ("power_corr", "magma", "Corr with ref power", (0, 1),
         "Theta power correlation with the max-power electrode"),
    ]
    if has_run:
        panels.append(("power_speed_corr", "RdBu_r", "Corr(power, speed)", "symmetric",
                       "Theta power vs. speed correlation"))
    if has_decode and metrics["dist_phase_corr"].notna().any():
        panels += [
            ("dist_phase_corr", "magma", "Circ-lin corr", (0, None),
             "Decode distance vs. theta phase correlation"),
            ("dist_pref_phase_deg", "twilight", "Phase (deg)", (-180, 180),
             "Theta phase where decode distance is most positive"),
        ]

    # Where to draw each electrode, and how big a panel that layout needs. Sessions whose
    # probe coordinates are unique keep the true-to-scale wide panel; sessions implanted with
    # several probes get one column per (probe, shank), because their rel_x / rel_y are
    # per-probe coordinates that would otherwise stack electrodes on top of each other.
    plot_metrics, layout = probe_layout(metrics, str(info["nwb_file_name"]))
    plot_metrics["theta_power_log10"] = np.log10(metrics["theta_power"])

    unit_electrodes = np.asarray(info.get("unit_electrodes", []), dtype=int).tolist()
    highlight = np.asarray(info.get("saved_reference_electrodes", []), dtype=int).tolist()

    fig_height = layout["panel_height"] * len(panels) + 0.6
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(layout["panel_width"], fig_height))
    axes = np.atleast_1d(axes)
    for ax, (column, cmap, cbar_label, clim, title) in zip(axes, panels):
        values = plot_metrics[column].values
        if clim == "symmetric":
            limit = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
            clim = (-limit, limit)
        elif clim is not None and clim[1] is None:
            top = np.nanmax(values) if np.isfinite(values).any() else 1.0
            clim = (clim[0], top)
        geometry_boxes(ax, plot_metrics, values, cmap, cbar_label, clim=clim,
                       reference_name=info["reference_name"],
                       unit_electrodes=unit_electrodes, highlight=highlight,
                       box_width=layout["box_width"], box_height=layout["box_height"],
                       label_numbers=label_numbers, aspect=layout["aspect"],
                       ylabel=layout["ylabel"], xticks=layout["xticks"],
                       xticklabels=layout["xticklabels"])
        ax.set_title(title, fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel(layout["xlabel"])

    n_units = int(np.asarray(info["n_units"]))
    n_spikes = int(np.asarray(info["n_spikes_all"]))
    # How much the phase of peak firing varies over the probe. A small spread means every
    # electrode sees theta at about the same phase; a large one means the probe crosses
    # something that rotates theta phase, which is what the cell layer does.
    spread = circular_spread(plot_metrics["peak_firing_phase_deg"].values)
    # Leave a fixed half inch at the top for the suptitle. tight_layout works in figure
    # fractions, so with a tall stack of panels that has to be converted from inches or the
    # title lands on top of the first panel's title.
    fig.tight_layout(rect=[0, 0, 1, 1 - 0.5 / fig_height])
    fig.suptitle(f"{info['session_id']}:  {n_units} good units, {n_spikes} spikes, "
                 f"{float(np.asarray(info['epoch_minutes'])):.0f} min, "
                 f"phase spread across probe {spread:.0f}°",
                 fontsize=14, y=1 - 0.12 / fig_height)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{info['session_id']}_phase_of_peak_firing.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"saved {path}")
    return fig


def plot_spike_phase_histograms(metrics, info, electrodes=None, n_show=6, save_dir=None):
    """Spike phase histograms for a few electrodes, as a sanity check on the map.

    Shows the pooled spike count vs theta phase over two cycles, so a real phase preference
    (a clear single hump) is easy to tell apart from a flat, unlocked electrode.

    Parameters:
        metrics (pd.DataFrame): First return value of compute_session_metrics
        info (dict): Second return value of compute_session_metrics
        electrodes (list): Electrode names to show. Default None = the `n_show` electrodes
            with the strongest locking (highest R), which is where the layer usually is.
        n_show (int): How many electrodes to show when `electrodes` is None. Default 6.
        save_dir (Path): Folder to save a PDF into. Default None = don't save.

    Returns:
        matplotlib.figure.Figure: The figure.
    """
    names = list(np.asarray(info["electrode_names"], dtype=str))
    hist_all = np.asarray(info["hist_all"], dtype=float)
    centers = np.asarray(info["bin_centers_deg"], dtype=float)

    if electrodes is None:
        ranked = metrics["peak_firing_R"].dropna().sort_values(ascending=False)
        electrodes = list(ranked.index[:n_show])
    electrodes = [str(e) for e in electrodes]

    n_cols = min(3, len(electrodes)) or 1
    n_rows = int(np.ceil(len(electrodes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows), squeeze=False)
    two_cycles = np.concatenate([centers, centers + 360])
    for ax, name in zip(axes.ravel(), electrodes):
        counts = hist_all[names.index(name)]
        ax.bar(two_cycles, np.concatenate([counts, counts]), width=360 / len(centers) * 0.9,
               color="0.6", edgecolor="0.3", linewidth=0.4)
        phase = metrics.loc[name, "peak_firing_phase_deg"]
        R = metrics.loc[name, "peak_firing_R"]
        for x in (phase, phase + 360):
            ax.axvline(x, color="red", lw=2)
        ax.set_title(f"electrode {name}: {phase:.0f}°, R = {R:.3f}", fontsize=9)
        ax.set_xticks([0, 180, 360, 540, 720])
        ax.set_xlabel("theta phase (deg; 0/360 = peak, 180 = trough)")
        ax.set_ylabel("spike count")
    for ax in axes.ravel()[len(electrodes):]:
        ax.axis("off")
    fig.suptitle(f"{info['session_id']}: pooled spike phase, strongest-locking electrodes",
                 fontsize=12)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{info['session_id']}_spike_phase_histograms.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"saved {path}")
    return fig
