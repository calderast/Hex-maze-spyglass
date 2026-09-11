"""

DEV TABLES FOR LOOKING AT SWEEPS.
SOME OR ALL OF THIS MAY BE CHANGED.

Theta sweeps: one row per theta cycle of the reference electrode, describing what the decode
did during that cycle.

A SWEEP is one full cycle of the reference theta, 0 to 360 degrees. Phase here is
peak-referenced (0/360 = LFP peak, 180 = trough), so a sweep runs peak to peak. Its start time
is the sample where phase wraps from ~360 back to ~0, and its end time is the start of the next
sweep. The partial cycles at the very beginning and end of the epoch are dropped, so every sweep 
stored here is a complete cycle.

For each sweep we ask what the decoded position was doing:

    Is it CONTINUOUS? Consecutive decoded positions inside the cycle should be close together. 
    The decoded position is median filtered first (to remove single sample jumps), then unconfident
    samples are masked out, then we see the distance between each consecutive decoded position.

    How far from the rat did it get? The minimum and maximum over the sweep are how far behind and 
    how far ahead the decode reached, and their difference is how much ground the
    sweep covered.

    NOTE the decode distance sign needs repairing before it can be trusted. 
    It is the sign of the cosine between the rat's heading and the direction to the decode, so a 
    decode sitting out to the SIDE has a cosine near zero and a sign decided by noise (we see
    e.g. the distance flip between -150 and +150 cm with the decoded position standing still). 
    repair_lateral_sign_flips finds those flips (sign changed, decode did not move) and
    unwraps them: each flip toggles the sign of the rest of that cycle, then the cycle is
    oriented by an |aheadness|-weighted vote. A real zero crossing has no flip to detect, so it
    survives, even in a cycle that also had a spurious flip.

    Is it SWEEPY? We expect the decode to be at/behind the rat in early theta (peak to trough)
    and sweep ahead through late theta (trough to peak). Against phase that is a sine starting at 
    180 degrees, -sin(phase), which is most behind at 90 and most ahead at 270. sweepiness is how 
    well the distance correlates with that template (1 ideal, -1 reversed, 0 unrelated), 
    sweep_amplitude_cm is how big the swing is, and sweep_peak_phase_deg is where the decode is 
    actually most ahead (should be ~270 if the template holds). This is a fake metric I made up to check
    theta phase and such.

Because max_jump_cm is stored per sweep, the continuity threshold can be changed in analysis
without repopulating: just re-compare max_jump_cm. The parameters used at populate time are all
saved on the row, so old entries stay self-documenting.

JUMP_THRESHOLD_CM is chosen in notebooks/Theta_Sweep_Threshold.ipynb, which plots the jump
distribution and how the continuous fraction responds to the threshold.

Typical usage:
    from spyglass_hexmaze.theta_sweeps import ThetaSweeps

    key = {"nwb_file_name": "IM-1947_20260415_.nwb", "epoch": 1}
    ThetaSweeps().populate(key, display_progress=True)
    sweeps = (ThetaSweeps & key).fetch1_dataframe()
"""

import datajoint as dj
import numpy as np
import pandas as pd

from spyglass.common import IntervalList, TaskEpoch
from spyglass.common.custom_nwbfile import AnalysisNwbfile as custom_AnalysisNwbfile
from spyglass.utils import SpyglassMixin

from spyglass_hexmaze.hex_maze_behavior import HexMazeBlock
from spyglass_hexmaze.hex_maze_decoding import (
    HexMazeDecodedPositionHex, HexMazeDecodedPositionHexV2, compute_aheadness,
)
from spyglass_hexmaze.hex_maze_theta import HexMazeThetaReference

# Same schema as the rest of the hex maze decoding tables
schema = dj.schema("hex_maze_decoding")

# These three are the parameters of the computation. They are module constants, and are written
# onto every row so a populated entry says which values produced it.

# Width of the median filter applied to decoded position before anything else, in samples. The
# decode is at 500 Hz, so 5 samples is 10 ms -- short next to a ~125 ms theta cycle, long enough
# to remove the single-sample excursions that would otherwise set a sweep's biggest jump.
MEDIAN_FILTER_SAMPLES = 5

# Decodes with spatial coverage at or above this are masked out. A decode spread over the whole
# maze is the model saying "the rat could be anywhere", so the position it happens to peak at is
# not a place the trajectory went. Same cutoff the rest of the hex maze theta code uses.
MAX_SPATIAL_COV = 200.0

# How far apart consecutive surviving decodes can be (cm) with the sweep still counting as one
# continuous trajectory. The MAP decode sits on a position grid whose bin is about 2.8 cm, and
# 99% of confident consecutive steps are within one bin, so this is "no more than about two
# bins". Chosen in notebooks/Theta_Sweep_Threshold.ipynb -- the one number in here that is a
# judgment call rather than a measurement.
JUMP_THRESHOLD_CM = 5.0


def sweep_start_times(reference_phase):
    """Start time of every complete theta cycle in a reference phase trace.

    A cycle starts where the phase wraps from ~2π back to ~0 (i.e. at the LFP peak, since the
    phase convention is peak-referenced). Consecutive wraps bound one sweep, so N wraps give
    N - 1 complete sweeps and the partial cycles at either end of the epoch are dropped.

    Parameters:
        reference_phase (pd.Series): Theta phase in radians [0, 2π), indexed by time in
            seconds. This is what HexMazeThetaReference.fetch1_reference_phase() returns.

    Returns:
        np.ndarray: Time of each cycle start, in seconds. Sweep i runs from element i to
            element i + 1, so the last element is only an end time, not a sweep of its own.
    """
    phase = reference_phase.to_numpy(dtype=float)
    # A drop of more than half a cycle between neighbouring samples is the 2π -> 0 wrap; any
    # real phase change between samples is far smaller than that
    wraps = np.flatnonzero(np.diff(phase) < -np.pi) + 1
    return reference_phase.index.to_numpy(dtype=float)[wraps]


def consecutive_jumps(decode_df):
    """Distance between each pair of consecutive decoded positions, in cm.

    Parameters:
        decode_df (pd.DataFrame): Time-indexed decode dataframe with decode_position_x and
            decode_position_y columns

    Returns:
        np.ndarray: Length len(decode_df) - 1, where element i is the distance from decoded
            sample i to sample i + 1.
    """
    x = decode_df["decode_position_x"].to_numpy(dtype=float)
    y = decode_df["decode_position_y"].to_numpy(dtype=float)
    return np.hypot(np.diff(x), np.diff(y))


def repair_lateral_sign_flips(decode_df, sweep_of_sample, jump_threshold_cm=JUMP_THRESHOLD_CM):
    """Undo the arbitrary +/- flips decode_distance takes when the decode is off to the side.

    decode_distance is sign(aheadness) * distance, where aheadness is the cosine of the angle
    between the rat's heading and the direction to the decoded position (compute_aheadness in
    hex_maze_decoding). When the decode sits out to the SIDE of the rat that cosine is near zero,
    so its sign is decided by noise: the stored distance flips between -150 and +150 cm while the
    decoded position has not moved at all, and every metric built on the signed distance --
    min, max, range, sweepiness -- reads that as an enormous sweep.

    A genuine change of side happens as the decode passes the rat, so the signed distance goes
    THROUGH zero and changes by a little. A spurious flip keeps the magnitude and changes the
    signed distance by twice it, with the decoded position sitting still. Hence the test: a sign
    change is spurious when the decoded position moved less than jump_threshold_cm while the
    signed distance changed by more.

    Repair is an UNWRAP, the same idea as unwrapping a phase: each spurious flip toggles the sign
    of every later sample in that sweep, which puts the run after the flip back on the side it was
    on before, and a genuine crossing (no flip detected) passes through untouched. So a cycle that
    holds both a real crossing and a spurious flip keeps the crossing and loses only the flip.

    Unwrapping fixes the signs RELATIVE to the first sample of the sweep, which may itself be one
    of the lateral samples that caused the trouble, so each sweep then has to be oriented. That is
    the majority vote: every sample backs its own run with the weight of |aheadness| -- how sure
    the ahead/behind call was in the first place -- and the heavier side keeps its original sign.
    A sweep with no spurious flip has one run and all weights on it, so it is provably unchanged.

    Parameters:
        decode_df (pd.DataFrame): Time-indexed decode dataframe, needs decode_distance,
            decode_position_x, decode_position_y, orientation, position_x and position_y
        sweep_of_sample (np.ndarray): Sweep index for each row of decode_df. Samples outside
            [0, n_sweeps) belong to no cycle; they are grouped and repaired like any other sweep,
            and every metric downstream ignores them anyway.
        jump_threshold_cm (float): Decoded movement below this counts as "it did not move", and a
            signed-distance change above it counts as a jump. Default JUMP_THRESHOLD_CM.

    Returns:
        tuple: (distance, info).
            distance (np.ndarray): decode_distance with the spurious flips unwrapped
            info (dict): n_sign_changes, n_spurious_flips and n_affected_sweeps, so a caller can
                report how much of the data this touched
    """
    distance = decode_df["decode_distance"].to_numpy(dtype=float)
    aheadness = compute_aheadness(decode_df).to_numpy(dtype=float)
    moved = consecutive_jumps(decode_df)   # how far the DECODED position moved between samples

    # Sign changes between samples adjacent in time and inside the same sweep
    sign_change = ((np.sign(distance[:-1]) * np.sign(distance[1:]) < 0)
                   & (sweep_of_sample[:-1] == sweep_of_sample[1:]))
    # ...that the decoded position did not move to justify
    spurious = (sign_change & (moved < jump_threshold_cm)
                & (np.abs(np.diff(distance)) > jump_threshold_cm))

    affected = np.unique(sweep_of_sample[:-1][spurious])
    info = {"n_sign_changes": int(sign_change.sum()),
            "n_spurious_flips": int(spurious.sum()),
            "n_affected_sweeps": int(len(affected))}
    if len(affected) == 0:
        return distance, info

    # Unwrap. flips_so_far counts the spurious flips before each sample; subtracting the count at
    # its own sweep's first sample leaves the flips seen INSIDE the sweep, and every odd-numbered
    # run is the one sitting on the wrong side. Note spurious flips only ever pair samples within
    # one sweep, so no toggle leaks across a sweep boundary.
    flips_so_far = np.concatenate([[0], np.cumsum(spurious)])
    at_sweep_start = (pd.Series(flips_so_far).groupby(sweep_of_sample)
                      .transform("first").to_numpy())
    unwrap = np.where((flips_so_far - at_sweep_start) % 2 == 1, -1.0, 1.0)

    # Orient each sweep: the runs vote with their |aheadness|, the side the decoder was surest
    # about wins, and the other run is flipped onto it. A clean sweep has unwrap = +1 throughout,
    # so its vote is the sum of |aheadness| >= 0 and its orientation comes out +1 -- unchanged.
    votes = pd.DataFrame({"sweep_number": sweep_of_sample,
                          "vote": unwrap * np.abs(np.nan_to_num(aheadness))})
    orientation = np.sign(votes.groupby("sweep_number")["vote"].sum())
    # A sweep whose votes cancel exactly (or that has no usable aheadness) keeps what it had
    orientation = orientation.replace(0, 1)

    repaired = distance * unwrap * orientation.reindex(sweep_of_sample).to_numpy(dtype=float)
    return repaired, info


def median_filter_decode(decode_df, window=MEDIAN_FILTER_SAMPLES):
    """Median filter the decoded position and distance, to drop single-sample excursions.

    A max-over-the-cycle statistic is exactly what one stray sample corrupts: at 500 Hz a single
    2 ms flicker to the far side of the maze would set a sweep's biggest jump and its furthest
    distance from the rat. A centred median over a few samples removes excursions shorter than
    half the window while leaving a real trajectory alone (unlike a mean, which would smear the
    excursion over its neighbours instead of discarding it).

    Only the decoded columns are filtered. The rat's own position, speed and the coverage are
    left as they are -- coverage in particular has to stay raw, since it is what the mask reads.

    Parameters:
        decode_df (pd.DataFrame): Time-indexed decode dataframe
        window (int): Median filter width in samples, odd so it is centred. Default
            MEDIAN_FILTER_SAMPLES.

    Returns:
        pd.DataFrame: Copy of decode_df with decode_position_x, decode_position_y and
            decode_distance median filtered.
    """
    filtered = decode_df.copy()
    columns = ["decode_position_x", "decode_position_y", "decode_distance"]
    # min_periods=1 so the half-window at each end keeps its samples instead of going NaN
    filtered[columns] = decode_df[columns].rolling(window, center=True, min_periods=1).median()
    return filtered


def compute_sweeps(reference_phase, decode_df, median_filter_samples=MEDIAN_FILTER_SAMPLES,
                   max_spatial_cov=MAX_SPATIAL_COV, jump_threshold_cm=JUMP_THRESHOLD_CM):
    """One row per theta cycle, describing what the decode did during that cycle.

    Kept separate from ThetaSweeps.make() so it can be run on a phase trace and a decode
    dataframe directly (e.g. while exploring in a notebook) without populating anything.

    Five steps, in this order:
        1. Repair the sign of decode_distance where a laterally-placed decode flipped it between
           ahead and behind without moving (see repair_lateral_sign_flips). Everything downstream
           reads the signed distance, so this has to happen before any of it.
        2. Median filter the decoded position and distance, so a single-sample excursion cannot
           set a sweep's biggest jump or its furthest distance.
        3. Mask out samples whose spatial coverage is at or above max_spatial_cov. Without this
           the biggest "jumps" are mostly the decoder being unsure rather than the trajectory
           moving.
        4. Record percent_good, how much of the cycle survived the mask. The other metrics are
           computed on the survivors whatever that fraction is, so percent_good is how much
           weight they deserve.
        5. Check continuity on the surviving samples, two ways: steps between samples adjacent
           in time (max_jump_cm, what is_continuous uses) and steps between consecutive
           survivors regardless of masked samples in between (max_jump_post_mask_cm).

    Parameters:
        reference_phase (pd.Series): Reference theta phase in radians [0, 2π), indexed by time
        decode_df (pd.DataFrame): Time-indexed decode dataframe, needs decode_position_x,
            decode_position_y, decode_distance, orientation, position_x, position_y, speed and
            spatial_cov (the position and orientation columns are what the sign repair needs)
        median_filter_samples (int): Median filter width for the decoded columns, in samples.
            Default MEDIAN_FILTER_SAMPLES.
        max_spatial_cov (float): Mask out decodes at or above this coverage. Default
            MAX_SPATIAL_COV.
        jump_threshold_cm (float): Biggest allowed step between consecutive surviving decodes
            for the sweep to count as continuous. Default JUMP_THRESHOLD_CM.

    Returns:
        pd.DataFrame: One row per complete theta cycle, indexed by sweep_number:
            start_time / end_time / duration: seconds; end_time is the next sweep's start
            n_samples: decode samples in the cycle, before the coverage mask
            n_good_samples: how many of those survived the mask
            percent_good: what percent of the cycle's samples survived the mask (0-100)
            max_jump_cm / mean_jump_cm: step between decodes that are adjacent in time and
                both survived the mask, so nothing that happened over a masked sample counts
            max_jump_post_mask_cm: step between consecutive SURVIVORS, which may have masked
                samples between them -- sample, masked, next sample is one step here. Always at
                least max_jump_cm, and comparing the two shows what the mask is hiding.
            is_continuous: 1 when max_jump_cm is within jump_threshold_cm, else 0. Comparing
                max_jump_post_mask_cm to the same threshold gives the stricter version, which
                counts a decode that moved while the decoder was briefly unsure.
            min_distance_cm / max_distance_cm: signed decode distance over the surviving
                samples, i.e. furthest behind and furthest ahead of the rat
            distance_range_cm: max_distance_cm - min_distance_cm, how much ground was covered
            mean_speed / mean_spatial_cov: over ALL samples in the cycle, so they describe the
                animal and the decoder rather than just the part that survived
            sweepiness: correlation of decode_distance with -sin(phase) over the surviving
                samples, in [-1, 1]. 1 = the expected shape (behind early, ahead late), -1 =
                reversed, 0 = no relation to the theta cycle. Shape only, not size.
            sweep_amplitude_cm: size of the first harmonic of distance over the cycle, i.e. how
                far the sweep actually swings
            sweep_peak_phase_deg: phase where that harmonic is most ahead of the rat, 270 if the
                template's phase holds -- measure it before assuming it does
            sweep_variance_explained: fraction of the distance variance a single sinusoid
                explains, in [0, 1]. "Is this cycle one clean sweep", with no assumption about
                where it peaks, so read it together with sweep_peak_phase_deg. NaN for sweeps
                with fewer than 3 surviving samples, as is sweepiness.
        A cycle with fewer than two surviving samples has no step to measure, so its jump and
        distance columns are NaN and is_continuous is 0 -- read that together with n_good_samples
        as "no evidence", not as "it jumped".
    """
    starts = sweep_start_times(reference_phase)
    if len(starts) < 2:
        raise ValueError(f"found {len(starts)} theta cycle starts, need at least 2")
    # Sweep i spans starts[i] to starts[i + 1], so the last start is only an end time
    n_sweeps = len(starts) - 1

    # Which sweep each decode sample falls in. searchsorted with side="right" gives the number
    # of starts at or before the sample, so subtracting 1 gives the sweep index; samples before
    # the first start come out -1 and samples after the last start come out n_sweeps.
    decode_time = decode_df.index.to_numpy(dtype=float)
    sweep_of_sample = np.searchsorted(starts, decode_time, side="right") - 1
    in_a_sweep = (sweep_of_sample >= 0) & (sweep_of_sample < n_sweeps)

    # 1. Repair the arbitrary sign flips a laterally-placed decode puts in decode_distance, before
    # anything reads that column. This comes first because the median filter cannot help here: a
    # column flipping between -150 and +150 has a perfectly stable median of either sign.
    repaired_distance, _ = repair_lateral_sign_flips(decode_df, sweep_of_sample, jump_threshold_cm)
    decode_df = decode_df.assign(decode_distance=repaired_distance)

    # 2. Median filter, on the full series, before anything is masked or split by sweep
    filtered_df = median_filter_decode(decode_df, window=median_filter_samples)

    # 3. Mask on the RAW coverage (the median filter deliberately left that column alone)
    good = in_a_sweep & (decode_df["spatial_cov"].to_numpy(dtype=float) < max_spatial_cov)

    # Reference theta phase at each decode sample, for the sweep shape metrics. The theta grid
    # and the decode grid have different sampling rates, so each decode sample takes the phase of
    # the nearest theta sample.
    phase_at_decode = np.mod(
        reference_phase.reindex(filtered_df.index, method="nearest").to_numpy(dtype=float),
        2 * np.pi)

    # Per-sample quantities. The distance columns come from the surviving samples, while speed
    # and coverage are grouped over every sample in the cycle so they describe the whole cycle.
    #
    # SWEEPINESS. A theta sequence should run from behind the rat to ahead of it once per cycle:
    # behind through early theta (peak to trough) and ahead through late theta (trough to peak).
    # As a function of phase that is a sine starting at 180 degrees, i.e. -sin(phase) -- zero at
    # the peak, most behind at 90, zero again at the trough, most ahead at 270. So the template
    # below IS that expected shape, and the columns are the pieces of two things measured against
    # it per sweep: the correlation of distance with the template (does it have the right shape)
    # and the first harmonic of distance (how big the swing is and where it actually peaks).
    distance = filtered_df["decode_distance"].to_numpy(dtype=float)[good]
    phase = phase_at_decode[good]
    template = -np.sin(phase)
    cos_phase, sin_phase = np.cos(phase), np.sin(phase)
    good_samples = pd.DataFrame(
        {"sweep_number": sweep_of_sample[good],
         "decode_distance": distance,
         "template": template,
         "distance_template": distance * template,
         "distance_sq": distance ** 2,
         "template_sq": template ** 2,
         # These five are the rest of the normal equations for the first-harmonic fit below
         "cos_phase": cos_phase,
         "sin_phase": sin_phase,
         "cos_sq": cos_phase ** 2,
         "sin_sq": sin_phase ** 2,
         "cos_sin": cos_phase * sin_phase,
         "distance_cos": distance * cos_phase,
         "distance_sin": distance * sin_phase})
    by_good = good_samples.groupby("sweep_number")
    all_samples = pd.DataFrame(
        {"sweep_number": sweep_of_sample[in_a_sweep],
         "speed": decode_df["speed"].to_numpy(dtype=float)[in_a_sweep],
         "spatial_cov": decode_df["spatial_cov"].to_numpy(dtype=float)[in_a_sweep]})
    by_all = all_samples.groupby("sweep_number")

    # 5. Steps between consecutive surviving decodes. A step counts only when BOTH endpoints
    # survived and are in the same sweep, so the step across a sweep boundary is not held against
    # either side (the decode is expected to move on to new ground there). The steps are measured
    # on the FULL series, so a pair is only ever two samples that were adjacent in time --
    # masking a sample out never joins the two either side of it into one long fake jump.
    jumps = consecutive_jumps(filtered_df)
    same_sweep = good[:-1] & good[1:] & (sweep_of_sample[:-1] == sweep_of_sample[1:])
    jumps_by_sweep = pd.DataFrame(
        {"sweep_number": sweep_of_sample[:-1][same_sweep],
         "jump": jumps[same_sweep]}).groupby("sweep_number")["jump"]

    # ...and the same thing measured AFTER the mask, where consecutive means consecutive among
    # the survivors: sample, masked sample, next sample is one step here, so a decode that moved
    # across the maze while the decoder was briefly unsure shows up. The two columns bracket what
    # the mask is hiding -- max_jump_cm ignores anything that happened over a masked sample,
    # max_jump_post_mask_cm charges the whole distance to the surviving pair, and it is always the
    # larger of the two because its pairs are a superset.
    survivor_positions = np.flatnonzero(good)
    survivor_sweep = sweep_of_sample[survivor_positions]
    post_mask_jumps = consecutive_jumps(filtered_df.iloc[survivor_positions])
    post_mask_same_sweep = survivor_sweep[:-1] == survivor_sweep[1:]
    post_mask_by_sweep = pd.DataFrame(
        {"sweep_number": survivor_sweep[:-1][post_mask_same_sweep],
         "jump": post_mask_jumps[post_mask_same_sweep]}).groupby("sweep_number")["jump"]

    # Reindex everything onto the full range of sweeps, so a cycle with no decode samples in it
    # still gets a row (counts 0, metrics NaN) instead of silently vanishing
    sweep_numbers = pd.RangeIndex(n_sweeps, name="sweep_number")
    sweeps = pd.DataFrame(index=sweep_numbers)
    sweeps["start_time"] = starts[:-1]
    sweeps["end_time"] = starts[1:]
    sweeps["duration"] = sweeps["end_time"] - sweeps["start_time"]
    sweeps["n_samples"] = by_all.size().reindex(sweep_numbers, fill_value=0)
    sweeps["n_good_samples"] = by_good.size().reindex(sweep_numbers, fill_value=0)

    # 4. How much of the cycle survived the mask, as a percent. Kept as a number rather than a
    # pass/fail flag so the cutoff is a decision made in analysis: percent_good < 50 reproduces
    # a "low confidence" flag, and any other cutoff is equally available. A cycle with no samples
    # at all comes out NaN -- there was nothing to keep or throw away.
    sweeps["percent_good"] = (100.0 * sweeps["n_good_samples"]
                              / sweeps["n_samples"].where(sweeps["n_samples"] > 0))

    sweeps["max_jump_cm"] = jumps_by_sweep.max().reindex(sweep_numbers)
    sweeps["mean_jump_cm"] = jumps_by_sweep.mean().reindex(sweep_numbers)
    sweeps["max_jump_post_mask_cm"] = post_mask_by_sweep.max().reindex(sweep_numbers)
    # Fewer than two surviving samples means no step to measure, so max_jump_cm is NaN and the
    # comparison is False: one point is not evidence of a continuous trajectory
    sweeps["is_continuous"] = sweeps["max_jump_cm"].le(jump_threshold_cm).astype("int8")
    sweeps["min_distance_cm"] = by_good["decode_distance"].min().reindex(sweep_numbers)
    sweeps["max_distance_cm"] = by_good["decode_distance"].max().reindex(sweep_numbers)
    sweeps["distance_range_cm"] = sweeps["max_distance_cm"] - sweeps["min_distance_cm"]
    sweeps["mean_speed"] = by_all["speed"].mean().reindex(sweep_numbers)
    sweeps["mean_spatial_cov"] = by_all["spatial_cov"].mean().reindex(sweep_numbers)

    # ---- sweepiness: does the distance trace the expected -sin(phase) shape? ------------------
    # Everything here comes out of per-sweep sums, so it stays one vectorized pass over ~40k
    # sweeps rather than a fit per sweep.
    sums = by_good.sum().reindex(sweep_numbers)
    n_good = sweeps["n_good_samples"].where(sweeps["n_good_samples"] > 0)

    # Correlation of distance with the template, in [-1, 1]. 1 = the ideal sweep shape (behind
    # early, ahead late), -1 = exactly reversed, 0 = no relation to the theta cycle. This is a
    # SHAPE measure and says nothing about size: a 2 cm wiggle of the right shape scores as high
    # as a 40 cm sweep.
    covariance = sums["distance_template"] - sums["decode_distance"] * sums["template"] / n_good
    distance_var = sums["distance_sq"] - sums["decode_distance"] ** 2 / n_good
    template_var = sums["template_sq"] - sums["template"] ** 2 / n_good
    # Guard the denominator: a sweep whose surviving samples barely span any phase, or whose
    # distance barely varies, has a near-zero variance and would divide out to a huge number. The
    # clip mops up the roundoff that can still push a correlation a hair past 1, and correlations
    # from fewer than 3 points are dropped outright as meaningless.
    denominator = np.sqrt(distance_var.where(distance_var > 0) * template_var.where(template_var > 0))
    enough = n_good.where(n_good >= 3)
    sweeps["sweepiness"] = (covariance / denominator).clip(-1, 1).where(enough.notna())

    # First harmonic of distance over the cycle, fitted as mean + a*cos(phase) + b*sin(phase).
    # Its size is how far the sweep actually swings, its angle is where the decode is most ahead
    # (270 degrees for the shape the template expects), and how much of the distance variance it
    # explains says whether the cycle is one clean sweep at all -- that last one assumes nothing
    # about WHERE the peak sits, so it does not penalise sweeps phased differently.
    #
    # This is a real least-squares fit rather than the textbook Fourier shortcut (a = 2*mean(d*cos)
    # and so on), because that shortcut needs cos and sin to be orthogonal over the sweep. They
    # only are when the samples span the cycle evenly, and a sweep with part of its cycle masked
    # out does not: the shortcut then inflates the amplitude and the explained variance and biases
    # the phase, exactly on the sweeps that deserve the least trust.
    fit_columns = ["cos_phase", "sin_phase", "cos_sq", "sin_sq", "cos_sin",
                   "decode_distance", "distance_cos", "distance_sin", "distance_sq"]
    fit = sums[fit_columns]
    count = sweeps["n_good_samples"].to_numpy(dtype=float)
    # Normal equations, one 3x3 system per sweep: [[n, Sc, Ss], [Sc, Scc, Scs], [Ss, Scs, Sss]]
    # times [mean, a, b] equals [Sd, Sdc, Sds]
    normal = np.empty((len(sweeps), 3, 3))
    normal[:, 0, 0] = count
    normal[:, 0, 1] = normal[:, 1, 0] = fit["cos_phase"]
    normal[:, 0, 2] = normal[:, 2, 0] = fit["sin_phase"]
    normal[:, 1, 1] = fit["cos_sq"]
    normal[:, 2, 2] = fit["sin_sq"]
    normal[:, 1, 2] = normal[:, 2, 1] = fit["cos_sin"]
    target = np.column_stack([fit["decode_distance"], fit["distance_cos"], fit["distance_sin"]])

    # Solve only the sweeps that can be solved: enough samples, and a system that is not
    # degenerate (all samples at nearly one phase makes it singular). np.linalg.det scales like
    # n**3 for even coverage, so the cutoff is relative to that rather than absolute.
    solvable = (count >= 3) & np.isfinite(target).all(axis=1)
    with np.errstate(invalid="ignore"):
        solvable &= np.abs(np.linalg.det(np.where(solvable[:, None, None], normal, np.eye(3)))) \
                    > 1e-9 * np.maximum(count, 1) ** 3
    coefficients = np.full((len(sweeps), 3), np.nan)
    if solvable.any():
        coefficients[solvable] = np.linalg.solve(normal[solvable], target[solvable])
    intercept, cosine_term, sine_term = coefficients.T

    sweeps["sweep_amplitude_cm"] = np.hypot(cosine_term, sine_term)
    sweeps["sweep_peak_phase_deg"] = np.degrees(np.arctan2(sine_term, cosine_term)) % 360

    # R² from the fitted coefficients: for a least-squares fit the residual sum of squares is
    # Sdd - coefficients . target, and the total is the distance variance already computed
    residual = fit["distance_sq"].to_numpy() - (coefficients * target).sum(axis=1)
    total = distance_var.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        variance_explained = np.where(total > 0, 1.0 - residual / total, np.nan)
    sweeps["sweep_variance_explained"] = pd.Series(
        variance_explained, index=sweep_numbers).where(enough.notna())
    return sweeps


@schema
class ThetaSweeps(SpyglassMixin, dj.Computed):
    """
    One row per theta cycle of the reference electrode, describing the decode during it.

    A sweep is one full cycle of the reference theta (0 to 360 degrees, peak to peak); its end
    time is the start of the next sweep. Per sweep we store how continuous the decoded
    trajectory was and how far ahead of / behind the rat it reached, using only decodes
    confident enough to mean anything (spatial_cov below MAX_SPATIAL_COV) after a median filter.
    See the module docstring for the definitions and for how the thresholds were chosen.
    """

    definition = """
    -> HexMazeThetaReference
    -> HexMazeDecodedPositionHex
    ---
    -> custom_AnalysisNwbfile
    theta_sweeps_object_id: varchar(128)
    median_filter_samples: int  # median filter width applied to the decoded columns
    max_spatial_cov: float      # decodes at or above this coverage were masked out
    jump_threshold_cm: float    # continuity threshold behind the stored is_continuous column
    """

    @property
    def key_source(self):
        # The two parents are keyed differently on time: HexMazeThetaReference by
        # target_interval_list_name (it comes from LFP) and HexMazeDecodedPositionHex by epoch
        # number. They only share nwb_file_name, so DataJoint's default key_source would pair
        # every theta reference with every decode of that session, including decodes of other
        # epochs. TaskEpoch maps interval name -> epoch, so joining through it keeps only the
        # pairs that describe the SAME epoch.
        epoch_of_interval = TaskEpoch.proj(target_interval_list_name="interval_list_name")
        return (HexMazeThetaReference * epoch_of_interval * HexMazeDecodedPositionHex).proj()

    def make(self, key):
        # Skip if already populated
        if self & key:
            return

        # Reference theta phase (radians [0, 2π), peak-referenced) averaged over the saved
        # reference electrodes, and the decode for the same epoch
        reference_phase = (HexMazeThetaReference & key).fetch1_reference_phase()
        decode_df = (HexMazeDecodedPositionHex & key).fetch1_dataframe()

        sweeps = compute_sweeps(reference_phase, decode_df,
                                median_filter_samples=MEDIAN_FILTER_SAMPLES,
                                max_spatial_cov=MAX_SPATIAL_COV,
                                jump_threshold_cm=JUMP_THRESHOLD_CM)
        # Record the parameters that produced this row (a params table takes over later)
        key["median_filter_samples"] = MEDIAN_FILTER_SAMPLES
        key["max_spatial_cov"] = MAX_SPATIAL_COV
        key["jump_threshold_cm"] = JUMP_THRESHOLD_CM

        # Save sweep_number as a column instead of the index (NWB requires an integer index)
        sweeps = sweeps.reset_index()

        # Create an AnalysisNwbfile with a link to the original nwb and add the df
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["theta_sweeps_object_id"] = builder.add_nwb_object(sweeps, "theta_sweeps")
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        """The sweeps for one entry, indexed by sweep_number."""
        return self.fetch_nwb()[0]["theta_sweeps"].set_index("sweep_number")


## Where the sweeps went: per-cycle location, speed and target

def _nearest_index(sorted_times, query_times):
    """Index of the nearest entry of sorted_times for each query time.

    Parameters:
        sorted_times (np.ndarray): Ascending times to look up in
        query_times (np.ndarray): Times to find neighbours for

    Returns:
        np.ndarray: One index into sorted_times per query time.
    """
    after = np.clip(np.searchsorted(sorted_times, query_times), 1, len(sorted_times) - 1)
    before = after - 1
    return np.where(query_times - sorted_times[before] <= sorted_times[after] - query_times,
                    before, after)


def _values_at(positions, values, index):
    """Look up `values` at the sample positions in `positions`, keeping NaN where there is none.

    Parameters:
        positions (pd.Series): Sample position per sweep (from idxmax/idxmin), NaN where the
            sweep had no surviving sample
        values (np.ndarray): Per-sample array to read, same length as the decode
        index (pd.Index): Sweep numbers to return, so every sweep gets a row

    Returns:
        pd.Series: One value per sweep, dtype object so hexes and status labels both work.
    """
    out = pd.Series(np.nan, index=index, dtype=object)
    present = positions.dropna()
    if len(present):
        out.loc[present.index] = values[present.to_numpy().astype(int)]
    return out


def trial_context(nwb_file_name, epoch):
    """Trial bounds and their block, for one epoch.

    Parameters:
        nwb_file_name (str): Session to read
        epoch (int): Epoch number, as HexMazeBlock keys it

    Returns:
        pd.DataFrame: One row per trial ordered by start_time, with start_time, end_time, block,
            block_trial_num and epoch_trial_num. Empty frame when the epoch has no trials.
    """
    trials = pd.DataFrame((HexMazeBlock.Trial
                           & {"nwb_file_name": nwb_file_name, "epoch": epoch}).fetch(as_dict=True))
    if not len(trials):
        return pd.DataFrame()

    # Trial bounds live in IntervalList, reached through the interval name the trial points at.
    # Fetch them in ONE query and look them up in memory: an epoch can have hundreds of trials,
    # and a fetch1 per trial is that many round trips to the database, which under load is
    # minutes rather than seconds.
    #
    # Restrict to the intervals THESE trials point at rather than taking the whole session.
    # IntervalList and HexMazeBlock.Trial share nwb_file_name and interval_list_name, so the
    # restriction below matches on exactly those two. Besides being a smaller fetch, it steps
    # around unrelated junk rows: Toby20250318_ carries a scratch interval ("test decoding
    # interval_") whose stored blob will not decode, and one bad row fails the entire fetch.
    intervals = (IntervalList & (HexMazeBlock.Trial
                                 & {"nwb_file_name": nwb_file_name, "epoch": epoch})).fetch(
        "interval_list_name", "valid_times", as_dict=True)

    # valid_times is usually (n_intervals, 2), but a single-interval row can come back flat as
    # (2,) -- atleast_2d makes [0][0] the first start and [-1][-1] the last end in both cases.
    bounds = {row["interval_list_name"]: (np.atleast_2d(row["valid_times"])[0][0],
                                          np.atleast_2d(row["valid_times"])[-1][-1])
              for row in intervals if len(row["valid_times"])}
    trials["start_time"] = [bounds[name][0] for name in trials["interval_list_name"]]
    trials["end_time"] = [bounds[name][1] for name in trials["interval_list_name"]]
    return (trials[["start_time", "end_time", "block", "block_trial_num", "epoch_trial_num"]]
            .sort_values("start_time").reset_index(drop=True))


def compute_sweep_stats(reference_phase, decode_df, trials=None,
                        median_filter_samples=MEDIAN_FILTER_SAMPLES,
                        max_spatial_cov=MAX_SPATIAL_COV,
                        jump_threshold_cm=JUMP_THRESHOLD_CM):
    """One row per theta cycle saying WHERE the sweep went, for the location analyses.

    Kept separate from ThetaSweepStats.make() the same way compute_sweeps is, so it can be run
    on a phase trace and a decode dataframe directly while exploring in a notebook.

    Cycles are cut exactly as compute_sweeps cuts them (same sweep_start_times on the same
    reference phase), and the distance columns go through the same sign repair and median filter,
    so sweep_number and the distances line up with ThetaSweeps row for row.

    Two different sample sets are used on purpose:
        the rat's own columns -- speed, orientation, its hex -- use EVERY sample in the cycle,
            because where the rat is does not depend on how sure the decoder was
        the decode's columns -- distances, visited hexes, open-status shares -- use only samples
            below max_spatial_cov, because a decode spread over the whole maze is not a place the
            trajectory went. percent_good says how much of the cycle that left.

    Hexes come from decode_hex_any, the true nearest hex whether or not it was open, with the
    open status alongside. decode_hex would snap a sweep into a barriered hex onto its nearest
    open neighbour, which is exactly the case worth seeing.

    The sign of decode_distance is the per-sample one (sign of the cosine between the rat's
    heading AT THAT SAMPLE and the direction to the decode), after repair_lateral_sign_flips.
    Its magnitude is graph distance and does not depend on heading at all.

    Parameters:
        reference_phase (pd.Series): Reference theta phase in radians [0, 2π), indexed by time
        decode_df (pd.DataFrame): Time-indexed HexMazeDecodedPositionHexV2 dataframe. Needs
            everything compute_sweeps needs plus hex, decode_hex_any, decode_hex_open_status and
            decode_hex_blocks_since_open.
        trials (pd.DataFrame): Optional output of trial_context. When given, each sweep gets the
            block and trial its START falls in; sweeps between trials get NaN.
        median_filter_samples (int): Median filter width for the decoded columns, in samples
        max_spatial_cov (float): Mask out decodes at or above this coverage
        jump_threshold_cm (float): Passed to repair_lateral_sign_flips

    Returns:
        pd.DataFrame: One row per complete theta cycle, indexed by sweep_number:
            start_time / end_time: seconds; end_time is the next cycle's start
            n_samples / n_good_samples / percent_good: decode samples in the cycle, how many
                survived the coverage mask, and that as a percent
            speed_min / speed_max / speed_mean: the rat's speed over the cycle (cm/s)
            rat_hex_start / rat_hex_end / n_rat_hexes: the hex the RAT was in at the first and
                last sample of the cycle that HAS one (NaN skipped), and how many it touched.
                NaN only when the cycle has no samples at all.
            orientation_start_deg / orientation_mean_deg: the rat's heading at the first sample
                with one and its circular mean over the cycle, degrees [0, 360)
            orientation_R: resultant length of the heading, 1 = head perfectly still
            orientation_net_change_deg: |last heading - first|, wrapped into [0, 180]
            orientation_total_rotation_deg: summed |change| between consecutive samples, so a
                head that turned and came back still counts
            ahead_max_cm / behind_max_cm: the extremes of the signed distance over surviving
                samples. Same as ThetaSweeps' max_distance_cm / min_distance_cm. Both keep the
                sign, so an all-behind cycle has a negative ahead_max_cm ("least behind").
            ahead_hex / behind_hex: decode_hex_any at those two samples, where the sweep reached
            ahead_hex_status / behind_hex_status: decode_hex_open_status of those hexes, so a
                sweep into a hex the barrier just closed is visible
            ahead_offset_s / behind_offset_s: seconds from the cycle start to those extremes
            ahead_phase_deg / behind_phase_deg: reference theta phase at those samples, degrees.
                The sweep template puts the ahead extreme near 270.
            furthest_abs_cm / furthest_hex: the larger extreme in absolute value, and its hex
            closest_abs_cm: the closest the decode ever got to the rat, unsigned. Not recoverable
                from the two extremes, and what "this sweep never represented the rat's own
                position" is measured with.
            distance_range_cm: ahead_max_cm - behind_max_cm
            max_jump_cm: biggest step between decodes ADJACENT IN TIME that both survived the
                mask. Same as ThetaSweeps' column of that name.
            max_jump_post_mask_cm: biggest step between CONSECUTIVE SURVIVORS, so a decode that
                moved while the decoder was briefly unsure is charged the whole distance. Always
                at least max_jump_cm. Same as ThetaSweeps' column of that name.
            path_length_cm: summed post-mask steps, large for a sweep that went out and came
                back, which distance_range_cm would call small
            is_continuous: 1 when max_jump_post_mask_cm is within jump_threshold_cm, else 0.
                NOTE this is a STRICTER rule than ThetaSweeps.is_continuous, which reads
                max_jump_cm and forgives a decode that moved across a masked gap. 0 also means
                "fewer than two surviving samples", i.e. no evidence.
            hexes_visited: every hex the surviving decode passed through, comma separated, in the
                order first reached ("17,22,29")
            hex_sample_counts: samples spent in each, same order ("12,19,8"), so a hex the sweep
                merely clipped can be told from the one it sat in
            n_hexes_visited: how many hexes that was
            share_open / share_closed_1_block_ago / share_closed_2plus_blocks_ago /
                share_never_open / share_outside_block: fraction of surviving samples whose hex
                had that status, from decode_hex_blocks_since_open
            mean_cov_good: mean spatial coverage over the surviving samples, the decode-quality
                control for anything drawn per hex
            share_decode_in_maze: fraction of surviving samples whose DECODE was inside the
                maze's physical footprint. decode_hex_any gives a nearest hex to any position,
                off the edge of the maze included, so filter on this before trusting the hexes.
            share_rat_in_maze: the same for the rat's own position, over every sample. NOT a
                tracking control and nowhere near 1: the footprint excludes the reward ports, so
                it reads ~0 whenever the rat is at one. Read it as "on the maze proper rather
                than parked at a port".
            block / block_trial_num / epoch_trial_num: only when `trials` is given
    """
    starts = sweep_start_times(reference_phase)
    if len(starts) < 2:
        raise ValueError(f"found {len(starts)} theta cycle starts, need at least 2")
    n_sweeps = len(starts) - 1

    decode_time = decode_df.index.to_numpy(dtype=float)
    sweep_of_sample = np.searchsorted(starts, decode_time, side="right") - 1
    in_a_sweep = (sweep_of_sample >= 0) & (sweep_of_sample < n_sweeps)

    # The same first two steps compute_sweeps takes, in the same order, so the distances here are
    # the stored ones and not a second opinion
    repaired_distance, _ = repair_lateral_sign_flips(decode_df, sweep_of_sample, jump_threshold_cm)
    decode_df = decode_df.assign(decode_distance=repaired_distance)
    filtered_df = median_filter_decode(decode_df, window=median_filter_samples)
    good = in_a_sweep & (decode_df["spatial_cov"].to_numpy(dtype=float) < max_spatial_cov)

    index = pd.RangeIndex(n_sweeps, name="sweep_number")
    sweep_all, sweep_good = sweep_of_sample[in_a_sweep], sweep_of_sample[good]
    good_positions = np.flatnonzero(good)

    def over_all(values):
        """Group a per-sample array by sweep, over every sample in the cycle."""
        return pd.Series(values[in_a_sweep]).groupby(sweep_all)

    def over_good(values):
        """Group a per-sample array by sweep, over the samples that survived the mask."""
        return pd.Series(values[good], index=good_positions).groupby(sweep_good)

    stats = pd.DataFrame(index=index)
    stats["start_time"] = starts[:-1]
    stats["end_time"] = starts[1:]
    stats["n_samples"] = over_all(np.ones(len(decode_time))).size().reindex(index).fillna(0)
    stats["n_good_samples"] = over_good(np.ones(len(decode_time))).size().reindex(index).fillna(0)
    stats["percent_good"] = 100 * stats["n_good_samples"] / stats["n_samples"].replace(0, np.nan)

    # --- the rat: every sample, confident or not -------------------------------------------
    speed = decode_df["speed"].to_numpy(dtype=float)
    stats["speed_min"] = over_all(speed).min().reindex(index)
    stats["speed_max"] = over_all(speed).max().reindex(index)
    stats["speed_mean"] = over_all(speed).mean().reindex(index)

    rat_hex = decode_df["hex"].to_numpy()
    stats["rat_hex_start"] = over_all(rat_hex).first().reindex(index)
    stats["rat_hex_end"] = over_all(rat_hex).last().reindex(index)
    stats["n_rat_hexes"] = over_all(rat_hex).nunique().reindex(index)

    # Heading is an angle, so its mean is the circular one: average the unit vectors and take the
    # direction back off. The length of that average (R) is how still the head was.
    orientation = decode_df["orientation"].to_numpy(dtype=float)
    mean_cos = over_all(np.cos(orientation)).mean().reindex(index)
    mean_sin = over_all(np.sin(orientation)).mean().reindex(index)
    stats["orientation_mean_deg"] = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
    stats["orientation_R"] = np.hypot(mean_cos, mean_sin)
    first_orientation = over_all(orientation).first().reindex(index)
    last_orientation = over_all(orientation).last().reindex(index)
    stats["orientation_start_deg"] = np.degrees(first_orientation) % 360
    # Wrap the difference through the unit circle so 359 -> 1 degrees reads as 2, not 358
    stats["orientation_net_change_deg"] = np.degrees(
        np.abs(np.angle(np.exp(1j * (last_orientation - first_orientation)))))
    # Total rotation counts every step, so a head that turned and came back is not called still.
    # A step only counts when both of its samples are in the same cycle.
    step = np.angle(np.exp(1j * np.diff(orientation)))
    step_sweep = sweep_of_sample[1:]
    step_in_sweep = (step_sweep == sweep_of_sample[:-1]) & in_a_sweep[1:]
    stats["orientation_total_rotation_deg"] = (
        pd.Series(np.degrees(np.abs(step))[step_in_sweep])
        .groupby(step_sweep[step_in_sweep]).sum().reindex(index))

    # --- the decode: confident samples only -------------------------------------------------
    signed_distance = filtered_df["decode_distance"].to_numpy(dtype=float)
    by_distance = over_good(signed_distance)
    stats["ahead_max_cm"] = by_distance.max().reindex(index)
    stats["behind_max_cm"] = by_distance.min().reindex(index)
    ahead_at = by_distance.idxmax().reindex(index)
    behind_at = by_distance.idxmin().reindex(index)

    # How close the decode ever got to the rat. This one is NOT recoverable from the signed
    # extremes: a cycle that crosses the rat has a closest approach of about zero however far
    # its two ends reached. It is what "this sweep never represented the rat's own position"
    # is measured with.
    stats["closest_abs_cm"] = over_good(np.abs(signed_distance)).min().reindex(index)
    stats["distance_range_cm"] = stats["ahead_max_cm"] - stats["behind_max_cm"]

    decode_hex = decode_df["decode_hex_any"].to_numpy()
    open_status = decode_df["decode_hex_open_status"].to_numpy()
    phase_at_sample = np.degrees(
        reference_phase.to_numpy(dtype=float)[
            _nearest_index(reference_phase.index.to_numpy(dtype=float), decode_time)]) % 360

    # Every column has to land on ONE type: the analysis NWB writer infers a dtype per column
    # and a mix of strings and NaN makes it guess float and then choke on the first label. So
    # hexes are numeric with NaN for "no surviving sample", and the status labels are strings
    # with "" for the same thing.
    for name, positions in [("ahead", ahead_at), ("behind", behind_at)]:
        stats[f"{name}_hex"] = pd.to_numeric(
            _values_at(positions, decode_hex, index), errors="coerce")
        stats[f"{name}_hex_status"] = (
            _values_at(positions, open_status, index).fillna("").astype(str))
        stats[f"{name}_offset_s"] = (
            pd.to_numeric(_values_at(positions, decode_time, index)) - stats["start_time"])
        stats[f"{name}_phase_deg"] = pd.to_numeric(_values_at(positions, phase_at_sample, index))

    # Whichever end of the cycle reached further from the rat, sign kept so it is still readable
    # as ahead or behind
    reach_ahead = stats["ahead_max_cm"].abs()
    reach_behind = stats["behind_max_cm"].abs()
    ahead_wins = reach_ahead >= reach_behind
    stats["furthest_abs_cm"] = np.where(ahead_wins, reach_ahead, reach_behind)
    stats["furthest_hex"] = np.where(ahead_wins, stats["ahead_hex"], stats["behind_hex"])

    # Continuity. Both jump columns are measured exactly as compute_sweeps measures them, so they
    # agree with their ThetaSweeps namesakes value for value.
    #
    # max_jump_cm: steps between samples ADJACENT IN TIME that both survived the mask, taken off
    # the full series, so nothing that happened across a masked sample counts.
    jumps = consecutive_jumps(filtered_df)
    step_survived = good[:-1] & good[1:] & (sweep_of_sample[:-1] == sweep_of_sample[1:])
    stats["max_jump_cm"] = (
        pd.DataFrame({"sweep_number": sweep_of_sample[:-1][step_survived],
                      "jump": jumps[step_survived]})
        .groupby("sweep_number")["jump"].max().reindex(index))

    # max_jump_post_mask_cm: steps between CONSECUTIVE SURVIVORS, which may have masked samples
    # between them -- sample, masked, next sample is one step here. Always at least max_jump_cm,
    # because its pairs are a superset.
    survivor_positions = np.flatnonzero(good)
    survivor_sweep = sweep_of_sample[survivor_positions]
    post_mask_jumps = consecutive_jumps(filtered_df.iloc[survivor_positions])
    post_mask_same_sweep = survivor_sweep[:-1] == survivor_sweep[1:]
    by_post_mask = pd.DataFrame(
        {"sweep_number": survivor_sweep[:-1][post_mask_same_sweep],
         "jump": post_mask_jumps[post_mask_same_sweep]}).groupby("sweep_number")["jump"]
    stats["max_jump_post_mask_cm"] = by_post_mask.max().reindex(index)
    stats["path_length_cm"] = by_post_mask.sum().reindex(index)

    # is_continuous uses the POST-MASK jump, which is the STRICTER of the two and is NOT what
    # ThetaSweeps.is_continuous uses -- that one reads max_jump_cm and so forgives a decode that
    # crossed the maze while the decoder was briefly unsure. Filtering to genuinely continuous
    # trajectories is the job here, so a gap the mask hid still counts as a jump. Expect this
    # column to be a subset of the ThetaSweeps one.
    #
    # A cycle with fewer than two surviving samples has no step to measure, so the jump is NaN
    # and the comparison is False -- it comes out 0, meaning "no evidence", not "it jumped".
    stats["is_continuous"] = (stats["max_jump_post_mask_cm"] <= jump_threshold_cm).astype(int)

    # --- where the sweep went, in order -----------------------------------------------------
    visited = pd.DataFrame({"sweep": sweep_good, "hex": decode_hex[good],
                            "position": good_positions}).dropna(subset=["hex"])
    if len(visited):
        # One row per (sweep, hex): how many samples it held, and the first one, which is what
        # puts the hexes in the order the sweep reached them
        per_hex = (visited.groupby(["sweep", "hex"])
                   .agg(n=("position", "size"), first_position=("position", "min"))
                   .reset_index().sort_values(["sweep", "first_position"]))
        per_hex["hex_text"] = per_hex["hex"].astype(int).astype(str)
        per_hex["n_text"] = per_hex["n"].astype(str)
        joined = per_hex.groupby("sweep").agg(
            hexes_visited=("hex_text", ",".join),
            hex_sample_counts=("n_text", ",".join),
            n_hexes_visited=("hex_text", "size"))
        stats["hexes_visited"] = joined["hexes_visited"].reindex(index).fillna("")
        stats["hex_sample_counts"] = joined["hex_sample_counts"].reindex(index).fillna("")
        stats["n_hexes_visited"] = joined["n_hexes_visited"].reindex(index).fillna(0)
    else:
        stats["hexes_visited"] = ""
        stats["hex_sample_counts"] = ""
        stats["n_hexes_visited"] = 0

    # Share of the surviving samples by how recently their hex was open. Taken off the numeric
    # blocks_since_open rather than the label, so the columns are the same set every epoch
    blocks_since = decode_df["decode_hex_blocks_since_open"].to_numpy(dtype=float)
    status_masks = {"open": blocks_since == 0,
                    "closed_1_block_ago": blocks_since == 1,
                    "closed_2plus_blocks_ago": blocks_since >= 2,
                    "never_open": blocks_since == -1,
                    "outside_block": blocks_since == -100}
    for name, mask in status_masks.items():
        stats[f"share_{name}"] = over_good(mask.astype(float)).mean().reindex(index)

    stats["mean_cov_good"] = over_good(
        decode_df["spatial_cov"].to_numpy(dtype=float)).mean().reindex(index)

    # Inside the maze's physical footprint. decode_hex_any assigns a nearest hex to ANY position,
    # including one off the edge of the maze, so a cycle whose decode wandered outside still
    # contributes hexes -- this is the column to filter that on. The footprint excludes the
    # reward ports, so the rat's own share is low whenever it is parked at one rather than
    # running the maze: a behavioral state flag, not a tracking control.
    stats["share_decode_in_maze"] = over_good(
        decode_df["decode_in_maze"].to_numpy(dtype=float)).mean().reindex(index)
    stats["share_rat_in_maze"] = over_all(
        decode_df["in_maze"].to_numpy(dtype=float)).mean().reindex(index)

    # --- which trial the cycle happened in ---------------------------------------------------
    if trials is not None and len(trials):
        # By the cycle's START: a cycle is ~125 ms and trial bounds are seconds apart, so the
        # handful that straddle a boundary are not worth a rule of their own
        cycle_start = stats["start_time"].to_numpy()
        trial_index = np.searchsorted(trials["start_time"].to_numpy(), cycle_start, side="right") - 1
        clipped = np.clip(trial_index, 0, len(trials) - 1)
        inside = (trial_index >= 0) & (cycle_start <= trials["end_time"].to_numpy()[clipped])
        for column in ["block", "block_trial_num", "epoch_trial_num"]:
            stats[column] = np.where(inside, trials[column].to_numpy()[clipped], np.nan)

    return stats


@schema
class ThetaSweepStats(SpyglassMixin, dj.Computed):
    """
    Where each theta cycle's sweep went: the rat's location and speed, and the sweep's target.

    One row per theta cycle, like ThetaSweeps, but describing the maze rather than the shape of
    the trajectory: the hex the rat was in, how fast it was going, which hexes the decode passed
    through and which hex it reached furthest ahead and furthest behind.

    Deliberately NOT downstream of ThetaSweeps -- it cuts its own cycles from the same reference
    phase with the same sweep_start_times, so sweep_number means the same thing in both and the
    two can be joined on it, but neither has to be populated for the other. The cost of that
    independence is that the three parameters are recorded here again rather than read off a
    ThetaSweeps row; if they ever diverge, the stored values say so.

    Typical usage:
        key = {"nwb_file_name": "Toby20250316_.nwb", "epoch": 1,
               "reference_name": "theta_ref_1"}
        ThetaSweepStats().populate(key, display_progress=True)
        stats = (ThetaSweepStats & key).fetch1_dataframe()

        # what hexes do sweeps reach, from each hex the rat stands in
        stats.groupby(["rat_hex_start", "ahead_hex"]).size()
    """

    definition = """
    -> HexMazeThetaReference
    -> HexMazeDecodedPositionHexV2
    ---
    -> custom_AnalysisNwbfile
    sweep_stats_object_id: varchar(128)
    median_filter_samples: int  # median filter width applied to the decoded columns
    max_spatial_cov: float      # decodes at or above this coverage were masked out
    jump_threshold_cm: float    # passed to repair_lateral_sign_flips
    """

    @property
    def key_source(self):
        # Same problem ThetaSweeps has: the theta reference is keyed by target_interval_list_name
        # and the decode by epoch number, so the default join would pair a reference with every
        # decode of the session, other epochs included. TaskEpoch maps interval name -> epoch.
        epoch_of_interval = TaskEpoch.proj(target_interval_list_name="interval_list_name")
        return (HexMazeThetaReference * epoch_of_interval * HexMazeDecodedPositionHexV2).proj()

    def make(self, key):
        # Skip if already populated
        if self & key:
            return

        reference_phase = (HexMazeThetaReference & key).fetch1_reference_phase()
        decode_df = (HexMazeDecodedPositionHexV2 & key).fetch1_dataframe()
        trials = trial_context(key["nwb_file_name"], key["epoch"])

        stats = compute_sweep_stats(reference_phase, decode_df, trials=trials,
                                    median_filter_samples=MEDIAN_FILTER_SAMPLES,
                                    max_spatial_cov=MAX_SPATIAL_COV,
                                    jump_threshold_cm=JUMP_THRESHOLD_CM)
        key["median_filter_samples"] = MEDIAN_FILTER_SAMPLES
        key["max_spatial_cov"] = MAX_SPATIAL_COV
        key["jump_threshold_cm"] = JUMP_THRESHOLD_CM

        # Save sweep_number as a column instead of the index (NWB requires an integer index)
        stats = stats.reset_index()

        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["sweep_stats_object_id"] = builder.add_nwb_object(stats, "theta_sweep_stats")
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_dataframe(self):
        """The per-cycle stats for one entry, indexed by sweep_number."""
        # fetch_nwb keys the result on the attribute name minus "_object_id", so this is
        # sweep_stats -- NOT "theta_sweep_stats", which is the object's name inside the NWB file
        return self.fetch_nwb()[0]["sweep_stats"].set_index("sweep_number")
