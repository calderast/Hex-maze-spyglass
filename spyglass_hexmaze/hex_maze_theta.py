"""
Theta-band LFP analysis for the hex maze task.

Split out of hex_maze_decoding.py because these tables are LFP signal processing not decoding.
They stay in the same DataJoint schema ("hex_maze_decoding") so the existing database tables are unchanged.

hex_maze_decoding re-exports both tables, so
    from spyglass_hexmaze.hex_maze_decoding import HexMazeThetaV1
keeps working.
"""

import h5py
import datajoint as dj
import numpy as np
import pandas as pd

import spyglass.common as sgc
from spyglass.common import AnalysisNwbfile
from spyglass.common.custom_nwbfile import AnalysisNwbfile as custom_AnalysisNwbfile
import spyglass.lfp as lfp
from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1, LFPBandSelection
from spyglass.utils import SpyglassMixin

# Same schema name as hex_maze_decoding because these tables already exist
# in that database schema (changing the name here would orphan them)
schema = dj.schema("hex_maze_decoding")

# Where make() puts the analytic signal inside the analysis NWB file. An NWB file is an HDF5
# file. "scratch" is NWB's standard spot for working data that has no formal neurodata type,
# and "analytic_signal" is the name make() passes to add_nwb_object. 
# HDF5 stores the table as a group holding one dataset (one flat array) per column, 
# which lets us read a single electrode without touching the others (faster)
# If spyglass ever changes where add_nwb_object puts things, we will have to change this.
STORED_GROUP = "scratch/analytic_signal"

# The theta band every session is filtered to. FIR band edges are [stop, pass, pass, stop]
# in Hz, so 5-11 Hz is the passband and 4/12 Hz are where the filter has fully rolled off
# (which is why the name says 5-11 but the edges show 4 and 12).
THETA_FILTER_NAME = "Theta 5-11 Hz"
THETA_BAND_EDGES = [4, 5, 11, 12]


@schema
class HexMazeThetaV1(SpyglassMixin, dj.Computed):
    """
    Computes the theta-band analytic signal from LFPBandV1 and saves it to an analysis
    NWB file. 

    Theta phase and theta power are not stored, because they are exact functions of the
    analytic signal z:
        phase = np.mod(np.angle(z), 2π)      power = |z|²
    
    We generally only care about power/phase for a select set of reference electrodes instead
    of all of them. Use fetch_theta_phase() / fetch_theta_power(), to derive them on read, 
    or average_theta_phase() / average_theta_power() for a set of electrodes.

    The analytic signal is stored as real + imaginary float columns because
    HDF5/NWB can't store complex dtype directly. To reconstruct:
        z = df['electrode 5_real'] + 1j * df['electrode 5_imag']

    PHASE CONVENTION: phase is peak-referenced per hippocampal convention, in radians [0, 2π]: 
    0/2π = LFP peak, π = trough. CA1 pyramidal cells fire near π, the theta trough
    (so MUA peaks around π)
    
    NOTE spyglass's LFPBandV1.compute_signal_phase returns np.angle(z) + π instead, 
    which puts the trough at 0 (shifted half a cycle from everything here!) so don't use that!!

    Dependencies (all must be populated before running make):
        LFPElectrodeGroup  →  LFPSelection  →  LFPV1
        LFPBandSelection  →  LFPBandV1

    Use HexMazeThetaV1.setup_theta_pipeline() to create all those entries,
    then call LFPBandV1().populate(lfp_band_key) before populating this table.
    """

    definition = """
    -> LFPBandV1
    ---
    -> custom_AnalysisNwbfile
    analytic_signal_object_id : varchar(128)   # real+imag parts of Hilbert transform
    """

    def make(self, key):
        # Skip if already populated
        if self & key:
            return
        # Get the list of electrodes selected for this LFP band entry
        electrode_ids = sorted(
            (LFPBandSelection.LFPBandElectrode & key).fetch("electrode_id").tolist()
        )

        lfp_band_entry = LFPBandV1 & key

        # Compute the complex analytic signal
        # DataFrame is (n_timesteps × n_electrodes), time-indexed, columns "electrode {id}".
        analytic_df = lfp_band_entry.compute_analytic_signal(electrode_list=electrode_ids)

        # Split complex analytic signal into real and imaginary float columns
        # so the DataFrame can be stored as HDF5 (NWB doesn't support complex dtype).
        # Use .values.real/.values.imag, since pandas DataFrame has no .real/.imag directly.
        # Real columns: "electrode 5_real", imaginary: "electrode 5_imag"
        analytic_stored = pd.concat(
            [
                pd.DataFrame(
                    analytic_df.values.real,
                    index=analytic_df.index,
                    columns=[c + "_real" for c in analytic_df.columns],
                ),
                pd.DataFrame(
                    analytic_df.values.imag,
                    index=analytic_df.index,
                    columns=[c + "_imag" for c in analytic_df.columns],
                ),
            ],
            axis=1,
        ).reset_index()  # move time index to column (NWB requires integer index)

        # Save it to an analysis NWB file
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["analytic_signal_object_id"] = builder.add_nwb_object(
                analytic_stored, "analytic_signal"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def _stored_path(self) -> str:
        """Absolute path of the analysis NWB file holding this entry's analytic signal.

        Returns:
            str: Path on disk. fetch1 also enforces that the restriction picks exactly one entry.
        """
        return AnalysisNwbfile.get_abs_path(self.fetch1("analysis_file_name"))

    def _read_stored_columns(self, column_names: list) -> dict:
        """Read named columns out of the stored table without loading the whole thing.

        The analytic signal is two float64 columns per electrode, so one epoch of a
        384-channel session is ~21 GB and takes minutes to load whole from /stelmo. Sad.
        But HDF5 stores each column as its own dataset, so reading just the columns
        we need instead takes under a second! yay!

        Parameters:
            column_names (list[str]): Column names to read, e.g. ["time", "electrode 41_real"]

        Returns:
            dict: {column_name: np.ndarray} for each requested column.
        """
        with h5py.File(self._stored_path(), "r") as f:
            group = f[STORED_GROUP]
            return {column: group[column][:] for column in column_names}

    def get_electrode_ids(self) -> list:
        """Which electrodes this entry actually has theta for.

        Note this is the electrodes THETA WAS COMPUTED FOR, which is not necessarily every
        electrode in the session -- for that, see spikesorting_helpers.get_electrode_ids().

        Returns:
            list[int]: Electrode ids present in the stored analytic signal, sorted.
        """
        with h5py.File(self._stored_path(), "r") as f:
            return sorted(int(c.split()[-1].rsplit("_", 1)[0])
                          for c in f[STORED_GROUP] if c.endswith("_real"))

    def fetch_time(self) -> np.ndarray:
        """Theta sample times for this entry, in seconds.

        Returns:
            np.ndarray: Sample times in seconds, on the session clock.
        """
        return self._read_stored_columns(["time"])["time"]

    def _resolve_electrodes(self, electrode_ids) -> list:
        """Normalize an electrode argument to a list of ints, with None meaning ALL."""
        if electrode_ids is None:
            return self.get_electrode_ids()
        return [int(e) for e in electrode_ids]

    def fetch_analytic_signal(self, electrode_ids=None) -> pd.DataFrame:
        """Analytic signal for the given electrodes, indexed by time.

        Reads only the columns you ask for rather than the whole table (WAY faster), so
        always name the electrodes you want if you know them.

        Parameters:
            electrode_ids (list): Electrode ids to read. Ints or strings ("57") both work.
                Default None = every electrode this entry has.

        Returns:
            pd.DataFrame: Time-indexed, with 'electrode N_real' / 'electrode N_imag' columns for each electrode.
        """
        electrode_ids = self._resolve_electrodes(electrode_ids)
        wanted = [f"electrode {e}_{part}" for e in electrode_ids for part in ("real", "imag")]
        data = self._read_stored_columns(wanted)
        return pd.DataFrame(data, index=pd.Index(self.fetch_time(), name="time"))

    def fetch_complex(self, electrode_ids=None) -> pd.DataFrame:
        """Complex analytic signal for the given electrodes, indexed by time.

        Gets the stored _real and _imag columns for these electrodes and puts them back together
        as complex numbers. (This is what phase and power are both built from)

        Parameters:
            electrode_ids (list): Electrode ids to read. Ints or strings ("57") both work.
                Default None = every electrode this entry has.

        Returns:
            pd.DataFrame: Time-indexed complex values, with an 'electrode N' column per electrode.
        """
        electrode_ids = self._resolve_electrodes(electrode_ids)
        analytic = self.fetch_analytic_signal(electrode_ids)
        return pd.DataFrame(
            {f"electrode {e}": (analytic[f"electrode {e}_real"].to_numpy()
                                + 1j * analytic[f"electrode {e}_imag"].to_numpy())
             for e in electrode_ids},
            index=analytic.index,
        )

    def fetch_theta_phase(self, electrode_ids=None) -> pd.DataFrame:
        """Theta phase for the given electrodes, indexed by time.

        Reads only the columns you ask for rather than the whole table (WAY faster), so
        always name the electrodes you want if you know them.

        Parameters:
            electrode_ids (list): Electrode ids to read. Ints or strings ("57") both work.
                Default None = every electrode this entry has.

        Returns:
            pd.DataFrame: Time-indexed, with an 'electrode N' column per requested electrode.
                Phase is in radians [0, 2π], peak-referenced (0/2π = LFP PEAK, π = trough),
                derived from the analytic signal as np.mod(np.angle(z), 2π).
        """
        z = self.fetch_complex(electrode_ids)
        return pd.DataFrame(np.mod(np.angle(z), 2 * np.pi), index=z.index, columns=z.columns)

    def fetch_theta_power(self, electrode_ids=None) -> pd.DataFrame:
        """Theta power for the given electrodes, indexed by time.

        Reads only the columns you ask for rather than the whole table (WAY faster), so
        always name the electrodes you want if you know them.

        Parameters:
            electrode_ids (list): Electrode ids to read. Ints or strings ("57") both work.
                Default None = every electrode this entry has.

        Returns:
            pd.DataFrame: Time-indexed, with an 'electrode N' column per requested electrode.
                Power is amplitude squared, derived from the analytic signal as |z|².
        """
        z = self.fetch_complex(electrode_ids)
        return pd.DataFrame(np.abs(z.to_numpy()) ** 2, index=z.index, columns=z.columns)

    def iter_complex(self, electrode_ids=None):
        """Yield (electrode_id, z) one electrode at a time, over a single open file.

        For anything that reduces every electrode down to a few numbers (per-electrode
        power, spike phase locking, correlations against a reference, etc). Memory stays at one
        electrode (~100 MB per hour-long epoch) instead of the tens of GB all of them would
        take. It is also ~2x faster than calling fetch_complex() per electrode, because
        it hands back plain numpy instead of creating a dataframe (the slow part)

        Parameters:
            electrode_ids (list): Electrode ids to read, in the order you want them. Ints or
                strings ("57") both work. Default None = every electrode this entry has.

        Yields:
            tuple: (electrode_id, z) where electrode_id is an int and z is the complex
                analytic signal for that electrode as a plain np.ndarray. 
                Use fetch_time() for the matching sample times (time is the same for all electrodes)
        """
        electrode_ids = self._resolve_electrodes(electrode_ids)
        with h5py.File(self._stored_path(), "r") as f:
            group = f[STORED_GROUP]
            for electrode in electrode_ids:
                yield electrode, (group[f"electrode {electrode}_real"][:]
                                  + 1j * group[f"electrode {electrode}_imag"][:])

    def average_analytic_signal(self, electrode_ids) -> pd.Series:
        """Average the complex analytic signal across a set of electrodes, indexed by time.

        Averaging the complex signal (not the phases) is the correct way to combine
        channels: in-phase theta adds, noise cancels. Entries can be ints or strings ("57").
        """
        z = self.fetch_complex(electrode_ids).mean(axis=1)
        z.name = "analytic_signal"
        return z

    def average_theta_phase(self, electrode_ids) -> pd.Series:
        """Averaged theta phase across a set of electrodes, indexed by time.

        Returns:
            pd.Series: Phase in radians [0, 2π], peak-referenced (0/2π = LFP PEAK, π =trough)
        """
        z = self.average_analytic_signal(electrode_ids)
        return pd.Series(np.mod(np.angle(z), 2 * np.pi), index=z.index,
                         name="theta_phase")

    def average_theta_power(self, electrode_ids) -> pd.Series:
        """Averaged theta power across a set of electrodes, indexed by time.

        This is the mean of the per-channel power envelopes (average of powers).
        For the power of the combined waveform instead (power of the average,
        which sags if channels drift out of phase) use
        np.abs(self.average_analytic_signal(electrode_ids)) ** 2.
        """
        return self.fetch_theta_power(electrode_ids).mean(axis=1).rename("theta_power")

    def get_high_theta_intervals(
        self,
        electrode_ids,
        threshold_percentile: float = 75.0,
        min_duration: float = 0.1,
    ) -> np.ndarray:
        """Return time intervals where theta power exceeds a percentile threshold.

        Parameters:
            electrode_ids (list): Electrodes to threshold on -- their per-channel theta power
                is averaged first, which is robust to small inter-channel phase differences.
                Pass your saved reference set (e.g. stratum radiatum); pass [57] for a single
                electrode. Ints or strings ("57") both work.
            threshold_percentile (float): Percentile of the theta power distribution to use
                as the threshold. Default 75 = top quartile of theta power.
            min_duration (float): Minimum interval duration in seconds. Intervals shorter
                than this are discarded. Default 0.1 s.

        Returns:
            np.ndarray: Array of shape (N, 2) of [start_time, end_time] rows in the Spyglass
                valid_times format. Empty (0, 2) array if no intervals pass the threshold.

        Examples:
        # Threshold on a saved reference set, top 25% of theta power
        high_theta = (HexMazeThetaV1 & key).get_high_theta_intervals([57, 58, 65])

        # Filter a dataframe to high-theta times
        mask = np.zeros(len(df), dtype=bool)
        for start, end in high_theta:
            mask |= (df.index >= start) & (df.index <= end)
        df_high_theta = df[mask]
        """
        power_series = self.average_theta_power(electrode_ids)
        power = power_series.to_numpy(dtype=float)
        times = power_series.index.to_numpy()

        # Threshold at the requested percentile
        threshold = float(np.nanpercentile(power, threshold_percentile))
        above = power >= threshold

        # Find contiguous runs where power is above threshold.
        # Pad with False at both ends so np.diff catches edges correctly.
        padded = np.concatenate([[False], above, [False]])
        diff = np.diff(padded.astype(int))
        # diff==1: False→True transition, index = start of run in `times`
        # diff==-1: True→False transition, index = exclusive end of run in `times`
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        intervals = []
        for s, e in zip(starts, ends):
            t_start = times[s]
            t_end = times[e - 1]  # last sample in the run
            if (t_end - t_start) >= min_duration:
                intervals.append([t_start, t_end])

        if not intervals:
            return np.empty((0, 2))
        return np.array(intervals)


    @classmethod
    def setup_theta_pipeline(
        cls,
        nwb_file_name: str,
        lfp_electrode_group_name: str,
        electrode_ids: list,
        interval_list_name: str,
        target_sampling_rate: int = 1000,
        lfp_band_sampling_rate: int = 1000,
    ) -> dict:
        """Set up the full LFP → theta pipeline for a given NWB file.

        Creates all upstream entries needed before calling LFPBandV1.populate():
          1. LFP lowpass filter matched to the file's actual raw sampling rate
             (fetched from the Raw table, not hardcoded per session)
          2. LFP electrode group (the set of channels to filter)
          3. LFPSelection entry (links electrode group + interval + filter)
          4. Populates LFPV1, the broadband LFP downsampled to ~target_sampling_rate Hz
             NOTE: This step takes ~2 hours locally for a full recording.
          5. Theta bandpass filter matched to the actual LFP output sampling rate
          6. LFPBandSelection entry (links LFP output + theta filter + interval)

        Parameters:
            nwb_file_name (str): NWB file to process (must already be inserted in the database)
            lfp_electrode_group_name (str): Name for the LFP electrode group. Arbitrary,
                but must be unique per file
            electrode_ids (list[int]): Electrode IDs (from the electrode table) to include
                in LFP processing
            interval_list_name (str): Name of the valid time interval to process
                (e.g. "00_r1" for Berke lab data)
            target_sampling_rate (int): Desired output rate for the broadband LFP in Hz.
                Default 1000. The actual rate may differ slightly due to integer decimation
                (e.g. raw 29998 Hz with target 1000 → decimation=29 → actual 1034 Hz).
            lfp_band_sampling_rate (int): Desired output rate for the theta-band LFP in Hz.
                Default 1000.

        The theta band itself is fixed (see THETA_FILTER_NAME / THETA_BAND_EDGES).

        Returns:
            dict: LFPBandSelection key. Pass this to LFPBandV1().populate() to run theta
                filtering, then to HexMazeThetaV1().populate() to compute and store results
        """
        # Get the actual raw sampling rate for this recording from the database. This varies per session
        # (e.g. 29998 Hz for IM-1478 instead of the exact 30000 Hz), so we need to fetch it for each one
        raw_sampling_rate = int(
            np.round((sgc.Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate"))
        )

        # The standard set of filters is designed for 30000 Hz data.
        # We copy the band edges from the standard LFP filter and re-design it at the
        # actual sampling rate, so cutoff frequencies are the same but filter coefficients differ
        sgc.FirFilterParameters().create_standard_filters()
        standard_filter = sgc.FirFilterParameters() & {
            "filter_name": "LFP 0-400 Hz",
            "filter_sampling_rate": 30000,
        }
        lfp_band_edges = standard_filter.fetch1("filter_band_edges")

        lfp_filter_name = f"LFP 0-400 Hz {raw_sampling_rate}Hz"
        if not (
            sgc.FirFilterParameters
            & {"filter_name": lfp_filter_name, "filter_sampling_rate": raw_sampling_rate}
        ):
            sgc.FirFilterParameters().add_filter(
                filter_name=lfp_filter_name,
                fs=raw_sampling_rate,
                filter_type="lowpass",
                band_edges=lfp_band_edges,
                comments=f"Standard LFP 0-400 Hz filter adapted for {raw_sampling_rate} Hz data",
            )

        # Create the LFP electrode group (which channels to compute LFP for)
        lfp.lfp_electrode.LFPElectrodeGroup.create_lfp_electrode_group(
            nwb_file_name=nwb_file_name,
            group_name=lfp_electrode_group_name,
            electrode_list=electrode_ids,
            skip_duplicates=True,
        )

        # Insert LFPSelection: links electrode group + valid time interval + filter into one entry
        lfp_s_key = {
            "nwb_file_name": nwb_file_name,
            "lfp_electrode_group_name": lfp_electrode_group_name,
            "target_interval_list_name": interval_list_name,
            "filter_name": lfp_filter_name,
            "filter_sampling_rate": raw_sampling_rate,
            "target_sampling_rate": target_sampling_rate,
        }
        lfp.v1.LFPSelection.insert1(lfp_s_key, skip_duplicates=True)

        # Populate LFPV1: applies the LFP filter and downsamples to ~target_sampling_rate Hz.
        # NOTE: For full recordings this takes ~2 hours when run locally.
        lfp.v1.LFPV1().populate(lfp_s_key)

        # Get the LFP merge ID so we can reference this result downstream
        lfp_key = {
            "merge_id": (lfp.LFPOutput.LFPV1() & lfp_s_key).fetch1("merge_id")
        }

        # The actual LFP sampling rate after integer decimation
        # (may differ from target, e.g. 29998 → 1034 Hz with decimation=29)
        lfp_sampling_rate = int(
            lfp.LFPOutput.merge_get_parent(lfp_key).fetch1("lfp_sampling_rate")
        )

        # Create the theta bandpass filter matched to the actual LFP output sampling rate.
        # We can't reuse a filter entry from a different session if their LFP rates differ.
        if not (
            sgc.FirFilterParameters
            & {"filter_name": THETA_FILTER_NAME, "filter_sampling_rate": lfp_sampling_rate}
        ):
            sgc.common_filter.FirFilterParameters().add_filter(
                THETA_FILTER_NAME,
                lfp_sampling_rate,
                "bandpass",
                THETA_BAND_EDGES,
                f"{THETA_FILTER_NAME} bandpass filter for {lfp_sampling_rate} Hz LFP data",
            )

        # Insert LFPBandSelection: links LFP output + theta filter + valid time interval
        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=nwb_file_name,
            lfp_merge_id=lfp_key["merge_id"],
            electrode_list=electrode_ids,
            filter_name=THETA_FILTER_NAME,
            interval_list_name=interval_list_name,
            reference_electrode_list=[-1],  # -1 means no reference electrode
            lfp_band_sampling_rate=lfp_band_sampling_rate,
        )

        # set_lfp_band_electrodes stores lfp_band_sampling_rate as
        # lfp_sampling_rate // decimation (integer floor division), not the
        # target we passed in. When the LFP rate isn't a clean multiple of the
        # target (e.g. 1034 Hz target 1000 → decimation=1 → stored rate 1034),
        # querying with the target value returns 0 rows.
        actual_lfp_band_rate = lfp_sampling_rate // (lfp_sampling_rate // lfp_band_sampling_rate)

        # Return the LFPBandSelection key.
        # Pass this to LFPBandV1().populate() then HexMazeThetaV1().populate().
        lfp_band_key = (
            LFPBandSelection
            & {
                "lfp_merge_id": lfp_key["merge_id"],
                "filter_name": THETA_FILTER_NAME,
                "lfp_band_sampling_rate": actual_lfp_band_rate,
            }
        ).fetch1("KEY")

        return lfp_band_key


@schema
class HexMazeThetaReference(SpyglassMixin, dj.Manual):
    """A named set of reference electrodes for a HexMazeThetaV1 entry.

    Lets you save the electrodes you chose for a layer (e.g. stratum radiatum) so that
    later you can pull theta phase or power as the AVERAGE of those electrodes, without
    re-running the selection. Averaging is done on the complex analytic signal (the
    correct way to combine phases), then phase and power are derived from that average.

    Typical usage:
        # save a selection (electrode_ids can be ints or strings like "57")
        HexMazeThetaReference.add_reference(
            key={"nwb_file_name": "IM-1478_20220727_.nwb"},
            reference_name="radiatum",
            electrode_ids=[57, 58, 65],
        )

        # later, fetch the averaged theta phase / power for that set
        ref = HexMazeThetaReference & {"nwb_file_name": "IM-1478_20220727_.nwb",
                                       "reference_name": "radiatum"}
        phase = ref.fetch1_reference_phase()   # radians [0, 2π], 0=LFP peak, π=trough
        power = ref.fetch1_reference_power()   # amplitude², indexed by time
    """

    definition = """
    -> HexMazeThetaV1
    reference_name : varchar(64)    # label for this electrode set, e.g. theta_ref_1
    ---
    electrode_ids  : blob           # list of electrode ids to average over
    description="" : varchar(255)   # optional note 
    """

    @classmethod
    def add_reference(cls, key, reference_name, electrode_ids, description="", replace=False):
        """Save a set of reference electrodes for one HexMazeThetaV1 entry.

        Parameters:
            key (dict): Restriction that uniquely identifies one HexMazeThetaV1 row
                (e.g. {"nwb_file_name": "IM-1478_20220727_.nwb"})
            reference_name (str): Label for this set (e.g. "radiatum"). Part of the primary
                key, so you can store several named sets per session.
            electrode_ids (list): Electrode ids to average over. Ints or strings ("57")
                both work.
            description (str): Optional note (e.g. "k=6 cluster 0, stratum radiatum")
            replace (bool): If True, overwrite an existing set stored under the same name
        """
        theta_key = (HexMazeThetaV1 & key).fetch1("KEY")
        cls.insert1(
            {
                **theta_key,
                "reference_name": reference_name,
                "electrode_ids": [int(e) for e in electrode_ids],
                "description": description,
            },
            replace=replace,
        )

    def fetch1_reference_analytic(self) -> pd.Series:
        """Complex analytic signal averaged across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_analytic_signal(self.fetch1("electrode_ids"))

    def fetch1_reference_phase(self) -> pd.Series:
        """Averaged theta phase across the saved reference electrodes, indexed by time.

        Returns:
            pd.Series: Phase in radians [0, 2π], peak-referenced (0/2π = LFP peak, π =
                trough), matching fetch_theta_phase().
        """
        return (HexMazeThetaV1 & self).average_theta_phase(self.fetch1("electrode_ids"))

    def fetch1_reference_power(self) -> pd.Series:
        """Averaged theta power (amplitude²) across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_theta_power(self.fetch1("electrode_ids"))


def circular_mean(angles_rad) -> tuple:
    """Circular mean of a set of angles.

    Parameters:
        angles_rad (array-like): Angles in radians.

    Returns:
        tuple: (mean angle in degrees [0, 360), mean resultant length R, Rayleigh p)
            R is 1 when every angle is identical and 0 when they are spread evenly.
            Rayleigh p tests "the angles are uniform" (Zar's approximation), so a small p
            means the angles really do cluster at the mean angle.
    """
    angles_rad = np.asarray(angles_rad, dtype=float)
    n = angles_rad.size
    if n == 0:
        return np.nan, np.nan, np.nan
    resultant = np.mean(np.exp(1j * angles_rad))
    R = np.abs(resultant)
    Z = n * R ** 2
    # The correction term goes negative for very large Z (where p is 0 anyway), so clip
    rayleigh_p = float(np.clip(np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n)), 0.0, 1.0))
    return np.degrees(np.angle(resultant)) % 360, R, rayleigh_p


def spike_phase_by_electrode(
    units: pd.DataFrame,
    theta_phase: pd.DataFrame,
    speed: pd.Series = None,
    speed_threshold: float = 5.0,
    min_spikes: int = 100,
    reference_column: str = None,
    sort_group_electrodes: dict = None,
) -> pd.DataFrame:
    """Theta phase at which spiking peaks on each electrode.

    Pools the spikes of every unit sitting on an electrode and takes the circular mean of
    theta phase at those spike times. Mostly useful for working out where on the probe we
    are: CA1 pyramidal cells fire near the TROUGH (180 deg) of local pyramidal-layer theta,
    so electrodes whose spikes are strongly locked near 180 deg of their own theta are in
    or near the cell layer, and the phase measured in a candidate reference electrode's
    theta tells you what phase spikes would come out at if you picked that reference.

    Units are assigned to electrodes by their peak channel. v0 (Frank lab) units have no
    peak channel, so if `sort_group_electrodes` is given, each v0 unit's spikes are counted
    on every electrode of its sort group instead (i.e. per tetrode, not per electrode).

    Parameters:
        units (pd.DataFrame): Good units, from `fetch_good_units` (needs spike_times and
            peak_channel, plus sort_group_id when using `sort_group_electrodes`)
        theta_phase (pd.DataFrame): Theta phase in radians indexed by time, with one
            "electrode N" column per electrode (from `HexMazeThetaV1.fetch1_theta_phase`)
        speed (pd.Series): Optional speed (cm/s) indexed by time. If given, only spikes
            that happen while the rat is running are used (theta is only clean then)
        speed_threshold (float): cm/s defining running, only used when `speed` is given
        min_spikes (int): Electrodes with fewer running spikes than this are skipped, since
            a mean phase from a handful of spikes is meaningless
        reference_column (str): Optional theta_phase column (e.g. "electrode 41") to also
            measure every electrode's spike phase in, for comparing electrodes probe-wide
        sort_group_electrodes (dict): Optional sort_group_id -> electrode IDs, from
            `spikesorting_helpers.sort_group_electrodes`. Only used for units with no peak
            channel (v0 sessions)

    Returns:
        pd.DataFrame: One row per electrode that cleared `min_spikes`, indexed by electrode
            name as a STRING (matching the theta tables and `fetch_electrode_geometry`),
            sorted numerically, with columns:
            n_units (int): Units contributing spikes to this electrode
            n_spikes (int): Spikes used (after the running filter)
            spike_phase_deg (float): Phase of peak spiking in this electrode's OWN theta,
                in degrees [0, 360) where 0 = theta peak and 180 = theta trough
            spike_phase_R (float): Phase locking strength, 0 (no preference) to 1 (all
                spikes at the same phase). Low R means spike_phase_deg means little
            spike_phase_p (float): Rayleigh p for that phase preference
            spike_phase_ref_deg / spike_phase_ref_R / spike_phase_ref_p: same three
                measures in the reference electrode's theta (only if `reference_column`)
    """
    phase_times = theta_phase.index.to_numpy()

    # Spike times of all the units sitting on each electrode
    spikes_by_electrode = {}
    units_by_electrode = {}
    for _, unit in units.iterrows():
        peak_channel = unit.get("peak_channel", np.nan)
        if not pd.isna(peak_channel):
            electrodes = [int(peak_channel)]
        elif sort_group_electrodes and not pd.isna(unit.get("sort_group_id", np.nan)):
            # No peak channel (v0): count this unit on every electrode of its sort group
            electrodes = sort_group_electrodes.get(int(unit["sort_group_id"]), [])
        else:
            continue
        spike_times = np.asarray(unit["spike_times"], dtype=float)
        for electrode in electrodes:
            spikes_by_electrode.setdefault(str(electrode), []).append(spike_times)
            units_by_electrode[str(electrode)] = units_by_electrode.get(str(electrode), 0) + 1

    rows = []
    for electrode, spike_time_arrays in spikes_by_electrode.items():
        column = f"electrode {electrode}"
        if column not in theta_phase.columns:   # electrode has units but no theta (e.g. bad channel)
            continue
        spike_times = np.sort(np.concatenate(spike_time_arrays))

        # Keep spikes inside the theta time range (which may only cover one run epoch)
        spike_times = spike_times[(spike_times >= phase_times[0]) & (spike_times <= phase_times[-1])]
        if speed is not None and spike_times.size:
            spike_times = spike_times[
                speed.reindex(spike_times, method="nearest").values >= speed_threshold]
        if spike_times.size < min_spikes:
            continue

        # Theta phase at each spike time (nearest theta sample: theta is ~1 kHz here, so
        # each sample is ~3 deg of a theta cycle, well below the spread we care about)
        after = np.clip(np.searchsorted(phase_times, spike_times), 1, len(phase_times) - 1)
        nearest = np.where(spike_times - phase_times[after - 1] <= phase_times[after] - spike_times,
                           after - 1, after)

        local_deg, local_R, local_p = circular_mean(theta_phase[column].values[nearest])
        row = {"electrode": electrode,
               "n_units": units_by_electrode[electrode],
               "n_spikes": int(spike_times.size),
               "spike_phase_deg": local_deg,
               "spike_phase_R": local_R,
               "spike_phase_p": local_p}
        if reference_column is not None:
            ref_deg, ref_R, ref_p = circular_mean(theta_phase[reference_column].values[nearest])
            row.update({"spike_phase_ref_deg": ref_deg,
                        "spike_phase_ref_R": ref_R,
                        "spike_phase_ref_p": ref_p})
        rows.append(row)

    columns = ["electrode", "n_units", "n_spikes", "spike_phase_deg", "spike_phase_R", "spike_phase_p"]
    if reference_column is not None:
        columns += ["spike_phase_ref_deg", "spike_phase_ref_R", "spike_phase_ref_p"]
    spike_phase = pd.DataFrame(rows, columns=columns).set_index("electrode")
    return spike_phase.loc[sorted(spike_phase.index, key=int)]
