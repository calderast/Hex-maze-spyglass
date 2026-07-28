"""
Theta-band LFP analysis for the hex maze task.

Split out of hex_maze_decoding.py because these tables are LFP signal processing not decoding.
They stay in the same DataJoint schema ("hex_maze_decoding") so the existing database tables are unchanged.

hex_maze_decoding re-exports both tables, so
    from spyglass_hexmaze.hex_maze_decoding import HexMazeThetaV1
keeps working.
"""

import datajoint as dj
import numpy as np
import pandas as pd

import spyglass.common as sgc
from spyglass.common.custom_nwbfile import AnalysisNwbfile as custom_AnalysisNwbfile
from spyglass.lfp.analysis.v1.lfp_band import LFPBandV1, LFPBandSelection
from spyglass.utils import SpyglassMixin

# Same schema name as hex_maze_decoding because these tables already exist 
# in that database schema (changing the name here would orphan them)
schema = dj.schema("hex_maze_decoding")


@schema
class HexMazeThetaV1(SpyglassMixin, dj.Computed):
    """
    Computes theta-band analytic signal, phase, and power from LFPBandV1.
    Saves all three to a single analysis NWB file.

    The analytic signal is stored as real + imaginary float columns because
    HDF5/NWB can't store complex dtype directly. To reconstruct:
        z = df['electrode 5_real'] + 1j * df['electrode 5_imag']

    Phase is in radians, shifted to [0, 2π] (matching spyglass convention).
    Power is amplitude squared (|analytic signal|²).

    Dependencies (all must be populated before running make):
        LFPElectrodeGroup  →  LFPSelection  →  LFPV1
        LFPBandSelection  →  LFPBandV1

    Use HexMazeThetaV1.setup_theta_pipeline() to create all those entries,
    then call LFPBandV1().populate(lfp_band_key) before populating this table.

    Typical usage:
        # Step 1: set up the pipeline and populate broadband LFP (~2h locally)
        lfp_band_key = HexMazeThetaV1.setup_theta_pipeline(
            nwb_file_name="IM-1478_20220726_.nwb",
            lfp_electrode_group_name="my_lfp_group",
            electrode_ids=[0, 1, 2, ...],
            interval_list_name="00_r1",
        )
        # Step 2: populate theta-band LFP
        LFPBandV1().populate(lfp_band_key)
        # Step 3: compute and store analytic signal, phase, power
        HexMazeThetaV1().populate(lfp_band_key)
    """

    definition = """
    -> LFPBandV1
    ---
    -> custom_AnalysisNwbfile
    analytic_signal_object_id : varchar(128)   # real+imag parts of Hilbert transform
    theta_phase_object_id     : varchar(128)   # instantaneous phase [0, 2π] in radians
    theta_power_object_id     : varchar(128)   # instantaneous power (amplitude²)
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

        # Compute analytic signal (complex), instantaneous phase, and power.
        # Each DataFrame has shape (n_timesteps × n_electrodes), time-indexed.
        # Column names are "electrode {electrode_id}" for each electrode.
        analytic_df = lfp_band_entry.compute_analytic_signal(electrode_list=electrode_ids)
        phase_df = lfp_band_entry.compute_signal_phase(electrode_list=electrode_ids)
        power_df = lfp_band_entry.compute_signal_power(electrode_list=electrode_ids)

        # Split complex analytic signal into real and imaginary float columns
        # so the DataFrame can be stored as HDF5 (NWB doesn't support complex dtype).
        # Use .values.real/.values.imag — pandas DataFrame has no .real/.imag directly.
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

        phase_stored = phase_df.reset_index()
        power_stored = power_df.reset_index()

        # Save all three DataFrames to a single analysis NWB file
        with custom_AnalysisNwbfile().build(key["nwb_file_name"]) as builder:
            key["analytic_signal_object_id"] = builder.add_nwb_object(
                analytic_stored, "analytic_signal"
            )
            key["theta_phase_object_id"] = builder.add_nwb_object(
                phase_stored, "theta_phase"
            )
            key["theta_power_object_id"] = builder.add_nwb_object(
                power_stored, "theta_power"
            )
            key["analysis_file_name"] = builder.analysis_file_name

        self.insert1(key, skip_duplicates=True)

    def fetch1_analytic_signal(self) -> pd.DataFrame:
        """Return the analytic signal DataFrame, indexed by time.

        Columns are 'electrode N_real' and 'electrode N_imag' for each electrode N.
        To reconstruct complex signal: df['electrode 5_real'] + 1j * df['electrode 5_imag']
        """
        return self.fetch_nwb()[0]["analytic_signal"].set_index("time")

    def fetch1_theta_phase(self) -> pd.DataFrame:
        """Return instantaneous theta phase in radians [0, 2π], indexed by time.

        Columns are 'electrode N' for each electrode N.
        """
        return self.fetch_nwb()[0]["theta_phase"].set_index("time")

    def fetch1_theta_power(self) -> pd.DataFrame:
        """Return instantaneous theta power (amplitude²), indexed by time.

        Columns are 'electrode N' for each electrode N.
        """
        return self.fetch_nwb()[0]["theta_power"].set_index("time")

    def average_analytic_signal(self, electrode_list) -> pd.Series:
        """Average the complex analytic signal across a set of electrodes, indexed by time.

        Averaging the complex signal (not the phases) is the correct way to combine
        channels: in-phase theta adds, noise cancels. electrode_list entries can be
        ints or strings ("57").
        """
        analytic = self.fetch1_analytic_signal()
        z = sum(
            analytic[f"electrode {int(e)}_real"] + 1j * analytic[f"electrode {int(e)}_imag"]
            for e in electrode_list
        ) / len(electrode_list)
        z.name = "analytic_signal"
        return z

    def average_theta_phase(self, electrode_list) -> pd.Series:
        """Averaged theta phase (radians [0, 2π]) across a set of electrodes, indexed by time."""
        z = self.average_analytic_signal(electrode_list)
        return pd.Series(np.mod(np.angle(z), 2 * np.pi), index=z.index, name="theta_phase")

    def average_theta_power(self, electrode_list) -> pd.Series:
        """Averaged theta power across a set of electrodes, indexed by time.

        This is the mean of the per-channel power envelopes (average of powers), which is
        robust to small phase differences between channels. For the *coherent* power of the
        combined waveform instead (power of the average, which sags if channels drift out
        of phase) use np.abs(self.average_analytic_signal(electrode_list)) ** 2.
        """
        power_df = self.fetch1_theta_power()
        columns = [f"electrode {int(e)}" for e in electrode_list]
        return power_df[columns].mean(axis=1).rename("theta_power")

    def get_high_theta_intervals(
        self,
        threshold_percentile: float = 75.0,
        electrode=None,
        min_duration: float = 0.1,
    ) -> np.ndarray:
        """Return time intervals where theta power exceeds a percentile threshold.

        Parameters:
            threshold_percentile (float): Percentile of the theta power distribution to use
                as the threshold. Default 75 = top quartile of theta power.
            electrode (str or list, optional): Which signal to threshold:
                - a single electrode as 3, "3", or "electrode 3": that one electrode.
                - a list/set of electrode ids (ints or "57" strings): average the per-channel
                  theta power across them — use this to threshold on a saved reference set
                  (e.g. stratum radiatum).
                - None (default): average power across all electrodes.
            min_duration (float): Minimum interval duration in seconds. Intervals shorter
                than this are discarded. Default 0.1 s.

        Returns:
            np.ndarray: Array of shape (N, 2) of [start_time, end_time] rows in the Spyglass
                valid_times format. Empty (0, 2) array if no intervals pass the threshold.

        Examples:
        # Average a set of reference electrodes (e.g. a saved radiatum set), top 25%
        high_theta = (HexMazeThetaV1 & key).get_high_theta_intervals(electrode=[57, 58, 65])

        # Use a single electrode
        high_theta = (HexMazeThetaV1 & key).get_high_theta_intervals(electrode="electrode 57")

        # Filter a dataframe to high-theta times
        mask = np.zeros(len(df), dtype=bool)
        for start, end in high_theta:
            mask |= (df.index >= start) & (df.index <= end)
        df_high_theta = df[mask]
        """
        # Select the theta-power signal to threshold
        if isinstance(electrode, (list, tuple, set, np.ndarray)):
            # Average the per-channel theta power across a set of reference electrodes
            # and threshold that (robust to small inter-channel phase differences).
            power_series = self.average_theta_power(electrode)
            power = power_series.to_numpy(dtype=float)
            times = power_series.index.to_numpy()
        else:
            power_df = self.fetch1_theta_power()
            times = power_df.index.to_numpy()
            if electrode is not None:
                # Accept 57, "57", or "electrode 57" -- normalize to the column name
                column = f"electrode {int(str(electrode).split()[-1])}"
                power = power_df[column].to_numpy(dtype=float)
            else:
                # Average across all electrodes (not recommended for sessions with
                # many channels from mixed brain regions)
                power = power_df.mean(axis=1).to_numpy(dtype=float)

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
        theta_filter_name: str = "Theta 5-11 Hz",
        theta_band_edges: list = None,
    ) -> dict:
        """Set up the full LFP → theta pipeline for a given NWB file.

        Creates all upstream entries needed before calling LFPBandV1.populate():
          1. LFP lowpass filter matched to the file's actual raw sampling rate
             (fetched from the Raw table — no hardcoding of session-specific rates)
          2. LFP electrode group (the set of channels to filter)
          3. LFPSelection entry (links electrode group + interval + filter)
          4. Populates LFPV1 — broadband LFP downsampled to ~target_sampling_rate Hz
             NOTE: This step takes ~2 hours locally for a full recording.
          5. Theta bandpass filter matched to the actual LFP output sampling rate
          6. LFPBandSelection entry (links LFP output + theta filter + interval)

        Parameters:
            nwb_file_name (str): NWB file to process (must already be inserted in the database)
            lfp_electrode_group_name (str): Name for the LFP electrode group — arbitrary,
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
            theta_filter_name (str): Name for the theta bandpass filter entry in
                FirFilterParameters
            theta_band_edges (list[float], optional): FIR band edges in Hz:
                [lo_pass, lo_stop, hi_stop, hi_pass]. Defaults to [4, 5, 11, 12]
                (standard theta band).

        Returns:
            dict: LFPBandSelection key — pass this to LFPBandV1().populate() to run theta
                filtering, then to HexMazeThetaV1().populate() to compute and store results
        """
        import spyglass.lfp as lfp

        if theta_band_edges is None:
            theta_band_edges = [4, 5, 11, 12]

        # Get the actual raw sampling rate for this recording from the database.
        # This varies per session (e.g. 29998 Hz for IM-1478 instead of the nominal 30000 Hz),
        # so we always fetch it rather than hardcoding.
        raw_sampling_rate = int(
            np.round((sgc.Raw & {"nwb_file_name": nwb_file_name}).fetch1("sampling_rate"))
        )

        # The standard set of filters is designed for 30000 Hz data.
        # We copy the band edges from the standard LFP filter and re-design it at the
        # actual sampling rate, so cutoff frequencies are the same but filter coefficients differ.
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
            & {"filter_name": theta_filter_name, "filter_sampling_rate": lfp_sampling_rate}
        ):
            sgc.common_filter.FirFilterParameters().add_filter(
                theta_filter_name,
                lfp_sampling_rate,
                "bandpass",
                theta_band_edges,
                f"Theta 5-11 Hz bandpass filter for {lfp_sampling_rate} Hz LFP data",
            )

        # Insert LFPBandSelection: links LFP output + theta filter + valid time interval
        LFPBandSelection().set_lfp_band_electrodes(
            nwb_file_name=nwb_file_name,
            lfp_merge_id=lfp_key["merge_id"],
            electrode_list=electrode_ids,
            filter_name=theta_filter_name,
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
                "filter_name": theta_filter_name,
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
        phase = ref.fetch1_reference_phase()   # radians [0, 2π], indexed by time
        power = ref.fetch1_reference_power()    # amplitude², indexed by time
    """

    definition = """
    -> HexMazeThetaV1
    reference_name : varchar(64)    # label for this electrode set, e.g. "radiatum"
    ---
    electrode_ids  : blob           # list of electrode ids to average over
    description="" : varchar(255)   # optional note about how the set was chosen
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
        """Averaged theta phase (radians [0, 2π]) across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_theta_phase(self.fetch1("electrode_ids"))

    def fetch1_reference_power(self) -> pd.Series:
        """Averaged theta power (amplitude²) across the saved reference electrodes."""
        return (HexMazeThetaV1 & self).average_theta_power(self.fetch1("electrode_ids"))
