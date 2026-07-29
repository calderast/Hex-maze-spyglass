"""
Helpers for reading spike sorting output across across Berke/Frank lab pipelines.
    
Berke lab (IM-*)              ->  spikesorting v1 (SpikeSortingRecordingSelection / CurationV1)
Frank lab (Lily, Toby, etc)   ->  spikesorting v0 (CuratedSpikeSorting)

Some differences between the two:
v1 units have a `curation_label` and a real `peak_channel` (single best electrode)
v0 units have a `label` and NO peak channel at all

These functions handle the differences so analysis code just works for both labs.
Anything keyed on the peak electrode (e.g. marking which electrodes carry units) is 
empty for Frank lab v0 sessions (because they have no peak channel).

NOTE none of this applies to the theta tables in hex_maze_theta. Spyglass LFP is always v1, so
every session goes through LFPBandV1 no matter which lab it came from. The v0 / v1 split is
only for spike sorting.
"""

import numpy as np
import pandas as pd
import pynwb

import spyglass.common as sgc
import spyglass.spikesorting.v1 as sgs1
from spyglass.spikesorting.v0.spikesorting_curation import CuratedSpikeSorting
from spyglass.spikesorting.v0.spikesorting_recording import SortGroup as SortGroupV0

# Unit labels for "not a well isolated single unit"
# Applied to the v1 `curation_label` and the v0 `label`
# We filter by what to reject rather than requiring label == "accept" because 
# it seems like some sessions (e.g. Luna20250218) have an empty label on every unit
BAD_UNIT_LABELS = "noise|reject|mua"

def sorting_version(nwb_file_name: str) -> str:
    """Which spyglass spike sorting pipeline a session was sorted with (based on nwb name).

    Berke lab IM-* sessions are sorted using v1, 
    Frank lab sessions (BraveLu, Lily, Nova, Toby, Luna...) are sorted using v0

    Parameters:
        nwb_file_name (str): NWB file to check (e.g. "Lily20251217_.nwb")

    Returns:
        str: "v1" or "v0".
    """
    return "v1" if nwb_file_name.startswith("IM-") else "v0"


def sort_group_table(nwb_file_name: str):
    """The SortGroup table (v1 or v0) that holds this session's sort groups.

    Parameters:
        nwb_file_name (str): NWB file to look up

    Returns:
        dj.Table: v1 SortGroup for Berke lab sessions, v0 SortGroup for Frank lab.
    """
    return sgs1.SortGroup if sorting_version(nwb_file_name) == "v1" else SortGroupV0


def get_electrode_ids(nwb_file_name: str) -> list:
    """Electrode IDs for a session, taken from its sort groups (so bad channels are excluded).

    Sort groups are built from good channels only, which makes them a convenient source of
    "electrodes worth processing" (this is what we use for theta tables)

    Parameters:
        nwb_file_name (str): NWB file to look up

    Returns:
        list[int]: Electrode IDs in the session's sort groups, in table order.
    """
    SortGroupTable = sort_group_table(nwb_file_name)
    electrodes = pd.DataFrame(
        (SortGroupTable * SortGroupTable.SortGroupElectrode & {"nwb_file_name": nwb_file_name})
        .fetch(as_dict=True))
    if electrodes.empty:
        return []
    return electrodes["electrode_id"].unique().tolist()


def fetch_good_units(
    nwb_file_name: str,
    curation_id: int = None,
    sorter_params_name: str = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch well-isolated units for a session, from whichever pipeline its lab uses.

    Works for Berke lab (v1) and Frank lab (v0) sessions without the caller knowing which is
    which. Units labeled noise / reject / mua are dropped.

    Parameters:
        nwb_file_name (str): NWB file to fetch units for (e.g. "IM-1478_20220726_.nwb")
        curation_id (int): Which curation to read. Default None = the most curated one
            available, except that v1 prefers 1 (not-whitened) over 3 (whitened).
            Conventions: 
            v1 (Berke) 0 = initial, 1 = not-whitened, 3 = whitened.
            v0 (Frank) 1 = auto-curated only, 2 = after manual curation.
        sorter_params_name (str): v0 only. Which sorter parameter set to read. Default None
            = whatever the session has (errors if it has more than one).
        verbose (bool): Print a one-line summary of what was loaded. Default True.

    Returns:
        pd.DataFrame: One row per good unit, with the same columns for either pipeline:
            spike_times (np.ndarray): Spike times in seconds, on the session clock
            unit_label (str): Curation label ("accept", "", etc)
            sort_group_id (int): Sort group (shank) the unit was sorted on
            peak_channel (float): Electrode id of the unit's peak channel, always NaN for v0
                (the v0 pipeline doesn't give us this)
            sorting_version (str): "v0" or "v1", so callers can tell what they got
        Indexed by a plain RangeIndex, since v0 unit ids restart in every sort group.
    """
    version = sorting_version(nwb_file_name)

    if version == "v1":
        units = _fetch_good_units_v1(nwb_file_name, curation_id)
    else:
        units = _fetch_good_units_v0(nwb_file_name, curation_id, sorter_params_name)

    units["sorting_version"] = version

    if verbose:
        print(f"{nwb_file_name}: {len(units)} good units from "
              f"{units['sort_group_id'].nunique()} sort groups ({version})")

    return units.reset_index(drop=True)


def _fetch_good_units_v1(nwb_file_name: str, curation_id: int = None) -> pd.DataFrame:
    """Fetch good units from the v1 pipeline (CurationV1). See `fetch_good_units`."""
    key = {"nwb_file_name": nwb_file_name}

    # proj() keeps only the primary key (drops nwb_file_name, a dependent attribute the
    # v1 sorting tables can't join on)
    recordings = (sgs1.SpikeSortingRecordingSelection & key).proj()
    curated = sgs1.CurationV1 * sgs1.SpikeSortingSelection * recordings

    available = set(curated.fetch("curation_id"))
    if not available:
        raise ValueError(f"No v1 curations found for {nwb_file_name}")
    if curation_id is None:
        # Prefer the not-whitened curation, else the most curated one we have
        curation_id = 1 if 1 in available else max(available)

    # sort_group_id lives on the recording selection, so carry it along with each sorting.
    # fetch() returns one array per attribute, so zip them into (recording_id, sort_group_id) pairs.
    sort_group_by_recording = dict(zip(
        *(sgs1.SpikeSortingRecordingSelection & key).fetch("recording_id", "sort_group_id")))

    frames = []
    for sorting_key in (curated & {"curation_id": curation_id}).fetch(
            "sorting_id", "curation_id", "recording_id", as_dict=True):
        # as_dataframe skips the slow spikeinterface recording rebuild
        units = sgs1.CurationV1.get_sorting(
            {k: sorting_key[k] for k in ("sorting_id", "curation_id")}, as_dataframe=True)
        units["sort_group_id"] = sort_group_by_recording.get(sorting_key["recording_id"], np.nan)
        frames.append(units)

    if not frames:
        raise ValueError(f"No v1 units for {nwb_file_name} at curation_id {curation_id}")

    units = pd.concat(frames)
    units = units[~units["curation_label"].astype(str).str.contains(BAD_UNIT_LABELS, case=False)]

    units = units.rename(columns={"curation_label": "unit_label"})
    return units[["spike_times", "unit_label", "sort_group_id", "peak_channel"]]


def _fetch_good_units_v0(
    nwb_file_name: str, curation_id: int = None, sorter_params_name: str = None
) -> pd.DataFrame:
    """Fetch good units from the v0 pipeline (CuratedSpikeSorting). See `fetch_good_units`."""
    key = {"nwb_file_name": nwb_file_name}

    # Work out which curation / sorter params / sort groups this session actually has, so
    # nothing has to be hardcoded per animal the way the old notebook cells did
    available = pd.DataFrame((CuratedSpikeSorting() & key).fetch(
        "sort_group_id", "curation_id", "sorter_params_name", "units_object_id", as_dict=True))
    if available.empty:
        raise ValueError(f"No v0 curations found for {nwb_file_name}")

    # Sort groups that yielded no units at all have an empty units_object_id, and fetch_nwb
    # returns no "units" key for them. Drop them here
    available = available[available["units_object_id"].astype(str) != ""]
    if available.empty:
        raise ValueError(f"No v0 sort groups with units for {nwb_file_name}")

    if curation_id is None:
        curation_id = available["curation_id"].max()
    available = available[available["curation_id"] == curation_id]

    if sorter_params_name is None:
        params = sorted(available["sorter_params_name"].unique())
        if len(params) > 1:
            raise ValueError(f"{nwb_file_name} has more than one v0 sorter_params_name "
                             f"{params}; pass sorter_params_name= to pick one")
        sorter_params_name = params[0]
    available = available[available["sorter_params_name"] == sorter_params_name]

    # v0 stores one units table per sort group, so read them a group at a time and tag each
    # unit with the group it came from
    frames = []
    for sort_group_id in sorted(available["sort_group_id"]):
        entries = (CuratedSpikeSorting() & {
            **key, "curation_id": curation_id, "sorter_params_name": sorter_params_name,
            "sort_group_id": sort_group_id}).fetch_nwb()
        if not entries or "units" not in entries[0]:
            continue
        units = entries[0]["units"]
        units["sort_group_id"] = sort_group_id
        frames.append(units)

    if not frames:
        raise ValueError(f"No v0 units for {nwb_file_name} at curation_id {curation_id}")

    units = pd.concat(frames)
    units = units[~units["label"].astype(str).str.contains(BAD_UNIT_LABELS, case=False)]

    # The v0 pipeline doesn't give us a peak channel, so we just leave it NaN 
    units = units.rename(columns={"label": "unit_label"})
    units["peak_channel"] = np.nan
    return units[["spike_times", "unit_label", "sort_group_id", "peak_channel"]]


def electrodes_with_units(units: pd.DataFrame) -> list:
    """Get list of electrodes that carry at least one good unit.

    Useful for marking electrodes with spikes so we can figure out where we are
    in the brain. This returns an empty list for sessions sorted with v0 (Frank lab) 
    because v0 doesn't give us a peak channel.

    Parameters:
        units (pd.DataFrame): Output of `fetch_good_units`

    Returns:
        list[int]: Sorted electrode ids, empty for v0 sessions.
    """
    return sorted({int(c) for c in units["peak_channel"].dropna()})


def fetch_electrode_geometry(nwb_file_name: str) -> pd.DataFrame:
    """Probe geometry (rel_x, rel_y) and bad-channel tag for every electrode in a session.

    Parameters:
        nwb_file_name (str): NWB file to fetch geometry for

    Returns:
        pd.DataFrame: Indexed by electrode name (a STRING, matching how electrodes are named
            in the theta tables) and sorted numerically, with columns rel_x, rel_y,
            bad_channel, and is_bad. bad_channel is stored inconsistently across sessions
            ("True" / "1" / 1), so is_bad normalizes it to a boolean.
    """
    geometry = pd.DataFrame(
        (sgc.Electrode * sgc.Probe.Electrode & {"nwb_file_name": nwb_file_name})
        .fetch("name", "rel_x", "rel_y", "bad_channel", as_dict=True)
    ).set_index("name")
    geometry = geometry.loc[sorted(geometry.index, key=int)]
    geometry["is_bad"] = geometry["bad_channel"].astype(str).str.lower().isin(["true", "1"])
    return geometry


def get_raw_electrical_series(nwbf) -> pynwb.ecephys.ElectricalSeries:
    """Get the raw ElectricalSeries out of an open NWB file.

    Berke lab files call it "ElectricalSeries" and Frank lab files call it "e-series", so we
    look it up by type instead of by name.

    Parameters:
        nwbf (pynwb.NWBFile): An open NWB file (what NWBHDF5IO.read() returns)

    Returns:
        pynwb.ecephys.ElectricalSeries: The raw ephys series.
    """
    series = [obj for obj in nwbf.acquisition.values()
              if isinstance(obj, pynwb.ecephys.ElectricalSeries)]
    if not series:
        raise ValueError(f"No ElectricalSeries in acquisition (found {list(nwbf.acquisition)})")
    if len(series) > 1:
        raise ValueError(f"More than one ElectricalSeries in acquisition: "
                         f"{[s.name for s in series]}")
    return series[0]
