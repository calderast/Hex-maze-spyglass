import datajoint as dj
from pynwb import NWBHDF5IO
from spyglass.common import Nwbfile, Session
from ndx_fiber_photometry import FiberPhotometryResponseSeries, ExcitationSource, OpticalFiber, Photodetector

schema = dj.schema("fiber_photometry")

def get_photometry_series(nwbfile) -> list[FiberPhotometryResponseSeries]:
    """Return all FiberPhotometryResponseSeries in the acquisition group of an NWB file."""
    return [
        obj for obj in nwbfile.acquisition.values()
        if isinstance(obj, FiberPhotometryResponseSeries)
    ]

def get_excitation_sources(nwbfile) -> list[ExcitationSource]:
    """Return all ExcitationSource devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ExcitationSource)
    ]

def get_photodetectors(nwbfile) -> list[Photodetector]:
    """Return all Photodetector devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, Photodetector)
    ]

def get_optic_fibers(nwbfile) -> list[OpticalFiber]:
    """Return all OpticalFiber devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, OpticalFiber)
    ]

def populate_all_photometry(nwb_file_name):
    nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)
    with NWBHDF5IO(nwb_file_path, mode="r") as io:
        nwbfile = io.read()
        # Get photometry data from the nwbfile
        photometry_series = get_photometry_series(nwbfile)
        excitation_sources = get_excitation_sources(nwbfile)
        photodetectors = get_photodetectors(nwbfile)
        optic_fibers = get_optic_fibers(nwbfile)
        
        # TODO

                        
@schema
class FiberPhotometrySeries(SpyglassMixin, dj.Manual):
    """
    Stores metadata for each FiberPhotometryResponseSeries in the NWB file.

    You can fetch the actual NWB series object by its name and file.
    """

    definition = """
    -> Session
    series_name: varchar(64)  # the name of the FiberPhotometryResponseSeries
    ---
    description: varchar(255)
    unit: varchar(16)
    sampling_rate: float
    """

    @classmethod
    def insert_from_nwb(cls, nwb_file_name):
        """Insert all FiberPhotometryResponseSeries in the NWB acquisition group."""
        file_path = Nwbfile().get_abs_path(nwb_file_name)
        with NWBHDF5IO(file_path, "r") as io:
            nwbfile = io.read()
            photometry_series = get_photometry_series(nwbfile)
            for series in photometry_series:
                cls.insert1(
                    dict(
                        nwb_file_name=nwb_file_name,
                        series_name=series.name,
                        description=series.description,
                        unit=series.unit,
                        sampling_rate=series.rate,
                    ),
                    skip_duplicates=True,
                )

    @classmethod
    def fetch_series(cls, nwb_file_name: str, series_name: str):
        """
        Fetch a FiberPhotometryResponseSeries object by file and series name.
        """
        file_path = Nwbfile().get_abs_path(nwb_file_name)
        with NWBHDF5IO(file_path, "r") as io:
            nwbfile = io.read()
            try:
                return nwbfile.acquisition[series_name]
            except KeyError:
                raise ValueError(f"Series '{series_name}' not found in {nwb_file_name}")
