import datajoint as dj
from pynwb import NWBHDF5IO
from spyglass.common import Nwbfile, Session
from spyglass.utils import SpyglassMixin, logger
import ndx_fiber_photometry
from ndx_fiber_photometry import FiberPhotometryResponseSeries, ExcitationSource, OpticalFiber, Photodetector

schema = dj.schema("fiber_photometry")

def get_photometry_series(nwbfile) -> list[FiberPhotometryResponseSeries]:
    """Return all FiberPhotometryResponseSeries in the acquisition group of an NWB file."""
    return [
        obj for obj in nwbfile.acquisition.values()
        if isinstance(obj, ndx_fiber_photometry.FiberPhotometryResponseSeries)
    ]

def get_excitation_sources(nwbfile) -> list[ExcitationSource]:
    """Return all ExcitationSource devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.ExcitationSource)
    ]

def get_photodetectors(nwbfile) -> list[Photodetector]:
    """Return all Photodetector devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.Photodetector)
    ]

def get_optic_fibers(nwbfile) -> list[OpticalFiber]:
    """Return all OpticalFiber devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.OpticalFiber)
    ]

def populate_all_photometry(nwb_file_name, config: dict = {}):
    nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)
    with NWBHDF5IO(nwb_file_path, mode="r") as io:
        nwbfile = io.read()

        logger.info(f"Populating photometry device tables from {nwb_file_name}")
        excitation_sources = ExcitationSource.insert_from_nwbfile(nwbfile, config=config.get("ExcitationSource"))
        photodetectors = Photodetector.insert_from_nwbfile(nwbfile, config=config.get("Photodetector"))
        optical_fibers = OpticalFiber.insert_from_nwbfile(nwbfile, config=config.get("OpticalFiber"))

        # print("excitation")
        # print(excitation_sources)
        # print("photodetector")
        # print(photodetectors)
        # print("optic fiber")
        # print(optic_fibers)
    
        # Get actual photometry series
        photometry_series = get_photometry_series(nwbfile)
        print(f"photometry series")
        print(photometry_series)


@schema
class ExcitationSource(SpyglassMixin, dj.Manual):
    definition = """
    excitation_source_name: varchar(80)
    ---
    manufacturer = "": varchar(1000)
    model = "": varchar(1000)
    illumination_type = "": varchar(1000)
    excitation_wavelength_in_nm = 0: float
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert excitation sources from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of excitation source object names found in the NWB file.
        """
        config = config or dict()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_fiber_photometry.ExcitationSource):
                device_dict = {
                    "excitation_source_name": device.name,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "illumination_type": device.illumination_type,
                    "excitation_wavelength_in_nm": device.excitation_wavelength_in_nm,
                }
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict["excitation_source_name"])
        # Append devices from config file
        if device_list := config.get("ExcitationSource"):
            device_inserts = [
                {
                    "excitation_source_name": device.get("name"),
                    "manufacturer": device.get("manufacturer"),
                    "model": device.get("model"),
                    "illumination_type": device.get("illumination_type"),
                    "excitation_wavelength_in_nm": device.get("excitation_wavelength_in_nm", 0),
                }
                for device in device_list
            ]
            cls.insert(device_inserts, skip_duplicates=True)
            device_name_list.extend([d["excitation_source_name"] for d in device_inserts])
        if device_name_list:
            logger.info(f"Inserted excitation sources {device_name_list}")
        else:
            logger.warning("No conforming excitation source metadata found.")
        return device_name_list


@schema
class Photodetector(SpyglassMixin, dj.Manual):
    definition = """
    photodetector_name: varchar(80)
    ---
    manufacturer = "": varchar(1000)
    model = "": varchar(1000)
    description = "": varchar(2000)
    detector_type = "": varchar(1000)
    detected_wavelength_in_nm = 0: float
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert photodetectors from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of photodetector object names found in the NWB file.
        """
        config = config or dict()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_fiber_photometry.Photodetector):
                device_dict = {
                    "photodetector_name": device.name,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "description": device.description,
                    "detector_type": device.detector_type,
                    "detected_wavelength_in_nm": device.detected_wavelength_in_nm,
                }
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict["photodetector_name"])
        # Append devices from config file
        if device_list := config.get("Photodetector"):
            device_inserts = [
                {
                    "photodetector_name": device.get("name"),
                    "manufacturer": device.get("manufacturer"),
                    "model": device.get("model"),
                    "description": device.get("description"),
                    "detector_type": device.get("detector_type"),
                    "detected_wavelength_in_nm": device.get("detected_wavelength_in_nm", 0),
                }
                for device in device_list
            ]
            cls.insert(device_inserts, skip_duplicates=True)
            device_name_list.extend([d["photodetector_name"] for d in device_inserts])
        if device_name_list:
            logger.info(f"Inserted photodetectors {device_name_list}")
        else:
            logger.warning("No conforming photodetector metadata found.")
        return device_name_list


@schema
class OpticalFiber(SpyglassMixin, dj.Manual):
    definition = """
    optical_fiber_name: varchar(80)
    ---
    manufacturer = "": varchar(1000)
    model = "": varchar(1000)
    numerical_aperture = 0: float
    core_diameter_in_um = 0: float
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf, config=None):
        """Insert optical fibers from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        config : dict
            Dictionary read from a user-defined YAML file containing values to
            replace in the NWB file.

        Returns
        -------
        device_name_list : list
            List of optic fiber object names found in the NWB file.
        """
        config = config or dict()
        device_name_list = list()
        for device in nwbf.devices.values():
            if isinstance(device, ndx_fiber_photometry.OpticalFiber):
                device_dict = {
                    "optical_fiber_name": device.name,
                    "manufacturer": device.manufacturer,
                    "model": device.model,
                    "numerical_aperture": device.numerical_aperture,
                    "core_diameter_in_um": device.core_diameter_in_um,
                }
                cls.insert1(device_dict, skip_duplicates=True)
                device_name_list.append(device_dict["optical_fiber_name"])
        # Append devices from config file
        if device_list := config.get("OpticalFiber"):
            device_inserts = [
                {
                    "optical_fiber_name": device.get("name"),
                    "manufacturer": device.get("manufacturer"),
                    "model": device.get("model"),
                    "numerical_aperture": device.get("numerical_aperture", 0),
                    "core_diameter_in_um": device.get("core_diameter_in_um", 0),
                }
                for device in device_list
            ]
            cls.insert(device_inserts, skip_duplicates=True)
            device_name_list.extend([d["optical_fiber_name"] for d in device_inserts])
        if device_name_list:
            logger.info(f"Inserted optical fibers {device_name_list}")
        else:
            logger.warning("No conforming optical fiber metadata found.")
        return device_name_list


@schema
class FiberPhotometrySeries(SpyglassMixin, dj.Manual):
    """
    Stores metadata for each FiberPhotometryResponseSeries in the NWB file.

    You can fetch the actual NWB series object by its name and file.
    """

    definition = """
    -> Session
    photometry_series_name: varchar(64)  # the name of the FiberPhotometryResponseSeries
    ---
    description: varchar(255)
    unit: varchar(16)
    rate: float
    starting_time = 0: float
    starting_time_unit = "seconds": varchar(100)
    offset = 0: float
    """

    @classmethod
    def insert_from_nwbfile(cls, nwbf):
        """Insert FiberPhotometryResponseSeries from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.

        Returns
        -------
        photometry_series_list : list
            List of FiberPhotometryResponseSeries names found in the NWB file.
        """
        photometry_series_list = list()
        for series in nwbf.acquisition.values():
            if isinstance(series, ndx_fiber_photometry.FiberPhotometryResponseSeries):
                series_dict = {
                    "photometry_series_name": series.name,
                    "description": series.description,
                    "unit": series.unit,
                    "rate": series.rate,
                    "starting_time": series.starting_time,
                    "starting_time_unit": series.starting_time_unit,
                    "offset": series.offset,
                }
                cls.insert1(series_dict, skip_duplicates=True)
                photometry_series_list.append(series_dict["photometry_series_name"])
                
    # TODO: need nwb_file_name because using Session as primary key.
    # It would be better practice to use TaskEpoch instead. but ndx-fiber-photometry doesn't support epochs?
    # As a hack we could put this in series comments but I don't like that.
    # Each series should also reference entries in excitationsource, opticalfiber, etc
    # Berke Lab nwbs should go back and properly implement virus info too so this can be properly linked

    @classmethod
    def fetch_series(cls, nwb_file_name: str, photometry_series_name: str):
        """
        Fetch a FiberPhotometryResponseSeries object by nwbfile and series name.
        """
        file_path = Nwbfile().get_abs_path(nwb_file_name)
        with NWBHDF5IO(file_path, "r") as io:
            nwbfile = io.read()
            try:
                series = nwbfile.acquisition[photometry_series_name]
                if isinstance(series, ndx_fiber_photometry.FiberPhotometryResponseSeries):
                    return series.data
            except KeyError:
                raise ValueError(f"FiberPhotometryResponseSeries '{photometry_series_name}' not found in {nwb_file_name}")
