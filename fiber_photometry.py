import datajoint as dj
from pynwb import NWBHDF5IO
from spyglass.common import Nwbfile, Session
from spyglass.utils import SpyglassMixin, logger
import ndx_fiber_photometry

schema = dj.schema("fiber_photometry")

def get_photometry_series(nwbfile) -> list[ndx_fiber_photometry.FiberPhotometryResponseSeries]:
    """Return all FiberPhotometryResponseSeries in the acquisition group of an NWB file."""
    return [
        obj for obj in nwbfile.acquisition.values()
        if isinstance(obj, ndx_fiber_photometry.FiberPhotometryResponseSeries)
    ]

def get_excitation_sources(nwbfile) -> list[ndx_fiber_photometry.ExcitationSource]:
    """Return all ExcitationSource devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.ExcitationSource)
    ]

def get_photodetectors(nwbfile) -> list[ndx_fiber_photometry.Photodetector]:
    """Return all Photodetector devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.Photodetector)
    ]

def get_optic_fibers(nwbfile) -> list[ndx_fiber_photometry.OpticalFiber]:
    """Return all OpticalFiber devices in the NWB file."""
    return [
        obj for obj in nwbfile.devices.values()
        if isinstance(obj, ndx_fiber_photometry.OpticalFiber)
    ]

def populate_all_fiber_photometry(nwb_file_name, config: dict = {}):
    nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)
    with NWBHDF5IO(nwb_file_path, mode="r") as io:
        nwbfile = io.read()

        logger.info(f"Populating photometry device tables from {nwb_file_name}")
        ExcitationSource.insert_from_nwbfile(nwbf=nwbfile, config=config.get("ExcitationSource"))
        Photodetector.insert_from_nwbfile(nwbf=nwbfile, config=config.get("Photodetector"))
        OpticalFiber.insert_from_nwbfile(nwbf=nwbfile, config=config.get("OpticalFiber"))

        logger.info(f"Populating fiber photometry series from {nwb_file_name}")
        FiberPhotometrySeries.insert_from_nwbfile(nwbf=nwbfile, nwb_file_name=nwb_file_name)


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
    def insert_from_nwbfile(cls, nwbf, nwb_file_name: str):
        """Insert FiberPhotometryResponseSeries from an NWB file

        Parameters
        ----------
        nwbf : pynwb.NWBFile
            The source NWB file object.
        nwb_file_name : str
            The NWB file name, used to identify the session in the `Session` table.

        Returns
        -------
        photometry_series_list : list
            List of FiberPhotometryResponseSeries names found in the NWB file.
        """

        session_key = {"nwb_file_name": nwb_file_name}

        photometry_series_list = list()
        for series in nwbf.acquisition.values():
            if isinstance(series, ndx_fiber_photometry.FiberPhotometryResponseSeries):
                series_dict = {
                    **session_key,
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
        if photometry_series_list:
            logger.info(f"Inserted photometry series {photometry_series_list}")
        else:
            logger.warning("No conforming photometry series found.")
        return photometry_series_list

    # It would be better practice to use TaskEpoch instead of Session. 
    # But ndx-fiber-photometry doesn't support identifying epochs.
    # As a hack we could put the epoch in series comments but I don't love that.
    # Each series should also reference entries in ExcitationSource, OpticalFiber, Photodetector, etc
    # Berke Lab nwbs should go back and properly implement virus info too so this can be properly linked
    # Lots of TODOs there... I think I will table it for now so we can move forward. 
    # But this will be annoying because it will require some level of re-conversion and reprocessing later on. 
    # Sorry :(

    @classmethod
    def fetch_series(cls, nwb_file_name: str, series_name: str):
        """
        Fetch a FiberPhotometryResponseSeries object by nwbfile and series name.
        
        Parameters
        ----------
        nwb_file_name : str
            The NWB file name
        series_name : str
            The name of the photometry series in the NWB file 
            (same as the name stored in the FiberPhotometrySeries table)

        Returns
        -------
        data : numpy.ndarray
            The data for the FiberPhotometryResponseSeries found in the NWB file
        timestamps : numpy.ndarray
            The timestamps corresponding to each sample in the FiberPhotometryResponseSeries
        """
        # 2 potential options for this funtion, not sure which is preferred:
    
        # nwb_file_path = Nwbfile().get_abs_path(nwb_file_name)

        # try:
        #     with NWBHDF5IO(nwb_file_path, "r", load_namespaces=True) as io:
        #         nwbfile = io.read()
        #         if series_name not in nwbfile.acquisition:
        #             raise ValueError(f"Series '{series_name}' not found in acquisition of '{nwb_file_name}'.")

        #         series = nwbfile.acquisition[series_name]
        #         if not isinstance(series, ndx_fiber_photometry.FiberPhotometryResponseSeries):
        #             raise TypeError(f"Series '{series_name}' is not a FiberPhotometryResponseSeries.")

        #         data = series.data[:]
        #         timestamps = series.get_timestamps()
        #         return data, timestamps

        # except FileNotFoundError:
        #     raise FileNotFoundError(f"NWB file not found at path: {nwb_file_path}")
        
        session_nwb = (Nwbfile() & {"nwb_file_name": nwb_file_name}).fetch_nwb()[0]

        if series_name not in session_nwb.acquisition:
            raise ValueError(f"Series '{series_name}' not found in acquisition of '{nwb_file_name}'.")

        series = session_nwb.acquisition[series_name]
        if not isinstance(series, ndx_fiber_photometry.FiberPhotometryResponseSeries):
            raise TypeError(f"Series '{series_name}' is not a FiberPhotometryResponseSeries.")

        data = series.data[:]
        timestamps = series.get_timestamps()
        return data, timestamps