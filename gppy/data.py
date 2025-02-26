from datetime import datetime, timedelta
from typing import Optional, List, Dict

import itertools
from pathlib import Path
import re
from .utils import read_header


class CalibrationData:
    """
    Represents and manages calibration data files (BIAS, DARK, FLAT).

    Provides specialized handling for calibration data, including:
    - Categorizing calibration files by type
    - Tracking calibration parameters
    - Managing processing state

    Attributes:
        calib_files (Dict[str, List[Path]]): Mapping of calibration types to file lists
        calib_params (Dict[str, dict]): Calibration parameters for each type
        _processed (bool): Internal flag to track processing state

    Methods:
        has_calib_files(): Check if any calibration files exist
        add_fits_file(): Add a FITS file to the calibration dataset
        mark_as_processed(): Mark the calibration dataset as processed

    Example:
        Automatically categorizes BIAS, DARK, and FLAT calibration files
        and tracks their metadata for further processing.
    """

    def __init__(self, folder_path: str or Path):
        if isinstance(folder_path, str):
            folder_path = Path(folder_path).parent

        self.folder_path = folder_path

        self.calib_files: Dict[str, List[Path]] = {
            calib_type: [] for calib_type in ["BIAS", "DARK", "FLAT"]
        }
        self.calib_params: Dict[str, dict] = {
            calib_type: {} for calib_type in ["BIAS", "DARK", "FLAT"]
        }
        self._processed = False

        self.fits_files: List[Path] = []

    @property
    def processed(self):
        """
        Check if the calibration dataset has been processed.

        Returns:
            bool: Processing state of the calibration dataset
        """
        return self._processed

    def has_calib_files(self) -> bool:
        """
        Check if any calibration files exist in the dataset.

        Returns:
            bool: True if calibration files are present, False otherwise
        """
        return any(len(files) > 0 for files in self.calib_files.values())

    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Add a FITS file to the calibration dataset.

        Parses the filename to determine calibration type and extracts metadata.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        filename = fits_path.name
        pattern = (
            r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        )
        match = re.match(pattern, filename)

        if match:
            calib_type = None
            for ctype in ["BIAS", "DARK", "FLAT"]:
                if ctype in match.group(4):
                    calib_type = ctype
                    break

            if calib_type:
                self.fits_files.append(fits_path)
                self.calib_files[calib_type].append(fits_path)

                if not self.calib_params[calib_type]:
                    self._parse_info(match)
                    self.calib_params[calib_type] = {
                        "filter": self.filter,
                        "binning": self.n_binning,
                        "exposure": self.exposure,
                    }
                return True
        return False

    def _parse_info(self, match) -> None:
        """
        Parse common metadata from filename using regex match.

        Extracts:
        - Precise observation timestamp
        - Observation filter
        - Detector binning
        - Exposure time

        Args:
            match (re.Match): Regex match object from filename parsing
        """
        self.unit = self.folder_path.parent.name

        # Parse folder name (e.g., 2001-02-23_gain2750 or 2001-02-23_ToO_gain2750)
        folder_parts = self.folder_path.name.split("_")
        self.date = folder_parts[0]

        if "ToO" in self.folder_path.name:
            self.too = True
        else:
            self.too = False

        # Find the gain part, handling potential additional components
        gain_part = next(
            (part for part in folder_parts if part.startswith("gain")), None
        )
        if gain_part is None:
            print(f"Could not find gain in folder name: {self.folder_path.name}")
        else:
            self.gain = int(re.search(r"gain(\d+)", gain_part).group(1))

        date_str = f"{match.group(2)}_{match.group(3)}"
        self.datetime = datetime.strptime(date_str, "%Y%m%d_%H%M%S")

        filter_str = match.group(5)
        self.filter = f"m{filter_str}" if filter_str.isdigit() else filter_str

        binning = match.group(6)
        self.n_binning = int(binning.split("x")[0])
        self.exposure = float(match.group(7))

    def mark_as_processed(self):
        """
        Mark the calibration dataset as processed.

        Sets the internal processed flag to True, indicating
        that all necessary processing has been completed.
        """
        self._processed = True

    @property
    def obs_params(self):
        return {
            "date": self.date,
            "unit": self.unit,
            "n_binning": self.n_binning,
            "gain": self.gain,
        }

    @property
    def name(self):
        return f"{self.date}_{self.n_binning}x{self.n_binning}_gain{self.gain}_{self.unit}_masterframe"

    def generate_masterframe(self):
        from .preprocess import MasterFrameGenerator

        master = MasterFrameGenerator(self.obs_params)
        master.run()


class ObservationData:

    _id_counter = itertools.count(1)

    """
    Represents and manages astronomical observation data files.

    Provides specialized handling for observation data, including:
    - Tracking observation targets
    - Managing processed target states
    - Supporting multiple targets and filters

    Attributes:
        target (Optional[str]): Current observation target
        _processed (set): Set of processed target information

    Methods:
        add_fits_file(): Add a FITS file to the observation dataset
        get_unprocessed(): Retrieve unprocessed targets
        mark_as_processed(): Mark specific targets as processed

    Example:
        Tracks observation targets, filters, and processing state
        for complex astronomical observation datasets.
    """

    def __init__(self, file_path: str or Path):
        self.target: Optional[str] = None

        self._id = next(self._id_counter)
        if isinstance(file_path, str):
            file_path = Path(file_path)

        self.file_path = file_path

        if "calib" in file_path.name:
            self._file_type = "processed"

        else:
            self._file_type = "raw"

        self._parse_info_from_header()
        self.too = False

    @property
    def id(self):
        return self._id

    def _parse_info_from_header(self) -> None:
        """
        Extract target information from a FITS filename.

        Args:
            file_path (Path): Path to the FITS file

        Returns:
            tuple: Target name and filter, or None if parsing fails
        """
        header = read_header(self.file_path)
        for attr, key in zip(["exposure", "gain", "filter", "date", "obj", "unit", "n_binning"], \
                             ["EXPOSURE", "GAIN", "FILTER", "DATE-LOC", "OBJECT", "TELESCOP", "XBINNING"]):  # fmt:skip
            if key == "DATE-LOC":
                header_date = datetime.fromisoformat(header[key])
                adjusted_date = header_date - timedelta(hours=12)
                final_date = adjusted_date.date()
                setattr(self, attr, final_date.isoformat())
            else:
                setattr(self, attr, header[key])

    @property
    def identifier(self):
        return (self.obj, self.filter)

    @property
    def obs_params(self):
        return {
            "date": self.date,
            "unit": self.unit,
            "gain": self.gain,
            "obj": self.obj,
            "filter": self.filter,
            "n_binning": self.n_binning,
        }

    @property
    def name(self):
        return f"{self.date}_{self.n_binning}x{self.n_binning}_gain{self.gain}_{self.obj}_{self.unit}_{self.filter}"

    def run_calibaration(self, save_path=None, verbose=True):
        from .config import Configuration
        from .preprocess import Calibration

        self.config = Configuration(self.obs_params, overwrite=True, verbose=verbose)
        if save_path:
            self.config.config.path.path_processed = save_path
        calib = Calibration(self.config)
        calib.run()

    def run_astrometry(self):
        from .astrometry import Astrometry

        if hasattr(self, "config"):
            astro = Astrometry(self.config)
        else:
            astro = Astrometry.from_file(self.file_path)
        astro.run()

    def run_photometry(self):
        from .photometry import Photometry

        if hasattr(self, "config"):
            phot = Photometry(self.config)
        else:
            phot = Photometry.from_file(self.file_path)
        phot.run()


class ObservationDataSet(ObservationData):

    def __init__(self, folder_path: str or Path):
        self._processed = set()
        self.obs_list = []

    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Add a FITS file to the observation dataset.

        Parses the filename to extract target and observation metadata.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        if all(ctype not in fits_path.name for ctype in ["BIAS", "DARK", "FLAT"]):
            obs = ObservationData(fits_path)
            self.obs_list.append(obs)
            return True
        return False

    @property
    def processed(self):
        """
        Get the set of processed targets.

        Returns:
            set: Set of processed target information
        """
        return self._processed

    def get_unprocessed(self) -> set:
        """
        Retrieve targets that have not yet been processed.

        Returns:
            set: Set of unprocessed target information
        """
        for obs in self.obs_list:
            if obs.identifier not in self._processed:
                yield obs

    def mark_as_processed(self, obs_identifier):
        """
        Mark a specific target as processed.

        Args:
            obs_identifier (Tuple[str, str]): Identifier of the observation to mark as processed
        """
        self._processed.add(obs_identifier)
