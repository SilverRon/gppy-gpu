from watchdog.events import FileSystemEventHandler
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable, Dict
from abc import ABC, abstractmethod
from ..logger import Logger

class RawDataFile(ABC):
    """
    Abstract base class for representing raw astronomical data files.

    Provides a common interface and basic implementation for parsing
    and managing raw data files from astronomical observations.

    Key Responsibilities:
    - Parse folder and filename metadata
    - Extract observation parameters
    - Manage FITS file collections
    - Support different data types (Calibration, Observation)

    Attributes:
        folder_path (Path): Path to the data folder
        unit (str): Observation unit name
        date (str): Observation date
        too (bool): Whether this is a Target of Opportunity (ToO) observation
        gain (int): Detector gain setting
        filter (Optional[str]): Observation filter
        n_binning (Optional[int]): Detector binning
        exposure (Optional[float]): Exposure time
        datetime (Optional[datetime]): Precise observation timestamp
        fits_files (List[Path]): List of FITS files in the dataset

    Example:
        Subclasses like CalibrationData and ObservationData inherit from this class
        to provide specialized parsing and processing for different data types.
    """

    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        
        self.unit = folder_path.parent.name
        
        # Parse folder name (e.g., 2001-02-23_gain2750 or 2001-02-23_ToO_gain2750)
        folder_parts = folder_path.name.split('_')
        self.date = folder_parts[0]
        
        if "ToO" in folder_path.name:
            self.too = True
        else:
            self.too = False

        # Find the gain part, handling potential additional components
        gain_part = next((part for part in folder_parts if part.startswith('gain')), None)
        if gain_part is None:
            raise ValueError(f"Could not find gain in folder name: {folder_path.name}")
        
        self.gain = int(re.search(r'gain(\d+)', gain_part).group(1))
        
        # Common attributes
        self.filter: Optional[str] = None
        self.n_binning: Optional[int] = None
        self.exposure: Optional[float] = None
        self.datetime: Optional[datetime] = None
        self.fits_files: List[Path] = []

    @abstractmethod
    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Abstract method to add a FITS file to the dataset.

        Must be implemented by subclasses to handle specific file parsing
        and metadata extraction.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        pass

    def _parse_common_info(self, match) -> None:
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
        date_str = f"{match.group(2)}_{match.group(3)}"
        self.datetime = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        
        filter_str = match.group(5)
        self.filter = f"m{filter_str}" if filter_str.isdigit() else filter_str
        
        binning = match.group(6)
        self.n_binning = int(binning.split('x')[0])
        self.exposure = float(match.group(7))


class CalibrationData(RawDataFile):
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

    def __init__(self, folder_path: Path):
        super().__init__(folder_path)
        self.calib_files: Dict[str, List[Path]] = {
            calib_type: [] for calib_type in ['BIAS', 'DARK', 'FLAT']
        }
        self.calib_params: Dict[str, dict] = {
            calib_type: {} for calib_type in ['BIAS', 'DARK', 'FLAT']
        }
        self._processed = False

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
        pattern = r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        match = re.match(pattern, filename)
        
        if match:
            calib_type = None
            for ctype in ['BIAS', 'DARK', 'FLAT']:
                if ctype in match.group(4):
                    calib_type = ctype
                    break
                    
            if calib_type:
                self.fits_files.append(fits_path)
                self.calib_files[calib_type].append(fits_path)
                
                if not self.calib_params[calib_type]:
                    self._parse_common_info(match)
                    self.calib_params[calib_type] = {
                        'filter': self.filter,
                        'binning': self.n_binning,
                        'exposure': self.exposure
                    }
                return True
        return False

    def mark_as_processed(self):
        """
        Mark the calibration dataset as processed.

        Sets the internal processed flag to True, indicating
        that all necessary processing has been completed.
        """
        self._processed = True


class ObservationData(RawDataFile):
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

    def __init__(self, folder_path: Path):
        super().__init__(folder_path)
        self.target: Optional[str] = None
        self._processed = set()  # Track processed targets
    
    def add_fits_file(self, fits_path: Path) -> bool:
        """
        Add a FITS file to the observation dataset.

        Parses the filename to extract target and observation metadata.

        Args:
            fits_path (Path): Path to the FITS file

        Returns:
            bool: Whether the file was successfully added to the dataset
        """
        filename = fits_path.name
        pattern = r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        match = re.match(pattern, filename)
        
        if match and not any(ctype in match.group(4) for ctype in ['BIAS', 'DARK', 'FLAT']):
            self.fits_files.append(fits_path)
            self._parse_common_info(match)
            self.target = match.group(4)
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
        current_targets = {self._get_target_from_file(f) for f in self.fits_files}
        return current_targets - self.processed

    def _get_target_from_file(self, file_path: Path) -> tuple:
        """
        Extract target information from a FITS filename.

        Args:
            file_path (Path): Path to the FITS file

        Returns:
            tuple: Target name and filter, or None if parsing fails
        """
        pattern = r"(\w+)_(\d{8})_(\d{6})_([^_]+)_([^_]+)_(\d+x\d+)_(\d+\.\d+)s_\d+\.fits"
        match = re.match(pattern, file_path.name)
        if match:
            target = match.group(4)
            filter_str = match.group(5)
            filter_val = f"m{filter_str}" if filter_str.isdigit() else filter_str
            return (target, filter_val)
        return None

    def mark_as_processed(self, target_info: tuple):
        """
        Mark a specific target as processed.

        Args:
            target_info (tuple): Target information to mark as processed
        """
        self._processed.add(target_info)


class Monitor(FileSystemEventHandler):
    """
    Advanced file system monitor for astronomical data processing.

    Provides real-time monitoring of file system events, specifically
    designed for tracking and processing astronomical observation data.

    Key Features:
    - Watch for new FITS files and observation folders
    - Automatically categorize data (Calibration vs Observation)
    - Support multiple processing callbacks
    - Flexible and extensible design

    Attributes:
        base_path (Path): Root directory for monitoring
        folder_data (Dict[str, Dict]): Tracked folder data
        callbacks (List[tuple]): Registered processing callbacks

    Methods:
        add_process(): Register a new processing callback
        on_created(): Handle file system creation events

    Example:
        >>> from watchdog.observers import Observer
        >>> monitor = Monitor(base_path='/path/to/observations')
        >>> monitor.add_process(process_calibration_data)
        >>> observer = Observer()
        >>> observer.schedule(monitor, base_path, recursive=True)
        >>> observer.start()
    """

    def __init__(self, base_path: Path):
        super().__init__()
        self.base_path = Path(base_path)
        self.folder_data: Dict[str, RawDataFile] = {}
        self.callbacks: List[tuple[Callable, dict]] = []
    
    def add_process(self, callback: Callable[[RawDataFile], None], **kwargs):
        """
        Register a processing callback for new data.

        Args:
            callback (Callable): Function to call when new data is detected
            **kwargs: Additional keyword arguments for the callback
        """

        self.callbacks.append((callback, kwargs))

    def _execute_callbacks(self, data: RawDataFile):
        """
        Execute all registered callbacks for a given data object.

        Args:
            data (RawDataFile): Data object to process
        """
        for callback, kwargs in self.callbacks:
            callback(data, **kwargs)
            
    def _is_observation_folder(self, path: Path) -> bool:
        """
        Validate if a path represents a valid observation folder.

        Args:
            path (Path): Path to check

        Returns:
            bool: Whether the path is a valid observation folder
        """
        try:
            if not re.match(r'\d{4}-\d{2}-\d{2}_gain\d+', path.name):
                return False
            if path.parent.parent != self.base_path:
                return False
            return True
        except Exception:
            return False

    def _process_new_file(self, file_path: Path):
        """
        Process a newly detected FITS file.

        Attempts to add the file to calibration or observation datasets
        and triggers appropriate callbacks.

        Args:
            file_path (Path): Path to the new FITS file
        """
        folder_path = str(file_path.parent)
        
        if folder_path not in self.folder_data:
            # Initialize both calibration and observation data handlers
            self.folder_data[folder_path] = {
                'calib': CalibrationData(file_path.parent),
                'obs': ObservationData(file_path.parent)
            }
        
        data_handlers = self.folder_data[folder_path]
        
        # Try to add file to calibration data first
        if data_handlers['calib'].add_fits_file(file_path):
            if not data_handlers['calib'].processed and data_handlers['calib'].has_calib_files():
                self._execute_callbacks(data_handlers['calib'])
        # If not a calibration file, try observation data
        elif data_handlers['obs'].add_fits_file(file_path):
            self._execute_callbacks(data_handlers['obs'])
    
    def on_created(self, event):
        """
        Handle file system creation events.

        Processes new FITS files and detects new observation folders.

        Args:
            event (FileSystemEvent): Watchdog file system event
        """
       
        path = Path(event.src_path)

        if not event.is_directory and path.suffix.lower() == '.fits':
            self._process_new_file(path)
        elif event.is_directory and self._is_observation_folder(path):
            logger = Logger()
            logger.info(f"New observation folder detected: {path}")

