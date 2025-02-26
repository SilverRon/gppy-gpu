from watchdog.events import FileSystemEventHandler
import re
from pathlib import Path
from typing import List, Callable, Dict
from ..logger import Logger
from ..data import CalibrationData, ObservationDataSet

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
        self.folder_data: Dict[str] = {}
        self.callbacks: List[tuple[Callable, dict]] = []
    
    def add_process(self, callback: Callable[[CalibrationData, ObservationDataSet], None], **kwargs):
        """
        Register a processing callback for new data.

        Args:
            callback (Callable): Function to call when new data is detected
            **kwargs: Additional keyword arguments for the callback
        """

        self.callbacks.append((callback, kwargs))

    def _execute_callbacks(self, data):
        """
        Execute all registered callbacks for a given data object.

        Args:
            data: Data object to process
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
                'obs': ObservationDataSet(file_path.parent)
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

