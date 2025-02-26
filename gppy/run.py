__package__ = "gppy"

from .config import Configuration
from .astrometry import Astrometry
from .photometry.photometry import Photometry

from .preprocess import Calibration, MasterFrameGenerator
from .const import RAWDATA_DIR
import time

from watchdog.observers import Observer

from .services.monitor import Monitor
from .data import ObservationDataSet, CalibrationData
from .services.queue import QueueManager, Priority

from .logger import Logger


def run_masterframe_generator(obs_params, queue=False):
    """
    Generate master calibration frames for a specific observation set.

    Master frames are combined calibration images (like dark, flat, bias) that
    help in reducing systematic errors in scientific observations.

    Args:
        obs_params (dict): Observation parameters including:
            - date: Observation date
            - unit: Observation unit/instrument
            - n_binning: Pixel binning factor
            - gain: Detector gain setting
        queue (bool, optional): Whether to use queue-based processing. Defaults to False.
    """
    master = MasterFrameGenerator(obs_params, queue=queue)
    master.run()
    del master


def run_scidata_reduction(obs_params, queue=False, processes = ["calibration", "astrometry", "photometry"], **kwargs):
    """
    Perform comprehensive scientific data reduction pipeline.

    This function orchestrates the complete data reduction process for scientific observations,
    including:
    1. Configuration initialization
    2. Calibration
    3. Astrometric processing
    4. Photometric analysis

    Args:
        obs_params (dict): Observation parameters including:
            - date: Observation date
            - unit: Observation unit/instrument
            - gain: Detector gain setting
            - obj: Target object name
            - filter: Observation filter
            - n_binning: Pixel binning factor
        queue (bool, optional): Whether to use queue-based processing. Defaults to False.
        **kwargs: Additional configuration parameters
    """
    logger = Logger(name="7DT pipeline logger", slack_channel="pipeline_report")
    try:
        config = Configuration(obs_params, logger=logger, overwrite=False, **kwargs)
        if not (config.config.flag.calibration) and "calibration" in processes:
            calib = Calibration(config, queue=queue)
            calib.run()
            del calib
        if not (config.config.flag.astrometry) and "astrometry" in processes:
            astrm = Astrometry(config, queue=queue)
            astrm.run()
            del astrm
        if not (config.config.flag.single_photometry) and "photometry" in processes:
            phot = Photometry(config, queue=queue)
            phot.run()
            del phot
    except Exception as e:
        logger.error(f"Error during abrupt stop: {e}")

def run_pipeline(data, queue: QueueManager):
    """
    Central pipeline processing function for different types of astronomical data.

    Handles two primary data types:
    1. CalibrationData: Generates master calibration frames
    2. ObservationDataSet: Processes scientific observations

    Processing includes:
    - Marking data as processed to prevent reprocessing
    - Adding tasks to the queue with appropriate priorities
    - Supporting Time-On-Target (TOO) observations with high priority

    Args:
        data (CalibrationData or ObservationDataSet): Input data to process
        queue (QueueManager): Task queue for managing parallel processing
    """
    if isinstance(data, CalibrationData):
        if not data.processed:
            data.mark_as_processed()

            queue.add_task(
                run_masterframe_generator,
                args=(data.obs_params,),
                kwargs={"queue": False},
                gpu=True,
                task_name=data.name,
            )
            time.sleep(0.1)
    elif isinstance(data, ObservationDataSet):
        for obs in data.get_unprocessed():
            data.mark_as_processed(obs.identifier)

            priority = Priority.HIGH if obs.too else Priority.MEDIUM

            queue.add_task(
                run_scidata_reduction,
                args=(obs.obs_params,),
                kwargs={"queue": False},
                priority=priority,
                task_name=obs.name,
            )

            time.sleep(0.1)

def start_monitoring():
    """
    Initialize and start the astronomical data monitoring system.

    This function:
    1. Creates a QueueManager for parallel task processing
    2. Sets up a Monitor for the raw data directory
    3. Configures a watchdog Observer to track file system changes
    4. Provides a continuous monitoring loop with graceful shutdown

    The monitoring system watches for new calibration and observation data,
    automatically triggering the appropriate processing pipelines.

    Monitoring can be stopped by pressing Ctrl+C.
    """
    queue = QueueManager(max_workers=20)

    monitor = Monitor(RAWDATA_DIR)
    monitor.add_process(run_pipeline, queue=queue)

    observer = Observer()
    observer.schedule(monitor, RAWDATA_DIR, recursive=True)
    observer.start()

    try:
        print(f"Starting monitoring of {RAWDATA_DIR}")
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        if queue:
            queue.abrupt_stop()
        print("\nMonitoring stopped")

    observer.join()
