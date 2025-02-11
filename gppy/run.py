__package__ = "gppy"

from .config import Configuration
from .preprocess.calibration import Calibration
from .astrometry import Astrometry
from .photometry.photometry import Photometry

from .preprocess.masterframe import MasterFrameGenerator
from .const import RAWDATA_DIR
import time

from watchdog.observers import Observer

from .services.monitor import Monitor, CalibrationData, ObservationData
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


def run_scidata_reduction(obs_params, queue=False, **kwargs):
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
    config = Configuration(obs_params, logger=logger, **kwargs)
    calib = Calibration(config, logger=config.logger)
    calib.run()
    astrm = Astrometry(config, logger=config.logger)
    astrm.run()
    phot = Photometry(config, logger=config.logger)
    phot.run()
    del config, calib, astrm, phot


# def run_combined_reduction(obs_params, queue=False, **kwargs):
#     imcom = Imcombine(config)
#     imcom.run
#     photo = Photometry(config)
#     photo.run()
#     del imcom, photo


def run_pipeline(data, queue: QueueManager):
    """
    Central pipeline processing function for different types of astronomical data.

    Handles two primary data types:
    1. CalibrationData: Generates master calibration frames
    2. ObservationData: Processes scientific observations

    Processing includes:
    - Marking data as processed to prevent reprocessing
    - Adding tasks to the queue with appropriate priorities
    - Supporting Time-On-Target (TOO) observations with high priority

    Args:
        data (CalibrationData or ObservationData): Input data to process
        queue (QueueManager): Task queue for managing parallel processing
    """
    if isinstance(data, CalibrationData):
        if not data.processed:
            data.mark_as_processed()

            obs_params = {
                "date": data.date,
                "unit": data.unit,
                "n_binning": data.n_binning,
                "gain": data.gain,
            }

            queue.add_task(
                run_masterframe_generator,
                args=(obs_params,),
                kwargs={"queue": False},
                gpu=True,
                task_name=f"{obs_params['date']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['unit']}_masterframe",
            )
            time.sleep(0.1)
    elif isinstance(data, ObservationData):
        for obs in data.get_unprocessed():
            data.mark_as_processed(obs)
            obj, filte = obs

            obs_params = {
                "date": data.date,
                "unit": data.unit,
                "gain": data.gain,
                "obj": obj,
                "filter": filte,
                "n_binning": data.n_binning,
            }
            priority = Priority.HIGH if data.too else Priority.MEDIUM

            queue.add_task(
                run_scidata_reduction,
                args=(obs_params,),
                kwargs={"queue": False},
                priority=priority,
                task_name=f"{obs_params['date']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['obj']}_{obs_params['unit']}_{obs_params['filter']}",
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


if __name__ == "__main__":
    start_monitoring()
