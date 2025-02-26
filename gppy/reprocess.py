import re
import time
from pathlib import Path

from .services.queue import QueueManager, Priority
from .services.monitor import CalibrationData, ObservationData
from .run import run_scidata_reduction, run_masterframe_generator

def reprocess_folder(folder, overwrite=False):
    """
    Reprocess all FITS files in a given folder and its subfolders.

    This function performs a comprehensive scan of the input folder, identifying
    and processing astronomical data folders. It handles both calibration and
    observation data, using a queue-based parallel processing approach.

    Processing steps:
    1. Validate input folder
    2. Identify valid data subfolders
    3. Initialize calibration and observation data handlers
    4. Scan and add FITS files to data handlers
    5. Process calibration and observation data
    6. Wait for queue to complete processing
    7. Report any processing errors

    Args:
        folder (str): Path to the folder containing data to be reprocessed
        overwrite (bool, optional): Flag to enable overwriting of existing 
            processed files. Defaults to False.

    Raises:
        RuntimeError: If any errors occur during folder processing
        ValueError: If the input folder is not a valid directory

    Example:
        >>> reprocess_folder('/path/to/data/folder')
        Finished processing files in /path/to/data/folder

    Note:
        - Supports folders with naming patterns like:
          YYYY-MM-DD(_ToO)?(_NxN)?_gainX
        - Uses QueueManager for parallel processing
        - Skips folders without FITS files
    """
    queue = QueueManager(max_workers=20)
    
    # Convert folder to Path object
    folder_path = Path(folder)
    
    # Check if folder exists
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Error: {folder} is not a valid directory")
        return
    
    # Track errors during processing
    processing_errors = []
    
    # Find all potential data folders 
    data_folders = []
    
    # Check if the input folder itself matches the data folder pattern
    if re.match(r'\d{4}-\d{2}-\d{2}(_ToO)?(_\dx\d)?_gain\d+', folder_path.name):
        data_folders.append(folder_path)
    
    # If not, search subfolders
    data_folders.extend([
        f for f in folder_path.iterdir() 
        if f.is_dir() and re.match(r'\d{4}-\d{2}-\d{2}(_ToO)?(_\dx\d)?_gain\d+', f.name)
    ])
    
    # If no folders found, print a warning
    if not data_folders:
        print(f"No valid data folders found in {folder_path}")
        return
    
    # Process each data folder
    for data_folder in data_folders:
        try:
            # Initialize data handlers for the folder
            calib_data = CalibrationData(data_folder)
            obs_data = ObservationData(data_folder)
            
            # Find and process all FITS files in the folder
            fits_files = list(data_folder.glob('**/*.fits'))
            
            if not fits_files:
                print(f"No FITS files found in {data_folder}")
                continue
            
            for fits_file in fits_files:
                calib_data.add_fits_file(fits_file)
                obs_data.add_fits_file(fits_file)
            
            # Process calibration data if exists
            if calib_data.has_calib_files() and not calib_data.processed:

                obs_params = {
                    "date": calib_data.date,
                    "unit": calib_data.unit,
                    "n_binning": calib_data.n_binning,
                    "gain": calib_data.gain,
                }

                queue.add_task(
                    run_masterframe_generator,
                    args=(obs_params,),
                    kwargs={"queue": False},
                    priority=Priority.LOW,
                    gpu=True,
                    task_name=f"{obs_params['date']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['unit']}_masterframe",
                )
            
            # Process observation data
            for obs in obs_data.get_unprocessed():
                obs_data.mark_as_processed(obs)
                obj, filte = obs

                obs_params = {
                    "date": obs_data.date,
                    "unit": obs_data.unit,
                    "gain": obs_data.gain,
                    "obj": obj,
                    "filter": filte,
                    "n_binning": obs_data.n_binning,
                }

                queue.add_task(
                    run_scidata_reduction,
                    args=(obs_params,),
                    kwargs={"queue": False},
                    priority=Priority.LOW,
                    task_name=f"{obs_params['date']}_{obs_params['n_binning']}x{obs_params['n_binning']}_gain{obs_params['gain']}_{obs_params['obj']}_{obs_params['unit']}_{obs_params['filter']}",
                )

                time.sleep(0.1)
        except Exception as e:
            error_msg = f"Error processing folder {data_folder}: {str(e)}"
            processing_errors.append(error_msg)
            print(error_msg)
    
    # Wait for queue to complete processing
    queue.wait_until_task_complete("all")

    # Raise an exception if any errors occurred during processing
    if processing_errors:
        raise RuntimeError("\n".join(processing_errors))

    # Print summary of processing
    print(f"Finished processing files in {folder}")


