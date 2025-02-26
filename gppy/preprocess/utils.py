import os
import time
import gc
import cupy as cp
from astropy.io import fits
from contextlib import contextmanager
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def write_link(fpath, content):
    """path to the link, and the path link is pointing"""
    with open(fpath, "w") as file:
        file.write(content)


@contextmanager
def load_data_gpu(fpath, ext=None):
    """Load data into GPU memory with automatic cleanup."""
    data = cp.asarray(fits.getdata(fpath, ext=ext), dtype="float32")
    try:
        yield data  # Provide the loaded data to the block
    finally:
        del data  # Free GPU memory when the block is exited
        gc.collect()  # Force garbage collection
        cp.get_default_memory_pool().free_all_blocks()

class FileCreationHandler(FileSystemEventHandler):
    def __init__(self, target_file):
        self.target_file = os.path.basename(target_file)
        self.created = False
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(self.target_file):
            self.created = True

def wait_for_masterframe(file_path, timeout=1800):
    """
    Wait for a file to be created using watchdog with timeout.
    
    Args:
        file_path (str): Path to the file to watch for
        timeout (int): Maximum time to wait in seconds (default: 1800 seconds / 30 minutes)
    
    Returns:
        bool: True if file was created, False if timeout occurred
    """
    # First check if file already exists
    if os.path.exists(file_path):
        return True

    directory = os.path.dirname(file_path) or '.'
    handler = FileCreationHandler(file_path)
    observer = Observer()
    observer.schedule(handler, directory, recursive=False)
    observer.start()
    
    try:
        start_time = time.time()
        while not handler.created:
            if time.time() - start_time > timeout:
                return False
            time.sleep(1)
        return True
    finally:
        observer.stop()
        observer.join()

def read_link(link, timeout=1800):
    """
    Check if the link exists using watchdog, wait for it if it doesn't, and then read its content.

    Args:
        link (str): The file path to check and read.
        timeout (int, optional): Maximum time (in seconds) to wait for the file. Defaults to 1200.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file is not found within the timeout period.
        KeyboardInterrupt: If the user interrupts the waiting process.
    """
    try:
        # Use wait_for_masterframe to watch for the file
        if not wait_for_masterframe(link, timeout=timeout):
            raise FileNotFoundError(
                f"File '{link}' was not created within {timeout} seconds."
            )
        
        # Small delay to ensure file is fully written
        time.sleep(0.1)
        
        # Read and return the file content
        with open(link, "r") as f:
            return f.read().strip()
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt while watching for a link. Exiting...")
        raise  # Re-raise the exception to terminate the following processes

def link_to_file(link):
    """Reformat link filename, not reading it"""
    import re

    pattern = r"\.link$"
    if re.match(pattern, link):
        return os.path.splitext(link)[0] + ".fits"
    else:
        raise ValueError("Not a link")

def search_with_date_offsets(template, max_offset=300, future=False):
    """
    Search for files based on a template, modifying embedded dates with offsets.
    future=False includes the current date

    Args:
        template (str): Template string with embedded dates (e.g., "/path/.../2025-01-01/.../20250102/...").
        max_offset (int, optional): Maximum number of days to offset (both positive and negative). Defaults to 2.

    Returns:
        str: A path to a closest existing master calibration frame file.
    """
    import re
    from datetime import datetime, timedelta

    # Regex to match dates in both YYYY-MM-DD and YYYYMMDD formats
    date_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}|\d{8}")

    # Extract all date strings from the template
    dates_in_template = date_pattern.findall(template)
    if not dates_in_template:
        raise ValueError("No date found in the template string.")

    date_night, date_utc = sorted(set(dates_in_template))

    # Parse dates into datetime objects
    date_night_format = "%Y-%m-%d"
    date_utc_format = "%Y%m%d"
    date_night_dt = datetime.strptime(date_night, date_night_format)
    date_utc_dt = datetime.strptime(date_utc, date_utc_format)

    if future:
        # Generate symmetric offsets: -1, +1, -2, +2, ..., up to max_offset
        offsets = [offset for i in range(1, max_offset + 1) for offset in (-i, i)]
    else:
        offsets = [-i for i in range(1, max_offset + 1)]
        offsets = [0] + offsets  # Include the original dates (offset 0)

    # Iterate through offsets
    for offset in offsets:
        # Adjust both dates by the offset
        adjusted_date_night_dt = date_night_dt + timedelta(days=offset)
        adjusted_date_utc_dt = date_utc_dt + timedelta(days=offset)

        # Format the adjusted dates
        adjusted_date_night = adjusted_date_night_dt.strftime(date_night_format)
        adjusted_date_utc = adjusted_date_utc_dt.strftime(date_utc_format)

        # Replace both dates in the template
        modified_path = template.replace(date_night, adjusted_date_night).replace(
            date_utc, adjusted_date_utc
        )

        # Check if the modified path exists
        if os.path.exists(modified_path):
            return modified_path


def write_IMCMB_to_header(header, inputlist, full_path=False):
    """this function was copied from the package eclaire"""
    if inputlist is not None:
        llist = len(inputlist)
        if llist <= 999:
            key = "IMCMB{:03d}"
        else:
            key = "IMCMB{:03X}"
            comment = "IMCMB keys are written in hexadecimal."
            # header.append("COMMENT", comment)  # original eclaire line
            header.add_comment(comment)
        for i, f in enumerate(inputlist, 1):
            header[key.format(i)] = f if full_path else os.path.basename(f)
    return header


# def calculate_average_date_obs(date_obs_list):
#     import numpy as np
#     from astropy.time import Time

#     t = Time(date_obs_list, format="isot", scale="utc")
#     avg_time = np.mean(t.jd)
#     avg_time = Time(avg_time, format="jd", scale="utc")

#     # 'YYYY-MM-DDTHH:MM:SS'
#     avg_time_str = avg_time.isot

#     return avg_time_str


# def isot_to_mjd(time):  # 20181026 to 2018-10-26T00:00:00:000 to MJD form
#     from astropy.time import Time

#     yr = time[0:4]  # year
#     mo = time[4:6]  # month
#     da = time[6:8]  # day
#     isot = yr + "-" + mo + "-" + da + "T00:00:00.000"  # 	ignore hour:min:sec
#     t = Time(isot, format="isot", scale="utc")  # 	transform to MJD
#     return t.mjd
