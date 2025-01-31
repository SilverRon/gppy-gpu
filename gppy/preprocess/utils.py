import os
import time
import gc
import numpy as np
import cupy as cp
from astropy.io import fits
from contextlib import contextmanager


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
        cp._default_memory_pool.free_all_blocks()  # Optional: Clear memory pool


def read_link(link, timeout=1200, interval=10):
    """
    Check if the link exists, wait for it if it doesn't, and then read its content.

    Args:
        link (str): The file path to check and read.
        timeout (int, optional): Maximum time (in seconds) to wait for the file. Defaults to 1200.
        interval (int, optional): Time interval (in seconds) between checks. Defaults to 10.

    Returns:
        str: The content of the file.

    Raises:
        FileNotFoundError: If the file is not found within the timeout period.
    """
    elapsed_time = 0

    # Wait and watch for the file
    try:
        while not os.path.exists(link):
            if elapsed_time >= timeout:
                raise FileNotFoundError(
                    f"File '{link}' was not created within {timeout} seconds."
                )
            time.sleep(interval)
            elapsed_time += interval
    except KeyboardInterrupt:
        print("KeyboardInterrupt while watching for a link. Exiting...")
        raise  # Re-raise the exception to terminate the followings

    # Read and return the file content
    with open(link, "r") as f:
        return f.read().strip()


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
            header.append("COMMENT", comment)
        for i, f in enumerate(inputlist, 1):
            header[key.format(i)] = f if full_path else os.path.basename(f)
    return header


#


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


# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# import os
# import time


# class FileCreatedHandler(FileSystemEventHandler):
#     def __init__(self, file, callback):
#         self.file = file
#         self.callback = callback

#     def on_created(self, event):
#         if event.src_path == self.file:
#             print(f"File created: {self.file}")
#             self.callback(self.file)


# def load_pointer_file(file):
#     def process_file(file):
#         with open(file, "r") as f:
#             mbias_file = f.read().strip()
#         # strip() removes trailing whitespace or newlines
#         print(f"Configuration updated with: {mbias_file}")
#         observer.stop()

#     # Check if the file already exists
#     if os.path.exists(file):
#         process_file(file)
#         return

#     # Set up the observer
#     event_handler = FileCreatedHandler(file, process_file)
#     observer = Observer()
#     directory = os.path.dirname(file) or "."
#     observer.schedule(event_handler, directory, recursive=False)

#     try:
#         print(f"Watching for the file: {file}")
#         observer.start()
#         while observer.is_alive():
#             time.sleep(1)  # Keeps the script running efficiently
#     except KeyboardInterrupt:
#         print("Stopping pointer file observer...")
#         observer.stop()
#     observer.join()
