import os
import fnmatch
from .const import RAWDATA_DIR
from tqdm import tqdm

def query_observations(include_keywords, exclude_keywords=None, copy_file=False, calibration=True, save_path="./"):
    """
    Recursively searches for .fits files in RAWDATA_DIR.
    
    Files are returned if they:
    - contain at least one of the include_keywords (if provided), and
    - do not contain any of the exclude_keywords (if provided).

    Parameters:
    include_keywords (list of str): Keywords that must appear in the file path or name.
    exclude_keywords (list of str): Keywords that must not appear in the file path or name.
                                    Default is ["bias", "dark", "flat"].

    Returns:
    list: List of paths to matching FITS files.
    """
    if exclude_keywords is None:
        exclude_keywords = ["bias", "dark", "flat"]

    matching_files = []

    for dirpath, _, filenames in os.walk(RAWDATA_DIR):
        for filename in fnmatch.filter(filenames, "*.fits"):
            full_path = os.path.join(dirpath, filename)
            full_path_lower = full_path.lower()
            
            # Check for exclude keywords first
            if any(keyword.lower() in full_path_lower for keyword in exclude_keywords):
                continue
            
            # Check for include keywords, if provided
            if any(keyword.lower() not in full_path_lower for keyword in include_keywords):
                continue

            matching_files.append(full_path)
    
    if copy_file:
        if calibration:
            from .data import ObservationData
            for full_path in tqdm(matching_files):
                ObservationData(full_path).run_calibaration(save_path=save_path, verbose=False)
        else:
            matching_files = [os.path.copy(full_path) for full_path in matching_files]

    return matching_files

