import os, sys
from datetime import date, datetime
from pathlib import Path

path_thisfile = Path(__file__).resolve()
path_root = path_thisfile.parent.parent.parent  # Careful! not a str
path_src = path_root / 'src'
if path_src not in map(Path, sys.path):
	sys.path.append(str(path_src))

def log2tmp(command, label):
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	path_tmp = path_root / 'tmp'
	if not path_tmp.exists():
		path_tmp.mkdir()
	sexlog = str(path_tmp / f"{label}_{timestamp}.log")
	# stderr is logged with stdout
	new_com = f"{command} > {sexlog} 2>&1"
	return new_com

PATH_ROOT = path_root


def find_gppy_gpu_src(depth=3):
    """Searches up and down 3 levels from the CWD for the 'gppy-gpu/src' directory."""
    cwd = Path(os.getcwd()).resolve()
    search_dir = 'gppy-gpu/src'

    # Search upwards from the CWD
    for up_level in range(depth + 1):
        try:
            search_path_up = cwd.parents[up_level] / search_dir
            if search_path_up.exists() and search_path_up.is_dir():
                return search_path_up
        except IndexError:
            # Stop when trying to access beyond the root directory
            break

    # Search downwards from the CWD (within depth)
    for root, dirs, _ in os.walk(cwd):
        current_depth = len(Path(root).relative_to(cwd).parts)
        if current_depth <= depth:
            # Now check if the full path contains the 'gppy-gpu/src'
            search_path_down = Path(root) / search_dir
            if search_path_down.exists() and search_path_down.is_dir():
                return search_path_down

    # If not found
    return None