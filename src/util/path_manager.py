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