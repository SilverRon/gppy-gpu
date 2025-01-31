#	Library
import os, sys, glob, subprocess
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
#	Astropy
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord
from astropy.time import Time
#	
from ccdproc import ImageFileCollection
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings(action='ignore')
import time
#
path_data = "/large_data/obsdata/7DT01/2023-10-13"
ic = ImageFileCollection(path_data, glob_include='*.fits', keywords='*')
#
print("="*30)
print()
print(f"{len(ic.files)} FRAMES FOUND")
print()
print("="*30)
print()
#	Bias
biasfiles = ic.filter(imagetyp='BIAS').files
print("-"*30)
print(f"{len(biasfiles)} BIAS FRAMES FOUND")
print("-"*30)
print()
#	Dark
darkfiles = ic.filter(imagetyp='DARK').files
print("-"*30)
print(f"{len(darkfiles)} DARK FRAMES FOUND")
print("-"*30)
dark_exptime_arr = np.unique(ic.filter(imagetyp='DARK').summary['exptime']) 
for dark_expt in dark_exptime_arr:
	_darkfiles = ic.filter(imagetyp='DARK', exptime=dark_expt).files
	print(f"- {dark_expt}s\t: {len(_darkfiles)}")
print("-"*30)
print()

#	Flat
flatfiles = ic.filter(imagetyp='FLAT').files
print("-"*30)
print(f"{len(flatfiles)} FLAT FRAMES FOUND")
print("-"*30)
flat_filter_arr = np.unique(ic.filter(imagetyp='FLAT').summary['filter'])
for flat_filter in flat_filter_arr:
	_flatfiles = ic.filter(imagetyp='FLAT', filter=flat_filter).files
	print(f"- {flat_filter}\t: {len(_flatfiles)}")
print("-"*30)
print()
#	LIGHT
lightfiles = ic.filter(imagetyp='LIGHT').files
print("-"*30)
print(f"{len(lightfiles)} LIGHT FRAMES FOUND")
print("-"*30)
light_filter_arr = np.unique(ic.filter(imagetyp='LIGHT').summary['filter'])
for light_filter in light_filter_arr:
	_lightfiles = ic.filter(imagetyp='LIGHT', filter=light_filter).files
	print(f"- {light_filter}\t: {len(_lightfiles)}")
print("-"*30)
print()