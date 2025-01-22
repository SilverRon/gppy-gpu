#	7DT_Routine.py
# - Automatically Process the image data from 7DT facilities and search for transients
# - This is an advanced version of `gpPy`
# - Author: Gregory S.H. Paek (23.10.10)
#%%
#------------------------------------------------------------
#	Library
#------------------------------------------------------------
# Built-in packages
import os
import sys
import re
import time
import gc
import glob
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import repeat
import subprocess
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import warnings
warnings.filterwarnings(action='ignore')
#------------------------------------------------------------
# Third-party packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.interactiveshell import InteractiveShell
from ccdproc import ImageFileCollection
import psutil
#	Astropy
from astropy.io import fits
import astropy.io.ascii as ascii
from astropy.wcs import WCS
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord
from astropy.time import Time
#------------------------------------------------------------
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

# Path Setup for Custom Packages
try:
	path_thisfile = Path(__file__).resolve()
	# ABSOLUTE path of gppy-gpu
	Path_root = path_thisfile.parents[2]  # Careful! not a str
	# sys.path.append('../../src')  # Deprecated
	Path_src = Path_root / 'src'
	Path_run = path_thisfile.parent
except NameError:
	# in case for ipython
	Path_src = find_gppy_gpu_src()
	Path_root = Path(Path_src).parent 
	Path_run = Path_root / 'run' / 'routine'

if Path_src not in map(Path, sys.path):
	sys.path.append(str(Path_src)) 
from preprocess import calib
from util import tool
#------------------------------------------------------------
#	plot setting
#------------------------------------------------------------
plt.ioff()
InteractiveShell.ast_node_interactivity = "last_expr"
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#------------------------------------------------------------
os.environ['TZ'] = 'Asia/Seoul'
time.tzset()
start_localtime = time.strftime('%Y-%m-%d_%H:%M:%S_(%Z)', time.localtime())

#------------------------------------------------------------
#	*Core Variables*
#------------------------------------------------------------
# n_binning = 2
n_binning = 1
verbose_sex = False
slack_report = True
verbose_gpu = True
local_astref = False
debug = False  # True
if debug:
	slack_report = False

#	N cores for Multiprocessing
# try:
# 	ncore = int(sys.argv[2])
# except:
# 	ncore = 2
ncore = 4
print(f"- Number of Cores: {ncore}")

project = '7DT/7DS'
obsmode = 'COMISSION'
print(f"[{project}] {obsmode}")

memory_threshold = 50

#------------------------------------------------------------
#	Ready
#------------------------------------------------------------
#	OBS
# This is for running the script with jupyter
try:
	obs = (sys.argv[1]).upper() # raises IndexError if fed no arguments
	if not bool(re.match(r"7DT\d{2}", obs)):
		raise IndexError("The input is not 7DT##. Switching to Manual Input")
except IndexError as e:
	print('No telescope arg given. Switching to Manual Input') # print(e)
	obs = input(f"7DT## (e.g. 7DT01):").upper()
except Exception as e:
	print(e)
	print(f'Unexpected behavior. Check parameter obs')

print(f'# Observatory : {obs.upper()}')

#------------------------------------------------------------
#	Path
#------------------------------------------------------------
#   Main Paths from path.json
with open(Path_run / 'path.json', 'r') as jsonfile:
	upaths = json.load(jsonfile)

path_base = upaths['path_base']  # '/home/snu/gppyTest_dhhyun/factory'  # '/large_data/factory'
path_obsdata = f'{path_base}/../obsdata' if upaths['path_obsdata'] == '' else upaths['path_obsdata']
path_processed = f'{path_base}/../processed_{n_binning}x{n_binning}_gain2750' if upaths['path_processed'] == '' else upaths['path_processed']
path_refcat = f'{path_base}/ref_cat' if upaths['path_refcat'] == '' else upaths['path_refcat']
path_ref_scamp = f'{path_base}/ref_scamp' if 'path_ref_scamp' not in upaths or upaths['path_ref_scamp'] == '' else upaths['path_ref_scamp']
path_log = f'{path_base}/log/{obs.lower()}.log' if 'key' not in upaths or upaths['key'] == '' else upaths['path_log']
	

# path_gal = f'{path_base}/../processed'
# path_gal = f'{path_base}/../processed_{n_binning}x{n_binning}'
# path_refcat = '/data4/gecko/factory/ref_frames/LOAO'
#------------------------------------------------------------

# path_ref = f'{path_base}/ref_frame/{obs.upper()}'
path_ref = f'{path_base}/ref_frame'
path_factory = f'{path_base}/{obs.lower()}'
# path_save = f'/data6/bkgdata/{obs.upper()}'
# path_log = 

# path_config = '/home/paek/config'
path_config = str(Path_root / 'config')  # '../../config'
path_keys = path_config  # f'../../config'
# path_default_gphot = f'{path_config}/gphot.{obs.lower()}_{n_binning}x{n_binning}.config'
path_default_gphot = f'{path_config}/gphot.7dt_{n_binning}x{n_binning}.config'
path_mframe = f'{path_base}/master_frame_{n_binning}x{n_binning}_gain2750'
# path_calib = f'{path_base}/calib'
#------------------------------------------------------------
#	Codes
#------------------------------------------------------------
# path_phot_sg = './phot/gregoryphot_2021.py'
path_phot_mp = str(Path_src / 'phot/gregoryphot_7DT_NxN.py')  # './phot/gregoryphot_7DT_NxN.py'
path_phot_sub = str(Path_src / 'phot/gregorydet_7DT_NxN.py')
path_subtraction = str(Path_src / "util/gregorysubt_7DT.py")
path_find = str(Path_src / 'phot/gregoryfind_7DT.py')
#------------------------------------------------------------
path_raw = f'{path_obsdata}/{obs.upper()}'
# rawlist = sorted(glob.glob(f'{path_raw}/2???-??-??_gain2750'))
rawlist = [os.path.abspath(path) for path in sorted(glob.glob(f'{path_raw}/2???-??-??_gain2750'))]
#------------------------------------------------------------
path_obs = f'{path_config}/obs.dat'
path_changehdr = f'{path_config}/changehdr.dat'
path_alltarget = f'{path_config}/alltarget.dat'
path_skygrid = str(Path_root / "data/skygrid/7DT")  # "../../data/skygrid/7DT"
skygrid_table = Table.read(f"{path_skygrid}/skygrid.fits")
tile_name_pattern = r"T\d{5}$"
# skygrid_table = Table.read(f"{path_skygrid}/displaycenter.txt", format='ascii')
# skygrid_table['tile'] = [f"T{val:0>5}" for val in skygrid_table['id']]
# skygrid_table.write(f"{path_skygrid}/skygrid.fits")
ccdinfo = tool.getccdinfo(obs, path_obs)
# Path for the astrometry
path_cfg = '/usr/local/astrometry/etc/astrometry.cfg'
#------------------------------------------------------------
#	Make Folders
#------------------------------------------------------------
if not os.path.exists(path_base):
	os.makedirs(path_base)
if not os.path.exists(path_ref):
	os.makedirs(path_ref)
if not os.path.exists(path_factory):
	os.makedirs(path_factory)
if not os.path.exists(path_refcat):
	os.makedirs(path_refcat)
if not os.path.exists(path_ref_scamp):
	os.makedirs(path_ref_scamp)
# if not os.path.exists(path_save):
# 	os.makedirs(path_save)
for imagetyp in ['zero', 'flat', 'dark']:
	path_mframe_imagetyp = f"{path_mframe}/{obs}/{imagetyp}"
	if not os.path.exists(path_mframe_imagetyp):
		print(f"mkdir {path_mframe_imagetyp}")
		os.makedirs(path_mframe_imagetyp)

# revision
if not os.path.exists(path_log):
	print(f"Creating {path_log}")

	path_log_parent = Path(path_log).parent
	if not path_log_parent.exists():
		path_log_parent.mkdir(parents=True)

	f = open(path_log, 'w')
	# columns
	f.write('date,start,end,note\n')  # f.write('date\n19941026')
	# example line
	f.write('/lyman/data1/obsdata/7DT01/1994-10-26_1x1_gain2750,1994-10-26_00:00:00_(KST),1994-10-26_00:01:00_(KST),-\n')
	f.close()
#------------------------------------------------------------
#	Constant
#------------------------------------------------------------

#------------------------------------------------------------
#	Table
#------------------------------------------------------------
# logtbl = ascii.read(path_log)#, delimiter=',', format='csv')
logtbl = Table.read(path_log, format='csv')
datalist = np.copy(logtbl['date'])
obstbl = ascii.read(path_obs)
hdrtbl = ascii.read(path_changehdr)
alltbl = ascii.read(path_alltarget)
keytbl = ascii.read(f'{path_keys}/keys.dat')
#------------------------------------------------------------
#	Time Log Table
#------------------------------------------------------------
timetbl = Table()
# timetbl['process'] = [
# 	'master_frame_bias',
# 	'master_frame_dark',
# 	'master_frame_flat',
# 	'data_reduction',
# 	'astrometry',
# 	'cr_removal',
# 	'photometry',
# 	'image_stack',
# 	'photometry_com',
# 	'subtraction',
# 	'photometry_sub',
# 	'transient_search',
# 	'total',
# ]
timetbl['process'] = [
	'master_frame_bias',
	'master_frame_dark',
	'master_frame_flat',
	'data_reduction',
	'astrometry_solve_field',
	'pre_source_extractor',
	'astrometry_scamp',
	'missfits',
	'get_polygon_info',
	'cr_removal',
	'photometry',
	'image_stack',
	'photometry_com',
	'subtraction',
	'photometry_sub',
	'transient_search',
	'data_transfer',
	'total',
]
timetbl['status'] = False
timetbl['time'] = 0.0 * u.second

#============================================================
#------------------------------------------------------------
#	Main Body
#------------------------------------------------------------
#============================================================
# If sys.argv[2]: path is declared
if len(sys.argv) > 2:
	path_new = os.path.abspath(sys.argv[2])
	if not Path(path_new).exists():
		print('Provided path does not exist')
		sys.exit()
else:
	# Find new data
	datalist = np.copy(logtbl['date'])
	rawlist = sorted(glob.glob(f'{path_raw}/2???-??-??_gain2750'))
	newlist = [os.path.abspath(s) for s in rawlist if (s not in datalist) and (s+'/' not in datalist)]
	if len(newlist) == 0:
		print('No new data')
		sys.exit()
	else:
		for ff, folder in enumerate(newlist):
			print(f"[{ff:0>2}] {folder}")
		user_input = input("Path or Index To Process:")
		# input --> digit (index)
		if user_input.isdigit():
			index = int(user_input)
			path_new = newlist[index]
		# input --> path (including '/')
		elif '/' in user_input:
			path_new = user_input
		# other
		else:
			print("Wrong path or index")
			sys.exit()

print(f"Selected Path: {path_new}")


# # revision. s for string
# newlist = [os.path.abspath(s) for s in rawlist if (s not in datalist) & (s+'/' not in datalist)]
# if len(newlist) == 0:
# 	print('No new data')
# 	sys.exit()
# else:
# 	if len(sys.argv) < 3:
# 		for ff, folder in enumerate(newlist):
# 			print(f"[{ff:0>2}] {folder}")

"""
# path = newlist[-1]
# path = newlist[3]
# path_raw = newlist[2]
# path_raw = newlist[0]

"""
# try:
# 	path_new = os.path.abspath(sys.argv[2])
# 	#revision
# 	if not Path(path_new).exists():
# 		print('Provided path does not exist')
# 		raise IndexError
# except:
# 	user_input = input("Path or Index To Process:")
# 	# 입력값이 숫자인 경우
# 	if user_input.isdigit():
# 		index = int(user_input)
# 		path_new = newlist[index]
# 	# 입력값이 경로 문자열인 경우 (여기서는 간단하게 '/'를 포함하는지만 확인)
# 	elif '/' in user_input:
# 		path_new = user_input
# 	# 그 외의 경우
# 	else:
# 		print("Wrong path or index")
# 		sys.exit()

# path = '/large_data/factory/../obsdata/7DT01/2023-00-00'
# path = '/large_data/factory/../obsdata/7DT01/2023-10-15'
tdict = dict()
starttime = time.time()

#------------------------------------------------------------
#	Log All Prints
#------------------------------------------------------------
path_save_all_log = f"{os.path.dirname(path_log)}/{obs.upper()}"
if not os.path.exists(path_save_all_log):
	print(f"Creat {path_save_all_log}")
	os.makedirs(path_save_all_log)

#	Skip Test Data
if '2023-00' in path_new:
	print(f"This is Test Data --> Skip All Report")
	pass
else:
	# sys.stdout = open(f'{path_save_all_log}/{os.path.basename(path_new)}_all.log', 'w')
	pass

path_data = f'{path_factory}/{os.path.basename(path_new)}'
print(f"="*60)
print(f"`gpPy`+GPU Start: {path_data} ({start_localtime})")
print(f"-"*60)

if os.path.exists(path_data):
	#	Remove old folder and re-copy folder
	# rmcom = f'rm -rf {path_data}'
	# print(rmcom)
	# os.system(rmcom)
	print(f"Removing existing factory dir: {path_data}")
	shutil.rmtree(path_data)


if not os.path.exists(path_data):
	os.makedirs(path_data)
obsinfo = calib.getobsinfo(obs, obstbl)



ic1 = ImageFileCollection(path_new, keywords='*')
# Tile Selection
if 'tile' not in upaths or upaths['tile'] == "":
	print('Processing all tiles for the given date')
	pass
else:
	try:
		tile = f"{int(upaths['tile']):05}"
		file_list = [str(f) for f in ic1.summary['file']]
		pattern = re.compile(fr'(?=.*{tile}|FLAT|DARK|BIAS)')
		filtered_files = [f for f in file_list if pattern.search(f)]
		if len(filtered_files) == 0:
			print(f'No T{tile:05} in the given date')
			sys.exit(1)
		ic1 = ImageFileCollection(path_new, filenames=filtered_files)
		print("\n#------------------------------------------------------------",
			  f"#	T{tile:05} Selected",
			  "#-----------------------------------------------------------\n",
			  sep = '\n',
		)
	except Exception as e:
		print('Tile Selection Failed\nProcessing All.', e)


#------------------------------------------------------------
#	Count the number of Light Frame
#------------------------------------------------------------
try:
	allobjarr = ic1.filter(imagetyp='LIGHT').summary['object']
	objarr = np.unique(ic1.filter(imagetyp='LIGHT').summary['object'])
	nobj = len(allobjarr)
	#	LIGHT FRAME TABLE
	# objtbl = Table()
	# objtbl['raw_image'] = [os.path.basename(inim) for inim in ic1.filter(imagetyp='LIGHT').files]
except:
	nobj = 0
#------------------------------------------------------------
#	Bias Number
#------------------------------------------------------------
try:
	bimlist = list(ic1.filter(imagetyp='Bias').summary['file'])
	biasnumb = len(bimlist)
	print(f"{biasnumb} Bias Frames Found")
except:
	biasnumb = 0
#------------------------------------------------------------
#	Dark Number
#------------------------------------------------------------
try:
	darkexptimelist = sorted(list(set(ic1.filter(imagetyp='dark').summary['exptime'])))
	darknumb = len(darkexptimelist)
except:
	darknumb = 0
#------------------------------------------------------------
#	Flat Number
#------------------------------------------------------------
try:
	filterlist = list(np.unique(ic1.filter(imagetyp='FLAT').summary['filter']))
	print(f"{len(filterlist)} filters found")
	print(f"Filters: {filterlist}")
	flatnumb = len(filterlist)
except:
	print(f"There is no flat frame")
	flatnumb = 0
#------------------------------------------------------------
# ### Marking the `GECKO` data


# testobj = 'S190425z'
# project = "7DT"
# obsmode = "MONITORING" # Default
# if 'OBJECT' in ic1.summary.keys():
# 	for obj in ic1.filter(imagetyp='LIGHT').summary['object']:
# 		if 'MS' in obj[:2]: # MS230425 (test event)
# 			print(obj)
# 			project = "GECKO"
# 			obsmode = "TEST"
# 		elif 'S2' in obj[:2]: # S230425 (super event)
# 			print(obj)
# 			project = "GECKO"
# 			obsmode = "FOLLOWUP" # Follow-up
# 		else:
# 			pass
# else:
# 	pass


# - Slack notification

if slack_report:
	OAuth_Token = keytbl['key'][keytbl['name']=='slack'].item()

	channel = '#pipeline'
	text = f'[`gpPy`+GPU/{project}-{obsmode}] Start Processing {obs} {os.path.basename(path_new)} Data ({nobj} objects) with {ncore} cores'

	param_slack = dict(
		token = OAuth_Token,
		channel = channel,
		text = text,
	)
	tool.slack_bot(**param_slack)
#
if len(glob.glob(f"{path_new}/*.fits")) == 0:

	end_localtime = time.strftime('%Y-%m-%d_%H:%M:%S_(%Z)', time.localtime())
	note = "No Fits Files"
	#	Logging
	log_text = f"{path_new},{start_localtime},{end_localtime},{note}\n"
	with open(path_log, 'a') as file:
		file.write(f"{log_text}")

	print(f"[EXIT!] {note}")
	sys.exit()


# %%
#------------------------------------------------------------
#	Preprocessing
#------------------------------------------------------------
from refactor import preprocessing as pp

ic_fdzobj, fdzimlist, objtbl = pp.preproc(obs, path_data, path_mframe, path_log,
										  verbose_gpu, path_new, timetbl,
										  biasnumb, bimlist, ic1, darknumb, darkexptimelist,
										  nobj, flatnumb, filterlist, start_localtime, objarr)

# %%

#------------------------------------------------------------
#	ASTROMETRY
#------------------------------------------------------------
from refactor import astrometry as ast

hdr = ast.astrom(path_data, ic_fdzobj, obsinfo, n_binning, path_config, memory_threshold,
           		 ncore, fdzimlist, timetbl, objtbl, verbose_sex, objarr, local_astref,
           		 tile_name_pattern, path_ref_scamp, upaths, debug, obs)


# %%

#------------------------------------------------------------
#	Photometry
#------------------------------------------------------------
t0_photometry = time.time()
print('#\tPhotometry')
path_infile = f'{path_data}/{os.path.basename(path_default_gphot)}'
path_new_gphot = f'{os.path.dirname(path_infile)}/gphot.config'

#	Copy default photometry configuration
cpcom = f'cp {path_default_gphot} {path_new_gphot}'
print(cpcom)
os.system(cpcom)

#	Read default photometry configuration
f = open(path_default_gphot, 'r')
lines = f.read().splitlines()
f.close()

#	Write photometry configuration for a single exposure frame
g = open(path_new_gphot, 'w')
for line in lines:
	if 'imkey' in line:
		line = f'imkey\t{path_data}/calib*0.fits'
	else:
		pass
	g.write(line+'\n')
g.close()

path_phot = path_phot_mp
#	Execute
#	(e.g. com = f'python {path_phot} {path_data} 1')
com = f'python {path_phot} {path_data} {ncore} {n_binning}'
print(com)
os.system(com)

delt_photometry = time.time() - t0_photometry
timetbl['status'][timetbl['process']=='photometry'] = True
timetbl['time'][timetbl['process']=='photometry'] = delt_photometry


#	IMAGE COMBINE

def calc_alignment_shift2(inim1, inim2, ra_dec_order=1):

	header1 = fits.getheader(inim1)
	header2 = fits.getheader(inim2)

	# 두 이미지의 WCS 객체를 생성합니다. (이 부분은 실제 FITS 파일에서 읽어와야 합니다.)
	wcs1 = WCS(header1)  # 첫 번째 이미지의 FITS 헤더
	wcs2 = WCS(header2)  # 두 번째 이미지의 FITS 헤더

	# 첫 번째 이미지의 참조 픽셀 좌표 (CRPIX)와 천문학적 좌표 (CRVAL)
	# crpix1_x1, crpix1_y1 = wcs1.wcs.crpix
	crval1_ra1, crval1_dec1 = wcs1.wcs.crval
	# crval1_ra1, crval1_dec1 = header1['RA'], header1['DEC']

	# 두 번째 이미지의 참조 픽셀 좌표 (CRPIX)와 천문학적 좌표 (CRVAL)
	# crpix2_x1, crpix2_y1 = wcs2.wcs.crpix
	crval2_ra1, crval2_dec1 = wcs2.wcs.crval
	# crval2_ra1, crval2_dec1 = header2['RA'], header2['DEC']

	# 천문학적 좌표 (RA, Dec)를 픽셀 좌표로 변환
	pix1_x, pix1_y = wcs1.all_world2pix(crval2_ra1, crval2_dec1, ra_dec_order)
	pix2_x, pix2_y = wcs2.all_world2pix(crval1_ra1, crval1_dec1, ra_dec_order)

	# x, y shift 값을 계산
	shift_x = pix2_x - pix1_x
	shift_y = pix2_y - pix1_y

	# shift_x, shift_y = 100, 100
	# print(f"x shift: {shift_x}, y shift: {shift_y}")

	# shifts = np.array(
	# 	[
	# 		[0, 0],
	# 		[-shift_x, -shift_y],
	# 	]
	# )
	return [-shift_x, -shift_y]


def calc_mean_dateloc(dateloclist):

	# 문자열을 datetime 객체로 변환
	datetime_objects = [datetime.fromisoformat(t) for t in dateloclist]

	# datetime 객체를 POSIX 시간으로 변환
	posix_times = [dt.timestamp() for dt in datetime_objects]

	# 평균 POSIX 시간 계산
	mean_posix_time = np.mean(posix_times)

	# 평균 POSIX 시간을 datetime 객체로 변환
	mean_datetime = datetime.fromtimestamp(mean_posix_time)

	# 필요한 경우, datetime 객체를 ISOT 형식의 문자열로 변환
	mean_isot_time = mean_datetime.isoformat()
	return mean_isot_time


def calc_alignment_shift(incat1, incat2, matching_sep=1,):
	intbl1 = Table.read(incat1, format='ascii.sextractor')
	intbl2 = Table.read(incat2, format='ascii.sextractor')

	c1 = SkyCoord(ra=intbl1['ALPHA_J2000'], dec=intbl1['DELTA_J2000'])
	c2 = SkyCoord(ra=intbl2['ALPHA_J2000'], dec=intbl2['DELTA_J2000'])

	indx, sep, _ = c1.match_to_catalog_sky(c2)

	_mtbl = hstack([intbl1, intbl2[indx]])
	_mtbl['sep'] = sep.arcsec
	mtbl = _mtbl[_mtbl['sep']<matching_sep]

	xdifarr = mtbl['X_IMAGE_2']-mtbl['X_IMAGE_1']
	ydifarr = mtbl['Y_IMAGE_2']-mtbl['Y_IMAGE_1']

	xshift = np.median(xdifarr)
	yshift = np.median(ydifarr)

	# xdifstd = np.std(xdifarr)
	# ydifstd = np.std(ydifarr)

	return [xshift, yshift]


def group_images(time_list, threshold):
	groups = []
	index_groups = []
	current_group = [time_list[0]]
	current_index_group = [0]  # 시작 인덱스

	for i in range(1, len(time_list)):
		if time_list[i] - time_list[i-1] <= threshold:
			current_group.append(time_list[i])
			current_index_group.append(i)
		else:
			groups.append(current_group)
			index_groups.append(current_index_group)
			current_group = [time_list[i]]
			current_index_group = [i]

	groups.append(current_group)  # 마지막 그룹을 추가
	index_groups.append(current_index_group)  # 마지막 인덱스 그룹을 추가
	return groups, index_groups

#	Image Stacking
t0_image_stack = time.time()


keywords_to_add = [
	"IMAGETYP",
	# "EXPOSURE",
	# "EXPTIME",
	# "DATE-LOC",
	# "DATE-OBS",
	"XBINNING",
	"YBINNING",
	"GAIN",
	"EGAIN",
	"XPIXSZ",
	"YPIXSZ",
	"INSTRUME",
	"SET-TEMP",
	"CCD-TEMP",
	"TELESCOP",
	"FOCALLEN",
	"FOCRATIO",
	"RA",
	"DEC",
	# "CENTALT",
	# "CENTAZ",
	# "AIRMASS",
	"PIERSIDE",
	"SITEELEV",
	"SITELAT",
	"SITELONG",
	"FWHEEL",
	"FILTER",
	"OBJECT",
	"OBJCTRA",
	"OBJCTDEC",
	"OBJCTROT",
	"FOCNAME",
	"FOCPOS",
	"FOCUSPOS",
	"FOCUSSZ",
	"ROWORDER",
	# "COMMENT",
	"_QUINOX",
	"SWCREATE"
]



image_stack_skip_tbl = Table.read(f'{path_config}/object_to_skip_stacking.txt', format='ascii')
image_stack_skip_list = list(image_stack_skip_tbl['object'])

ic_cal = ImageFileCollection(path_data, glob_include='calib*0.fits', keywords='*')

#	Time to group
threshold = 300./(60*60*24) # [MJD]

t_group = 0.5/24 # 30 min

grouplist = []
stacked_images = []
for obj in np.unique(ic_cal.summary['object']):
	if obj in image_stack_skip_list:
		print(f"Skip Image Stacking Process for {obj}")
	else:
		for filte in np.unique(ic_cal.filter(object=obj).summary['filter']):

			print(f"[{obj},{filte}]==============================")
			
			checklist = []
			_imagearr = ic_cal.filter(object=obj, filter=filte).summary['file']
			#	Check Number of All Images


			if len(_imagearr) > 0:
				_mjdarr = Time(ic_cal.filter(object=obj, filter=filte).summary['date-obs'], format='isot').mjd

				groups, index_groups = group_images(
					time_list=_mjdarr,
					threshold=threshold
					)

				print("Groups:", groups)
				print("Index Groups:", index_groups)

				for gg, (group, indx_group) in enumerate(zip(groups, index_groups)):
					print(f"[{gg:0>2}] {indx_group}")

					if len(group) == 0:
						print(f"{_imagearr[indx_group][0]} Single image exists")
					elif len(group) > 1:
						grouped_images = _imagearr[indx_group]
						print(f"{len(grouped_images)} images to stack")
						for ii, inim in enumerate(grouped_images):
							if ii == 0:	
								print(f"- {ii:0>4}: {inim} <-- Base Image")
							else:
								print(f"- {ii:0>4}: {inim}")
						
						#	Base Image for the Alignment
						baseim = grouped_images[0]
						basehdr = fits.getheader(baseim)
						# print(f"BASE IMAGE: {baseim}")
						basecat = baseim.replace('fits', 'cat')
						path_imagelist = f"{os.path.dirname(baseim)}/{os.path.basename(baseim).replace('fits', 'image.list')}"

						#	Images to Combine for SWarp
						f = open(path_imagelist, 'w')
						for inim in grouped_images:
							f.write(f"{inim}\n")
						f.close()

						#	Get Header info
						dateloclist = []
						mjdlist = []
						exptimelist = []
						airmasslist = []
						altlist = []
						azlist = []
						for _inim in grouped_images:
							#	Open Image Header
							with fits.open(inim) as hdulist:
								# Get the primary header
								header = hdulist[0].header
								mjdlist.append(Time(header['DATE-OBS'], format='isot').mjd)
								exptimelist.append(header['EXPTIME'])
								airmasslist.append(header['AIRMASS'])
								dateloclist.append(header['DATE-LOC'])
								altlist.append(header['CENTALT'])
								azlist.append(header['CENTAZ'])
						exptime_combined = tool.convert_number(np.sum(exptimelist))
						mjd_combined = np.mean(mjdlist)
						jd_combined = Time(mjd_combined, format='mjd').jd
						dateobs_combined = Time(mjd_combined, format='mjd').isot
						airmass_combined = np.mean(airmasslist)
						dateloc_combined = calc_mean_dateloc(dateloclist)
						alt_combined = np.mean(altlist)
						az_combined = np.mean(azlist)

						#	Center Coordinate
						#	Tile OBJECT (e.g. T01026)
						if bool(re.match(tile_name_pattern, obj)):
							print(f"{obj} is 7DT SkyGrid. Use Fixed RA, Dec!")
							indx_skygrid = skygrid_table['tile'] == obj
							ra, dec = skygrid_table['ra'][indx_skygrid][0], skygrid_table['dec'][indx_skygrid][0]
							c_tile = SkyCoord(ra, dec, unit=u.deg)

							objra = c_tile.ra.to_string(unit=u.hourangle, sep=':', pad=True)
							objdec = c_tile.dec.to_string(unit=u.degree, sep=':', pad=True, alwayssign=True)
							pass
						#	Non-Tile OBJECT
						else:
							print(f"{obj} is pointed (RA, Dec)")
							objra = header['OBJCTRA']
							objdec = header['OBJCTDEC']

							objra = objra.replace(' ', ':')
							objdec = objdec.replace(' ', ':')
						center = f"{objra},{objdec}"

						datestr, timestr = calib.extract_date_and_time(dateobs_combined)
						comim = f"{path_data}/calib_{obs}_{obj}_{datestr}_{timestr}_{filte}_{exptime_combined}.com.fits"

						#	Image Combine
						t0_com = time.time()
						# swarpcom = f"swarp -c {path_config}/7dt_{n_binning}x{n_binning}.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -RESAMPLE_DIR {path_data} -CENTER_TYPE MANUAL -CENTER {center} -GAIN_KEYWORD EGAIN"
						swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -RESAMPLE_DIR {path_data} -CENTER_TYPE MANUAL -CENTER {center} -GAIN_KEYWORD EGAIN"
						print(swarpcom)
						os.system(swarpcom)

						#	Get Genenral Header from Base Image
						with fits.open(baseim) as hdulist:
							header = hdulist[0].header
							chdr = {key: header.get(key, None) for key in keywords_to_add}

						#	Put General Header Infomation on the Combined Image
						with fits.open(comim) as hdulist:
							data = hdulist[0].data
							header = hdulist[0].header
							for key in list(chdr.keys()):
								header[key] = chdr[key]

						#	Effective EGAIN
						N_combine = len(grouped_images)
						gain_default = hdr['EGAIN']
						effgain = (2/3)*N_combine*gain_default

						#	Additional Header Information
						keywords_to_update = {
							'EGAIN'   : (effgain,          'Effective EGAIN, [e-/ADU] Electrons per A/D unit'),
							'FILTER'  : (filte,            'Active filter name'),
							'DATE-OBS': (dateobs_combined, 'Time of observation (UTC) for combined image'),
							'DATE-LOC': (dateloc_combined, 'Time of observation (local) for combined image'),
							'EXPTIME' : (exptime_combined, '[s] Total exposure duration for combined image'),
							'EXPOSURE': (exptime_combined, '[s] Total exposure duration for combined image'),
							'CENTALT' : (alt_combined,     '[deg] Average altitude of telescope for combined image'),
							'CENTAZ'  : (az_combined,      '[deg] Average azimuth of telescope for combined image'),
							'AIRMASS' : (airmass_combined, 'Average airmass at frame center for combined image (Gueymard 1993)'),
							'MJD'     : (mjd_combined,     'Modified Julian Date at start of observations for combined image'),
							'JD'      : (jd_combined,      'Julian Date at start of observations for combined image'),
						}

						#	Header Update
						with fits.open(comim, mode='update') as hdul:
							# 헤더 정보 가져오기
							header = hdul[0].header

							# 여러 헤더 항목 업데이트
							for key, (value, comment) in keywords_to_update.items():
								header[key] = (value, comment)

							# 변경 사항 저장
							hdul.flush()
						stacked_images.append(comim)
						delt_com = time.time() - t0_com
						print(f"Combied Time: {delt_com:.3f} sec")


delt_image_stack = time.time() - t0_image_stack
timetbl['status'][timetbl['process']=='image_stack'] = True
timetbl['time'][timetbl['process']=='image_stack'] = delt_image_stack

# %%
# ## Photometry for combined images

t0_photometry_com = time.time()
#	Write photometry configuration
h = open(path_new_gphot, 'w')
for line in lines:
	if 'imkey' in line:
		line = f'imkey\t{path_data}/c*com.fits'
	else:
		pass
	h.write(line+'\n')
h.close()
#	Execute
path_phot = path_phot_mp
com = f'python {path_phot} {path_data} {ncore} {n_binning}'
print(com)
os.system(com)
delt_photometry_com = time.time() - t0_photometry_com
timetbl['status'][timetbl['process']=='photometry_com'] = True
timetbl['time'][timetbl['process']=='photometry_com'] = delt_photometry_com
# %%
t0_transient_searh = time.time()
#======================================================================
#	Generate Mask Images
#======================================================================
def create_mask_images(input_image, mask_suffix="mask.fits", force_run=False):
	mask_filename = input_image.replace("fits", mask_suffix)
	if (not os.path.exists(mask_filename)) | (force_run == True):
		data = fits.getdata(input_image)
		mask = np.zeros_like(data, dtype=int)
		mask[data == 0] = 1
		mask[data != 0] = 0
		fits.writeto(mask_filename, mask.astype(np.int8), overwrite=True)
	return mask_filename


def combine_or_mask(in_mask_image, ref_mask_image, mask_suffix="all_mask.fits"):
	inmask = fits.getdata(in_mask_image)
	refmask = fits.getdata(ref_mask_image)
	mask = np.logical_or.reduce([inmask, refmask])
	mask_filename = in_mask_image.replace("mask.fits", mask_suffix)
	fits.writeto(mask_filename, mask.astype(np.int8), overwrite=True)
	return mask_filename

reference_images = []
sci_mask_images = []
ref_mask_images = []
all_mask_images = []

for stack_image in stacked_images:
	part = os.path.basename(stack_image).split("_")
	obj = part[2]
	filte = part[5]
	path_ref_frame = f"{path_ref}/{filte}"
	# _reference_images = []
	# for ref_src in ['7DT', 'PS1']: ref_PS1_T14548_00000000_000000_r_0.fits
	_reference_images_ps1 = glob.glob(f"{path_ref_frame}/ref_PS1_{obj}_*_*_{filte}_0.fits")
	_reference_images_7dt = glob.glob(f"{path_ref_frame}/ref_7DT_{obj}_*_*_{filte}_*.fits")
	_reference_images = _reference_images_7dt + _reference_images_ps1
	_reference_images = [ref for ref in _reference_images if 'mask' not in ref]

	if len(_reference_images) > 0:
		ref_image = _reference_images[0]
		#	Run
		sci_mask_image = create_mask_images(stack_image)
		ref_mask_image = create_mask_images(ref_image)
		all_mask_image = combine_or_mask(sci_mask_image, ref_mask_image, mask_suffix="all_mask.fits")
	else:
		ref_image = None
		sci_mask_image = None
		ref_mask_image = None
		all_mask_image = None
	#	Save
	reference_images.append(ref_image)
	sci_mask_images.append(sci_mask_image)
	ref_mask_images.append(ref_mask_image)
	all_mask_images.append(all_mask_image)
#======================================================================
#	Image Subtraction
#======================================================================
for ss, (inim, refim, inmask_image, refmask_image, allmask_image) in enumerate(zip(stacked_images, reference_images, sci_mask_images, ref_mask_images, all_mask_images)):
	if refim != None:
		#	Subtraction Command
		subtraction_com = f"python {path_subtraction} {inim} {refim} {inmask_image} {refmask_image} {allmask_image}"
		print(subtraction_com)
		os.system(subtraction_com)
		#	Outputs
		hdim = inim.replace("fits", "subt.fits")

		_hcim = f"{os.path.basename(refim).replace('fits', 'conv.fits')}"
		dateobs = os.path.basename(inim).split("_")[3]
		timeobs = os.path.basename(inim).split("_")[4]
		part_hcim = _hcim.split("_")
		part_hcim[3] = dateobs
		part_hcim[4] = timeobs
		hcim = f"{path_data}/{'_'.join(part_hcim)}"

		#	Photometry Command for Subtracted Image
		phot_subt_com = f"python {path_phot_sub} {hdim} {inmask_image}"
		print(phot_subt_com)
		os.system(phot_subt_com)
		#	Transient Search Command --> Skip
		# search_com = f"python {path_find} {inim} {refim} {hcim} {hdim}"
		# print(search_com)
		# os.system(search_com)

delt_transient_searh = time.time() - t0_transient_searh
timetbl['status'][timetbl['process']=='transient_searh'] = True
timetbl['time'][timetbl['process']=='transient_searh'] = delt_transient_searh
# %%
#======================================================================
#	Clean the data
#======================================================================
file_to_remove_pattern_list = [
	#	Reduced Images
	'fdz*',
	#	Reduced Images & WCS Files
	'afdz*',
	#	Zero & Dark Corrected Flat Files
	# 'dz*',
	#	Inverted images
	'*inv*',
	#	Tmp catalogs
	'*.pre.cat',
	'*.image.list',
	#	backup header
	'*.head.bkg',


]

for file_to_remove_pattern in file_to_remove_pattern_list:
	rmcom = f'rm {path_data}/{file_to_remove_pattern}'
	print(rmcom)
	os.system(rmcom)
# %%
#======================================================================
#	File Transfer
#======================================================================
t0_data_transfer = time.time()

# def move_file(file_path, destination_folder):
#     file_name = os.path.basename(file_path)
#     shutil.move(file_path, f"{destination_folder}/{file_name}")
def move_file(file_path, destination_folder):
	file_name = os.path.basename(file_path)
	destination_path = os.path.join(destination_folder, file_name)
	shutil.move(file_path, destination_path)
	print(f"Moved {file_path} to {destination_path}")
#----------------------------------------------------------------------
ic_all = ImageFileCollection(path_data, glob_include='calib*.fits', keywords=['object', 'filter',])
#----------------------------------------------------------------------
#	Header File
#----------------------------------------------------------------------
image_files = [f"{path_data}/{inim}" for inim in ic_all.summary['file']]
for ii, inim in enumerate(image_files):
	header_file = inim.replace('fits', 'head')
	imheadcom = f"imhead {inim} > {header_file}"
	subprocess.run(imheadcom, shell=True)
	print(f"[{ii:>4}/{len(image_files):>4}] {inim} --> {header_file}", end='\r')
print()
#----------------------------------------------------------------------
objarr = np.unique(ic_all.summary['object'][~ic_all.summary['object'].mask])
print(f"OBJECT Numbers: {len(objarr)}")

#	Transient Candidates
for oo, obj in enumerate(objarr):
	print("-"*60)
	print(f"[{oo:>4}/{len(objarr):>4}] OBJECT: {obj}")
	_filterarr = list(np.unique(ic_all.filter(object=obj).summary['filter']))
	print(f"FILTER: {_filterarr} ({len(_filterarr)})")
	#	Filter
	for filte in _filterarr:
		#	Path to Destination
		path_destination = f'{path_processed}/{obj}/{obs}/{filte}'
		#
		path_phot = f"{path_destination}/phot"
		path_transient = f"{path_destination}/transient"
		path_transient_cand_png = f"{path_transient}/png_image"
		path_transient_cand_fits = f"{path_transient}/fits_image"
		path_header = f"{path_destination}/header"

		#	Check save path
		paths = [path_destination, path_phot, path_transient, path_transient_cand_png, path_transient_cand_fits, path_header]
		for path in paths:
			if not os.path.exists(path):
				os.makedirs(path)
		#------------------------------------------------------------
		#	Files
		#------------------------------------------------------------
		#	Single Frames
		#------------------------------------------------------------
		#	calib_7DT01_T09373_20240423_032804_r_120.fits
		# single_frames = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.fits"))
		single_frames = sorted([f for f in glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.fits") if not re.search(r'\.com\.', f)])
		print(f"Single Frames: {len(single_frames)}")
		# single_phot_catalogs = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.phot.cat"))
		single_phot_catalogs = sorted([f for f in glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.phot.cat") if not re.search(r'\.com\.', f)])
		print(f"Phot Catalogs (single): {len(single_phot_catalogs)}")
		single_phot_pngs = sorted([f for f in glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*MAG_*.png") if not re.search(r'\.com\.', f)])
		print(f"Phot PNGs (single): {len(single_phot_pngs)}")
		#------------------------------------------------------------
		#	Stacked Frames
		#------------------------------------------------------------
		stack_frames = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.fits"))
		print(f"Stack Frames: {len(stack_frames)}")
		stack_phot_catalogs = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.phot.cat"))
		print(f"Phot Catalogs (stack): {len(stack_phot_catalogs)}")
		stack_phot_pngs = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.MAG_*.png"))
		print(f"Phot PNGs (stack): {len(stack_phot_pngs)}")
		#------------------------------------------------------------
		#	Subtracted Frames
		#------------------------------------------------------------
		#	ref_PS1_T09373_00000000_000000_r_0.conv.fits
		convolved_ref_frames = sorted(glob.glob(f"{path_data}/ref_*_{obj}_*_*_{filte}_*.conv.fits"))
		print(f"Convolved Ref Frames: {len(convolved_ref_frames)}")
		subt_frames = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.subt.fits"))
		print(f"Subt Frames: {len(subt_frames)}")
		subt_phot_catalogs = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.subt.phot.cat"))
		print(f"Phot Subt Catalogs: {len(subt_phot_catalogs)}")

		ssf_regions = sorted(glob.glob(f"{path_data}/*com.ssf.reg"))
		print(f"SSF Regions: {len(ssf_regions)}")
		ssf_catalogs = sorted(glob.glob(f"{path_data}/*com.ssf.txt"))
		print(f"SSF Catalogs: {len(ssf_catalogs)}")
		#------------------------------------------------------------
		#	Transient Candidates
		#------------------------------------------------------------
		#	calib_7DT01_T12936_20240423_034610_r_360.com.437.sci.fits
		sci_snapshots = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.*.sci.fits"))
		#	ref_PS1_T12936_00000000_000000_r_0.conv.437.ref.fits
		ref_snapshots = sorted(glob.glob(f"{path_data}/ref_*_{obj}_*_*_{filte}_*.conv.*.ref.fits"))
		#	calib_7DT01_T12936_20240423_034610_r_360.com.subt.437.sub.fits
		sub_snapshots = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.subt.*.sub.fits"))
		print(f"Snapshots (fits): {len(sci_snapshots)}, {len(ref_snapshots)}, {len(sub_snapshots)}")

		sci_png_snapshots = sorted(glob.glob(f"{path_data}/*.sci.png"))
		ref_png_snapshots = sorted(glob.glob(f"{path_data}/*.ref.png"))
		sub_png_snapshots = sorted(glob.glob(f"{path_data}/*.sub.png"))
		print(f"Snapshots (png): {len(sci_png_snapshots)}, {len(ref_png_snapshots)}, {len(sub_png_snapshots)}")

		#	calib_7DT01_T09373_20240423_033207_r_360.com.subt.flag.summary.csv
		flag_tables = sorted(glob.glob(f"{path_data}//calib_*_{obj}_*_*_{filte}_*.subt.flag.summary.csv"))
		print(f"Flag Summary Tables: {flag_tables}")
		#	calib_7DT01_T12931_20240423_031935_r_360.com.subt.transient.cat
		transient_catalogs = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.subt.transient.cat"))
		print(f"Transient Catalogs: {len(transient_catalogs)}")
		transient_regions = sorted(glob.glob(f"{path_data}/calib_*_{obj}_*_*_{filte}_*.com.subt.transient.reg"))
		print(f"Transient Regions: {len(transient_regions)}")
		#------------------------------------------------------------
		#	Header Files
		#------------------------------------------------------------
		header_files = sorted(glob.glob(f"{path_data}/*.head"))
		print(f"Header Files: {len(header_files)}")
		#	Grouping files
		files_to_base = single_frames + stack_frames
		files_to_phot = single_phot_catalogs + single_phot_pngs \
			+ stack_phot_catalogs + stack_phot_pngs
		files_to_transient = convolved_ref_frames + subt_frames + subt_phot_catalogs \
			+ ssf_regions + ssf_catalogs \
			+ transient_catalogs + transient_regions + flag_tables
		files_to_candidate_fits = sci_snapshots + ref_snapshots + sub_snapshots
		files_to_candidate_png = sci_png_snapshots + ref_png_snapshots + sub_png_snapshots		
		files_to_header = header_files

		destination_paths = [
			path_destination, 
			path_phot,
			path_transient, 
			path_transient_cand_fits, 
			path_transient_cand_png,
			path_header,
			]

		all_files = [
			files_to_base, 
			files_to_phot,
			files_to_transient, 
			files_to_candidate_fits, 
			files_to_candidate_png,
			files_to_header,
			]
		#	Move
		# with ThreadPoolExecutor(max_workers=ncore) as executor:
		# 	for files, destination in zip(all_files, destination_paths):
		# 		executor.map(lambda file_path: move_file(file_path, destination), files)
		# print("Done")

		with ThreadPoolExecutor(max_workers=ncore) as executor:
			for files, destination in zip(all_files, destination_paths):
				# move_file 함수를 partial을 이용해 목적지 경로를 고정합니다.
				move_func = partial(move_file, destination_folder=destination)
				executor.map(move_func, files)

delt_data_transfer = time.time() - t0_data_transfer
timetbl['status'][timetbl['process']=='data_transfer'] = True
timetbl['time'][timetbl['process']=='data_transfer'] = delt_data_transfer
#======================================================================
#	Report the LOG
#======================================================================
end_localtime = time.strftime('%Y-%m-%d_%H:%M:%S_(%Z)', time.localtime())
note = "-"
log_text = f"{path_new},{start_localtime},{end_localtime},{note}\n"

with open(path_log, 'a') as file:
	file.write(f"{log_text}")

objtbl.write(f"{path_data}/data_processing.log", format='csv')

#======================================================================
#	Slack message
#======================================================================
# delt_total = round(timetbl['time'][timetbl['process']=='total'].item()/60., 1)
delt_total = (time.time() - starttime)
timetbl['status'][timetbl['process']=='total'] = True
timetbl['time'][timetbl['process']=='total'] = delt_total
#   Time Table
timetbl['time'].format = '1.3f'
timetbl.write(f'{path_data}/obs.summary.log', format='csv', overwrite=True)

if slack_report:
	channel = '#pipeline'
	text = f'[`gpPy`+GPU/{project}-{obsmode}] Processing Complete {obs} {os.path.basename(path_new)} Data ({nobj} objects) with {ncore} cores taking {delt_total/60.:.1f} mins'

	param_slack = dict(
		token = OAuth_Token,
		channel = channel,
		text = text,
	)

	tool.slack_bot(**param_slack)