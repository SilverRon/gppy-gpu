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
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import repeat
import subprocess
import multiprocessing
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
	Path_root = path_thisfile.parent.parent.parent  # Careful! not a str
	# sys.path.append('../../src')  # Deprecated
	Path_src = Path_root / 'src'
	Path_run = path_thisfile.parent
except NameError:
	Path_src = find_gppy_gpu_src()
	Path_root = Path(Path_src).parent 
	Path_run = Path_root / 'run' / 'routine'

if Path_src not in map(Path, sys.path):
	sys.path.append(str(Path_src)) 
from preprocess import calib
from util import tool
from util.path_manager import log2tmp
#------------------------------------------------------------
#	plot setting
#------------------------------------------------------------
plt.ioff()
InteractiveShell.ast_node_interactivity = "last_expr"
#------------------------------------------------------------
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
# verbose_gpu = False
verbose_gpu = True
local_astref = False
debug = False

#	N cores for Multiprocessing
# try:
# 	ncore = int(sys.argv[2])
# except:
# 	ncore = 2
ncore = 4
print(f"- Number of Cores: {ncore}")

memory_threshold = 50
#------------------------------------------------------------
#	Ready
#------------------------------------------------------------
#	OBS
# revision
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
#%%
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
path_default_gphot = f'{path_config}/gphot.{obs.lower()}_{n_binning}x{n_binning}.config'
path_mframe = f'/lyman/data1/factory/master_frame_{n_binning}x{n_binning}_gain2750'
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
#%%
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

#	Skil Test Data
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
	rmcom = f'rm -rf {path_data}'
	print(rmcom)
	os.system(rmcom)
else:
	pass

if not os.path.exists(path_data):
	os.makedirs(path_data)
obsinfo = calib.getobsinfo(obs, obstbl)
#%%
# Added Tile Selection Feature

ic1 = ImageFileCollection(path_new, keywords='*')
if 'tile' not in upaths or upaths['tile'] == "":
	pass
else:
	try:
		tile = f"{int(upaths['tile']):05}"
		file_list = [str(f) for f in ic1.summary['file']]
		pattern = re.compile(fr'(?=.*{tile}|FLAT|DARK|BIAS)')
		filtered_files = [f for f in file_list if pattern.search(f)]
		filtered_ic = ImageFileCollection(path_new, filenames=filtered_files)
		ic1 = filtered_ic
	except Exception as e:
		print('Tile Selection Failed\n', e)
#%%
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

# %%
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
project = '7DT/7DS'
obsmode = 'COMISSION'
print(f"[{project}] {obsmode}")


# - Slack notification

# %%
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
# ### Master Frame
# - Bias

# %%
# st = time.time()
#   GPU
#============================================================
#	GPU Library
#------------------------------------------------------------
#	GPU image process library
from eclaire import FitsContainer, reduction, fixpix, imalign, imcombine
t00 = time.time()
#	Memory management
# import cupy
import cupy as cp

#	Pick GPU unit
odd_unit_obs = ['7DT01', '7DT03', '7DT05', '7DT07', '7DT09', '7DT11', '7DT13', '7DT15', '7DT17', '7DT19',]
even_unit_obs = ['7DT02', '7DT04', '7DT06', '7DT08', '7DT10', '7DT12', '7DT14', '7DT16', '7DT18', '7DT20']

print(f"="*60)
print()
if obs in odd_unit_obs:
	cp.cuda.Device(0).use()
	print(f"Use the first GPU Unit ({obs})")
elif obs in even_unit_obs:
	cp.cuda.Device(1).use()
	print(f"Use the second GPU Unit ({obs})")

mempool = cp.get_default_memory_pool()
if verbose_gpu:
	print(f"Default GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

print()
print(f"="*60)
#------------------------------------------------------------
#	BIAS
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Bias
#-----------------------------------------------------------
""")
t0_bias = time.time()



if biasnumb != 0:
	#   Stacking with GPU
	bfc = FitsContainer(bimlist)
	if verbose_gpu:
		print(f"Bias fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

	boutput = f"{path_data}/zero.fits"

	# t0 = time.time()
	mbias = imcombine(
		bfc.data, name=boutput, list=bimlist, overwrite=True,
		combine='median' # specify the co-adding method
		#width=3.0 # specify the clipping width
		#iters=5 # specify the number of iterations
	)
	# delt = time.time() - t0
	fits.writeto(f"{path_data}/zero.fits", data=cp.asnumpy(mbias), header=fits.getheader(bimlist[0]), overwrite=True)

	if verbose_gpu: print(f"bias combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

	dateobs_mzero = tool.calculate_average_date_obs(ic1.filter(imagetyp='bias').summary['date-obs'])

	date = dateobs_mzero[:10].replace('-', '')
	zeroim = f'{path_mframe}/{obs}/zero/{date}-zero.fits'
	if not os.path.exists(os.path.dirname(zeroim)):
		os.makedirs(os.path.dirname(zeroim))
	cpcom = f'cp {path_data}/zero.fits {zeroim}'
	print(cpcom)
	os.system(cpcom)
	plt.close('all')

	#	Clear the momory pool
	del bfc
	mempool.free_all_blocks()
	if verbose_gpu:
		print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")
	timetbl['status'][timetbl['process']=='master_frame_bias'] = True

else:
	#	IF THERE IS NO FLAT FRAMES, BORROW FROM CLOSEST OTHER DATE
	print('\nNO BIAS FRAMES\n')
	pastzero = np.array(glob.glob(f'{path_mframe}/{obs}/zero/*zero.fits'))

	#	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
	deltime = []

	#	Zero
	_zeromjd = Time(ic1.summary['date-obs'][0], format='isot').mjd
	for date in pastzero:
		pastzeromjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[0])
		deltime.append(np.abs(_zeromjd-pastzeromjd))
	indx_closest = np.where(deltime == np.min(deltime))
	tmpzero = pastzero[indx_closest][0]
	print(f'Borrow {tmpzero}')
	mbias = cp.asarray(fits.getdata(tmpzero), dtype='float32')

delt_bias = time.time() - t0_bias
print(f"Bias Master Frame: {delt_bias:.3f} sec")
timetbl['time'][timetbl['process']=='master_frame_bias'] = delt_bias
#------------------------------------------------------------
##	Dark
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Dark
#------------------------------------------------------------
""")
t0_dark = time.time()

darkdict = dict()
# if (darknumb > 0) & ((nobj > 0) | (flatnumb > 0)):
if (darknumb > 0):
	dark_process = True
	for i, exptime in enumerate(darkexptimelist):
		print(f'PRE PROCESS FOR DARK ({exptime} sec)\t[{i+1}/{len(darkexptimelist)}]')
		dimlist = list(ic1.filter(imagetyp='Dark', exptime=exptime).summary['file'])
		print(f"{len(dimlist)} Dark Frames Found")
		dfc = FitsContainer(dimlist)

		if verbose_gpu: print(f"Dark fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		doutput = f'{path_data}/dark-{int(exptime)}.fits'

		# t0 = time.time()
		mdark = imcombine(
			# dfc.data, list=dimlist, overwrite=True,
			dfc.data, name=doutput, list=dimlist, overwrite=True,
			combine='median' # specify the co-adding method
			#width=3.0 # specify the clipping width
			#iters=5 # specify the number of iterations
			)-mbias
		
		#	Apply BIAS Image on the Dark Image
		# 기존 FITS 파일 읽기
		with fits.open(doutput) as hdul:
			# 데이터 섹션 가져오기
			data = hdul[0].data
			
			# 데이터 수정하기 (예: 모든 픽셀 값을 2배로 만들기)
			new_data = cp.asnumpy(cp.array(data) - mbias)
			
			# 새로운 HDU 생성
			new_hdu = fits.PrimaryHDU(new_data)
			
			# 기존 헤더 정보 복사
			new_hdu.header = hdul[0].header

			# 새로운 HDU 리스트 생성
			new_hdul = fits.HDUList([new_hdu])
			
			# 수정된 데이터를 새로운 FITS 파일로 저장
			new_hdul.writeto(doutput, overwrite=True)

		# delt = time.time() - t0

		# t_np = time.time()
		# data - mbias_np
		# print(time.time()-t_np)
		# 0.19986248016357422

		# t_cp = time.time()
		# mdark_cp - mbias
		# print(time.time()-t_cp)
		# 0.0005612373352050781

		if verbose_gpu: print(f"dark combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		#	Clear the momory pool
		del dfc
		mempool.free_all_blocks()
		gc.collect()
		if verbose_gpu: print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")
		darkdict[str(int(exptime))] = mdark

		dateobs_mdark = tool.calculate_average_date_obs(ic1.filter(imagetyp='dark').summary['date-obs'])
		date = dateobs_mdark[:10].replace('-', '')

		darkim = f'{path_mframe}/{obs}/dark/{int(exptime)}-{date}-dark.fits'

		if not os.path.exists(os.path.dirname(darkim)):
			os.makedirs(os.path.dirname(darkim))

		cpcom = f'cp {path_data}/dark-{int(exptime)}.fits {darkim}'
		print(cpcom)
		os.system(cpcom)
		plt.close('all')
		del mdark
		mempool.free_all_blocks()
		gc.collect()

	timetbl['status'][timetbl['process']=='master_frame_dark'] = True
else:
	#	Borrow
	print('\nNO DARK FRAMES\n')
	if nobj > 0:
		objexptimelist = list(set(ic1.filter(imagetyp='Light').summary['exptime']))
		exptime = np.max(objexptimelist)
		pastdark = np.array(glob.glob(f'{path_mframe}/{obs}/dark/*-dark.fits'))

	elif flatnumb > 0:
		objexptimelist = list(set(ic1.filter(imagetyp='Flat').summary['exptime']))
		exptime = np.max(objexptimelist)
		pastdark = np.array(glob.glob(f'{path_mframe}/{obs}/dark/*-dark.fits'))


	#	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
	deltime = []
	delexptime = []
	darkexptimes = []
	_darkmjd = Time(ic1.summary['date-obs'][0], format='isot').mjd
	for date in pastdark:
		darkmjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[1])
		darkexptime = int( os.path.basename(date).split('-')[0] )
		darkexptimes.append(darkexptime)
		deltime.append(np.abs(_darkmjd-darkmjd))

	deldarkexptime_arr = np.abs(np.array(darkexptimes)-exptime)

	indx_closet = np.where(
		(deltime == np.min(deltime)) &
		# (darkexptimes == np.max(darkexptimes))
		(deldarkexptime_arr == np.min(deldarkexptime_arr))
	)
	if len(indx_closet[0]) == 0:
		indx_closet = np.where(
			(deltime == np.min(deltime))
		)
	else:
		pass

	tmpdark = pastdark[indx_closet][-1]
	# exptime = int(fits.getheader(tmpdark)['exptime'])
	exptime = int(np.array(darkexptimes)[indx_closet])

	mdark = cp.asarray(fits.getdata(tmpdark), dtype='float32')
	darkdict[f'{exptime}'] = mdark
	del mdark

delt_dark = time.time() - t0_dark
print(f"Dark Master Frame: {delt_dark:.3f} sec")
timetbl['time'][timetbl['process']=='master_frame_dark'] = delt_dark

darkexptimearr = np.array([float(val) for val in list(darkdict.keys())])
#------------------------------------------------------------
#	Flat
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Flat
#------------------------------------------------------------
""")

t0_flat = time.time()

#
flatdict = dict()
if flatnumb > 0:

	#	master flat dictionary
	for filte in filterlist:
		print(f'- {filte}-band')
		fimlist = []

		flat_raw_imlist = list(ic1.filter(imagetyp='FLAT', filter=filte).summary['file'])
		flat_raw_exptarr = cp.array(ic1.filter(imagetyp='FLAT', filter=filte).summary['exptime'].data.data)[:, None, None]
		_ffc = FitsContainer(flat_raw_imlist)

		_exptarr = np.array([int(expt) for expt in list(darkdict.keys())])

		closest_dark_exptime = np.min(_exptarr)
		exptime_scale_arr = flat_raw_exptarr / closest_dark_exptime

		#	Bias Correction
		_ffc.data -= mbias
		#	Dark Correction
		_ffc.data -= darkdict[str(int(closest_dark_exptime))] * exptime_scale_arr

		#	Normalization
		_ffc.data /= cp.median(_ffc.data, axis=(1, 2), keepdims=True)
	
		if verbose_gpu:
			print(f"Flat fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		#	Generate Master Flat
		foutput = f'{path_data}/n{filte}.fits'
		mflat = imcombine(
			_ffc.data, name=foutput, list=flat_raw_imlist, overwrite=True,
			combine='median' # specify the co-adding method
			#width=3.0 # specify the clipping width
			#iters=5 # specify the number of iterations
		)
		#--------------------------------------------------------

		dateobs_mflat = tool.calculate_average_date_obs(ic1.filter(imagetyp='FLAT', filter=filte).summary['date-obs'])
		date = dateobs_mflat[:10].replace('-', '')
		flatim = f'{path_mframe}/{obs}/flat/{date}-n{filte}.fits'


		#	Save to the database
		if not os.path.exists(os.path.dirname(flatim)):
			os.makedirs(os.path.dirname(flatim))

		cpcom = f'cp {path_data}/n{filte}.fits {flatim}'
		print(cpcom)
		os.system(cpcom)

		#	Save to the dictionary 
		flatdict[filte] = mflat

		if verbose_gpu: print(f"flat combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		#	Clear the momory pool
		del _ffc
		del mflat
		mempool.free_all_blocks()
		gc.collect()

		if verbose_gpu: print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

	timetbl['status'][timetbl['process']=='master_frame_flat'] = True

else:
	pass

delt_flat = time.time() - t0_flat
print(f"Flat Master Frame: {delt_flat:.3f} sec")
timetbl['time'][timetbl['process']=='master_frame_flat'] = delt_flat

#------------------------------------------------------------
#	Object correction
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Object correction
#------------------------------------------------------------
""")
if nobj == 0:

	end_localtime = time.strftime('%Y-%m-%d_%H:%M:%S_(%Z)', time.localtime())
	note = "No Light Frame"
	log_text = f"{path_new},{start_localtime},{end_localtime},{note}\n"

	with open(path_log, 'a') as file:
		file.write(f"{log_text}")
	print(f"[EXIT!] {note}")
	sys.exit()

print(f"{nobj} OBJECT: {list(objarr)}")
t0_data_reduction = time.time()

#	OBJECT FRAME filter list

from astropy.table import Table
objtbl = Table()
objtbl['image'] = list(ic1.filter(imagetyp='LIGHT').files)
objtbl['object'] = list(ic1.filter(imagetyp='LIGHT').summary['object'])
objtbl['filter'] = list(ic1.filter(imagetyp='LIGHT').summary['filter'])
objtbl['exptime'] = list(ic1.filter(imagetyp='LIGHT').summary['exptime'])
objtbl['data_reduction'] = False
objtbl['astrometry'] = False
objtbl['photometry'] = False
#
BATCH_SIZE = 10  # 한 번에 처리할 이미지 수, 필요에 따라 조정
#
for filte in np.unique(objtbl['filter']):
	for exptime in np.unique(objtbl['exptime'][objtbl['filter']==filte]):
		fnamelist = list(objtbl['image'][(objtbl['filter']==filte) & (objtbl['exptime']==exptime)])
		outfnamelist = [f"{path_data}/fdz{os.path.basename(fname)}" for fname in fnamelist]
		
		print(f"{len(fnamelist)} OBJECT Correction: {exptime}s in {filte}-band")

		# 여기서 fnamelist를 BATCH_SIZE 만큼씩 나눠서 처리합니다.
		for i in range(0, len(fnamelist), BATCH_SIZE):
			batch_fnames = fnamelist[i:i + BATCH_SIZE]
			batch_outfnames = outfnamelist[i:i + BATCH_SIZE]

			print(f"[{i:0>4}] BATCH")

			ofc = FitsContainer(batch_fnames)
			if verbose_gpu: print(f"Object fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

			#	Take Dark Frame
			indx_closest_dark = np.argmin(np.abs(darkexptimearr-exptime))
			closest_dark_exptime = darkexptimearr[indx_closest_dark]
			exptime_scale = exptime/closest_dark_exptime

			#	Take Flat Frame
			if filte in list(flatdict.keys()):
				pass
			else:
				print("No Master Flat. Let's Borrow")
				dateobs = fits.getheader(fnamelist[0])['DATE-OBS']
				_objmjd = Time(dateobs, format='isot').mjd
				pastflat = np.array(glob.glob(f'{path_mframe}/{obs}/flat/*n{filte}*.fits'))
				print(pastflat)

				deltime = []
				for date in pastflat:

					flatmjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[0])
					deltime.append(np.abs(_objmjd-flatmjd))

				print(deltime)
				indx_closet = np.where(deltime == np.min(deltime))
				tmpflat = pastflat[indx_closet].item()

				with fits.open(tmpflat, mode='readonly') as hdul:
					mflat = cp.asarray(hdul[0].data, dtype='float32')
				flatdict[filte] = mflat
				del mflat
				mempool.free_all_blocks()

			#	Reduction
			ofc.data = reduction(ofc.data, mbias, darkdict[str(int(closest_dark_exptime))], flatdict[filte])
			ofc.write(batch_outfnames, overwrite=True)
			del ofc
			mempool.free_all_blocks()
			# gc.collect()
			if verbose_gpu: print(f"Object correction GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

#	Check Memory pool
if verbose_gpu: print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")
#	Clear all momories pool
mempool.free_all_blocks()
#	Bias
del mbias
#	Dark Dictionary
for key in list(darkdict.keys()):
    del darkdict[key]
del darkdict
#	Dark Dictionary
for key in list(flatdict.keys()):
    del flatdict[key]
del flatdict
mempool.free_all_blocks()
if verbose_gpu: print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

#	Add Reduced LIGHT FRAME 
# objtbl['reduced_image'] = [f"fdz{inim}" if os.path.exists(f"{path_data}/fdz{inim}"s) else None for inim in objtbl['raw_image']]

#	Corrected image list
fdzimlist = sorted(glob.glob(f"{path_data}/fdz*.fits"))

#	Logging for the data reduction
for ii, inim in enumerate(objtbl['image']):
	fdzim = f"{path_data}/fdz{os.path.basename(inim)}"
	if fdzim in fdzimlist:
		objtbl['data_reduction'][ii] = True

ic_fdzobj = ImageFileCollection(path_data, keywords='*', filenames=fdzimlist)

delt_data_reduction = time.time() - t0_data_reduction

print(f"OBJECT Correction: {delt_data_reduction:.3f} sec / {len(ic1.filter(imagetyp='LIGHT').summary)} frames")

timetbl['status'][timetbl['process']=='data_reduction'] = True
timetbl['time'][timetbl['process']=='data_reduction'] = delt_data_reduction
#
cp.get_default_memory_pool().free_all_blocks()
cp.cuda.set_allocator(None)