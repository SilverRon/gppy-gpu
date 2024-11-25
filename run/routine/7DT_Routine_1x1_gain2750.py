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
verbose_gpu = False
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
	mempool.free_all_blocks()
	del bfc
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
		mempool.free_all_blocks()
		del dfc
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
		mempool.free_all_blocks()
		del _ffc
		del mflat

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

				deltime = []
				for date in pastflat:

					flatmjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[0])
					deltime.append(np.abs(_objmjd-flatmjd))

				indx_closet = np.where(deltime == np.min(deltime))
				tmpflat = pastflat[indx_closet].item()

				with fits.open(tmpflat, mode='readonly') as hdul:
					mflat = cp.asarray(hdul[0].data, dtype='float32')
				flatdict[filte] = mflat

			#	Reduction
			ofc.data = reduction(ofc.data, mbias, darkdict[str(int(closest_dark_exptime))], flatdict[filte])
			ofc.write(batch_outfnames, overwrite=True)
			del ofc
			if verbose_gpu: print(f"Object correction GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

#	Check Memory pool
if verbose_gpu: print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")
#	Clear all momories pool
mempool.free_all_blocks()
# del ofc
del mbias
# del mdark
# del mflat
del darkdict
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
#
#------------------------------------------------------------
#	ASTROMETRY
#------------------------------------------------------------
# st_ = time.time()
# deltlist = []
# for nn, (_fname, obj, ra, dec) in enumerate(ic_fdzobj.summary['file', 'object', 'ra', 'dec']):
# 	fname = f"{path_data}/{_fname}"
# 	calib.astrometry(fname, obsinfo['pixscale'], ra, dec, obsinfo['fov']/60., 15, None)
# 	_delt = time.time() - st_
# 	deltlist.append(_delt)
# 	print(f"[{nn+1}/{len(fdzimlist)}] {_fname} {_delt:.3f} sec",)

# delt = time.time() - st_
# print(f"Astrometry was done: {delt:.3f} sec/{len(fdzimlist)} ({np.median(deltlist):.3f} sec/image)")
#------------------------------------------------------------
t0_astrometry_solve_field = time.time()
#------------------------------------------------------------
from concurrent.futures import ProcessPoolExecutor

fnamelist = [f"{path_data}/{_fname}" for _fname in ic_fdzobj.summary['file']]
pixscale = obsinfo['pixscale']*n_binning
objectlist = [pixscale]*len(fnamelist)
ralist = list(ic_fdzobj.summary['ra'])
declist = list(ic_fdzobj.summary['dec'])
fovlist = [obsinfo['fov']/60]*len(fnamelist)
cpulimitlist = [10]*len(fnamelist)
cfglist = [f"{path_data}/{_fname.replace('fits', 'ast.sex')}" for _fname in ic_fdzobj.summary['file']]
_ = [None]*len(fnamelist)

#	Make Source EXtractor Configuration
# presexcom = f"source-extractor -c {conf_simple} {inim} -FILTER_NAME {conv_simple} -STARNNW_NAME {nnw_simple} -PARAMETERS_NAME {param_simple} -CATALOG_NAME {precat}"
precatlist = [f"{path_data}/{_fname.replace('fits', 'pre.cat')}" for _fname in ic_fdzobj.summary['file']]

conf_simple = f"{path_config}/simple.sex"
param_simple = f"{path_config}/simple.param"
nnw_simple = f"{path_config}/simple.nnw"
conv_simple = f"{path_config}/simple.conv"

def modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple, pixscale):

	#
	import re
	#	CATALOG_NAME
	pattern_cat_to_find = 'CATALOG_NAME     test.cat       # name of the output catalog'
	pattern_cat_to_replace = f'CATALOG_NAME     {_precat}       # name of the output catalog'
	#	PARAMETERS_NAME
	pattern_param_to_find = 'PARAMETERS_NAME  simple.param  # name of the file containing catalog contents'
	pattern_param_to_replace = f'PARAMETERS_NAME  {param_simple}  # name of the file containing catalog contents'
	#	FILTER_NAME
	pattern_conv_to_find = 'FILTER_NAME      simple.conv   # name of the file containing the filter'
	pattern_conv_to_replace = f'FILTER_NAME      {conv_simple}   # name of the file containing the filter'
	#	STARNNW_NAME
	pattern_nnw_to_find = 'STARNNW_NAME     simple.nnw    # Neural-Network_Weight table filename'
	pattern_nnw_to_replace = f'STARNNW_NAME     {nnw_simple}    # Neural-Network_Weight table filename'
	#	Pixel Scale
	pattern_pixscale_to_find = 'PIXEL_SCALE      0.51         # size of pixel in arcsec (0=use FITS WCS info)'
	pattern_pixscale_to_replace = f'PIXEL_SCALE      {pixscale}         # size of pixel in arcsec (0=use FITS WCS info)'


	pattern_to_find_list = [
		pattern_cat_to_find,
		# pattern_param_to_find,
		pattern_conv_to_find,
		pattern_nnw_to_find,
		pattern_pixscale_to_find,
	]

	pattern_to_replace_list = [
		pattern_cat_to_replace,
		# pattern_param_to_replace,
		pattern_conv_to_replace,
		pattern_nnw_to_replace,
		pattern_pixscale_to_replace,
	]

	# 파일을 읽고 각 행을 확인하면서 패턴에 맞는 텍스트를 수정
	with open(conf_simple, 'r') as file:
		text = file.read()

	for pattern_to_find, pattern_to_replace in zip(pattern_to_find_list, pattern_to_replace_list):
		text = re.sub(pattern_to_find, pattern_to_replace, text)

	with open(_outcfg, 'w') as file:
		file.write(text)

#	
for _precat, _outcfg in zip(precatlist, cfglist):
	modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple, pixscale)

st_ = time.time()
#	Move to the path_data
original_directory = os.getcwd()
os.chdir(path_data)
#	Copy default.sex (High DETECT_THRESH)
cpcom_default_cfg = f"cp {path_config}/default.sex {path_data}"
print(cpcom_default_cfg)
os.system(cpcom_default_cfg)

#	Astrometry
while psutil.virtual_memory().percent > memory_threshold:
	print(f"Memory Usage is above {memory_threshold}% ({psutil.virtual_memory().percent}%) - Start the Astrometry!!!")
	time.sleep(10)

print(f"Memory Usage is below {memory_threshold}% - Start the Astrometry!!!")
with ProcessPoolExecutor(max_workers=ncore) as executor:
	# results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, cfglist, _))
	results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, _, _))


delt = time.time() - st_
#	Move back to the original path
os.chdir(original_directory)
print(f"Astrometry was done: {delt:.3f} sec/{len(fdzimlist)}")


#------------------------------------------------------------
astrometry_suffix_list = ['axy', 'corr', 'xyls', 'match', 'rdls', 'solved', 'wcs']
for suffix in astrometry_suffix_list:
	rmcom = f"rm {path_data}/*.{suffix}"
	print(rmcom)

delt_astrometry_solve_field = time.time() - t0_astrometry_solve_field
timetbl['status'][timetbl['process']=='astrometry_solve_field'] = True
timetbl['time'][timetbl['process']=='astrometry_solve_field'] = delt_astrometry_solve_field
# timetbl['status'][timetbl['process']=='astrometry'] = True
# timetbl['time'][timetbl['process']=='astrometry'] = int(time.time() - st_)

#	Add solved-image
# objtbl['astrometry_image'] = [f"afdz{inim}" if os.path.exists(f"{path_data}/afdz{inim}") else None for inim in objtbl['raw_image']]

afdzimlist = sorted(glob.glob(f"{path_data}/afdz*.fits"))

#	Logging for the data reduction
for ii, inim in enumerate(objtbl['image']):
	fdzim = f"{path_data}/afdz{os.path.basename(inim)}"
	if fdzim in afdzimlist:
		objtbl['astrometry'][ii] = True

#	Rename
# for inim in afdzimlist: calib.fnamechange(inim, obs)
# calimlist = sorted(glob.glob(f"{path_data}/calib*.fits"))

#------------------------------------------------------------
#	Astrometry Correction
#------------------------------------------------------------
def run_pre_sextractor(inim, outcat, param_simple, conv_simple, nnw_simple, pixscale):
	# outhead = inim.replace('fits', 'head')
	# outcat = inim.replace('fits', 'pre.cat')

	#	Pre-Source EXtractor
	sexcom = f"source-extractor -c {conf_simple} {inim} -CATALOG_NAME {outcat} -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME {param_simple} -FILTER_NAME {conv_simple} -STARNNW_NAME {nnw_simple} -PIXEL_SCALE {pixscale}"
	print(sexcom)
	# os.system(sexcom)
	if verbose_sex:
		os.system(cpcom_default_cfg)
	else:
		# Redirect SE output to a tmp log
		os.system(log2tmp(sexcom, "presex"))

outcatlist = []
outheadlist = []

# for inim in calimlist:
for inim in afdzimlist:
	outcat = inim.replace('fits', 'cat')
	outhead = inim.replace('fits', 'head')

	outcatlist.append(outcat)
	outheadlist.append(outhead)

t0_pre_source_extractor = time.time()

#	Pre-Source EXtractor
st_ = time.time()
with ProcessPoolExecutor(max_workers=ncore) as executor:
	# results = list(executor.map(run_pre_sextractor, calimlist, outcatlist, [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist)))
	results = list(executor.map(run_pre_sextractor, afdzimlist, outcatlist, [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist), [pixscale]*len(outcatlist)))
delt = time.time() - st_
# print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(calimlist)} (ncroe={ncore})")
print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(afdzimlist)} (ncroe={ncore})")

delt_pre_source_extractor = time.time() - t0_pre_source_extractor
timetbl['status'][timetbl['process']=='pre_source_extractor'] = True
timetbl['time'][timetbl['process']=='pre_source_extractor'] = delt_pre_source_extractor

#
t0_astrometry_scamp = time.time()

#	Catalog list for Scamp
path_cat_scamp_list = f"{path_data}/cat.scamp.list"
print(f"Generate Catalog List for SCAMP: {path_cat_scamp_list}")
s = open(path_cat_scamp_list, 'w')
for incat in outcatlist:
	s.write(f"{incat}\n")
s.close()

#	Head list for MissFits
path_head_missfits_list = f"{path_data}/head.missfits.list"
print(f"Generate Head List for MissFits: {path_head_missfits_list}")
m = open(path_head_missfits_list, 'w')
for inhead in outheadlist:
	m.write(f"{inhead}\n")
m.close()

#	Image list for MissFits
path_image_missfits_list = f"{path_data}/image.missfits.list"
print(f"Generate Image List for MissFits: {path_image_missfits_list}")
i = open(path_image_missfits_list, 'w')
# for inim in calimlist:
for inim in fdzimlist:
	i.write(f"{inim}\n")
i.close()

#	SCAMP
# scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list}"
# scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} -AHEADER_GLOBAL {path_config}/{obs.lower()}.ahead"
# print(scampcom)
# os.system(scampcom)

#	SCAMP (input CATALOG)
print(f"= = = = = = = = = = = = Astrometric Correction = = = = = = = = = = = =")
for oo, obj in enumerate(objarr):
	print(f"[{oo+1}/{len(objarr)}] {obj}")

	path_cat_scamp_list = f"{path_data}/{obj}.cat.scamp.list"
	s = open(path_cat_scamp_list, 'w')
	obj_outcatlist = [incat for incat in outcatlist if obj in os.path.basename(incat)]
	for incat in obj_outcatlist:
		s.write(f"{incat}\n")
	s.close()

	if local_astref and (re.match(tile_name_pattern, obj)) and (obj not in ['T04231', 'T04409', 'T04590']):
		astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
		if debug:
			print('='*79)
			print('astrefcat', astrefcat)
			print('='*79)
		scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
	else:
		scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"

	#	Run
	# scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} {scamp_addcom}"
	# print(scampcom)
	# os.system(scampcom)

	# Run with subprocess
	scampcom = ["scamp", "-c", f"{path_config}/7dt.scamp", f"@{path_cat_scamp_list}"]
	# if debug:
	# 	scampcom = ["scamp", "-c", f"{path_config}/7dt.scamp_vanilla", f"@{path_cat_scamp_list}"]
	scampcom += scamp_addcom.split()
	print(" ".join(scampcom))  # Join the command list for printing

	try:
		result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		print(result.stdout.decode())  # 명령어 실행 결과 출력
	except subprocess.CalledProcessError as e:
		print(f"Command failed with error code {e.returncode}")
		print(f"stderr output: {e.stderr.decode()}")


#	Rename afdz*.head --> fdz*.head
for inhead in outheadlist: os.rename(inhead, inhead.replace('afdz', 'fdz'))

delt_astrometry_scamp = time.time() - t0_astrometry_scamp
timetbl['status'][timetbl['process']=='astrometry_scamp'] = True
timetbl['time'][timetbl['process']=='astrometry_scamp'] = delt_astrometry_scamp

#	MissFits
t0_missfits = time.time()

##	Single-Thread MissFits Compile
# missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list}"
missfitscom = f"missfits @{path_image_missfits_list}"
##	Multi-Threads MissFits Compile
# missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list} -NTHREADS {ncore}"
print(missfitscom)
os.system(missfitscom)

delt_missfits = time.time() - t0_missfits
timetbl['status'][timetbl['process']=='missfits'] = True
timetbl['time'][timetbl['process']=='missfits'] = delt_missfits

#	Rename fdz*.fits (scamp astrometry) --> calib*.fits
for inim in fdzimlist: calib.fnamechange(inim, obs)
calimlist = sorted(glob.glob(f"{path_data}/calib*.fits"))

for inim , _inhead in zip(calimlist, outheadlist):
	inhead = _inhead.replace('afdz', 'fdz')
	outhead = inim.replace('fits', 'head').replace('head', 'head.bkg')
	os.rename(inhead, outhead)
	
#------------------------------------------------------------
#	Remove SIP Keywords
#------------------------------------------------------------
# 이미지 파일 이름
# image_file = 'calib_7DT01_NGC0253_20231010_062322_r_60.fits'

# SIP 관련 키워드 리스트
# sip_and_wcs_keywords = [
# 	'A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER',
# 	'A_0_2', 'A_1_1', 'A_2_0', 'B_0_2', 'B_1_1', 'B_2_0',
# 	'AP_0_0', 'AP_0_1', 'AP_0_2', 'AP_1_0', 'AP_1_1', 'AP_2_0',
# 	'BP_0_0', 'BP_0_1', 'BP_0_2', 'BP_1_0', 'BP_1_1', 'BP_2_0',
# 	'CTYPE1', 'CTYPE2',
# 	'PV1_0', 'PV1_1', 'PV1_2', 'PV1_4', 'PV1_5', 'PV1_6', 'PV1_7', 'PV1_8', 'PV1_9', 'PV1_10',
# 	'PV2_0', 'PV2_1', 'PV2_2', 'PV2_4', 'PV2_5', 'PV2_6', 'PV2_7', 'PV2_8', 'PV2_9', 'PV2_10'
# ]

# for inim in calimlist:
# 	# FITS 파일 열기
# 	with fits.open(inim, mode='update') as hdul:
# 		header = hdul[0].header

# 		# SIP 관련 키워드 제거
# 		for key in sip_and_wcs_keywords:
# 			if key in header:
# 				del header[key]

# 		# 변경 사항 저장
# 		hdul.flush()



#	Update Coordinate on the Image
#------------------------------------------------------------
##	TAN --> TPV Projection
##	Center RA & Dec
##	RA, Dec Polygons
##	Rotation angle
#------------------------------------------------------------
print(f"Update Center & Polygon Info ...")
t0_get_polygon_info = time.time()

#	Correct CTYPE (TAN --> TPV)
for inim in calimlist:
	with fits.open(inim, mode='update') as hdul:
		# 헤더 데이터 불러오기
		hdr = hdul[0].header

		# 헤더 정보 변경 또는 추가
		hdr['CTYPE1'] = ('RA---TPV', 'WCS projection type for this axis')
		hdr['CTYPE2'] = ('DEC--TPV', 'WCS projection type for this axis')
		# 변경된 내용 저장
		hdul.flush()

# t0_wcs = time.time()
for cc, calim in enumerate(calimlist):
	# Extract WCS information (center, CD matrix)
	center, vertices, cd_matrixs = tool.get_wcs_coordinates(calim)
	cd1_1, cd1_2, cd2_1, cd2_2 = cd_matrixs

	# updates = [
	# 	("CTYPE1", 'RA---TPV', 'WCS projection type for this axis'),
	# 	("CTYPE2", 'DEC--TPV', 'WCS projection type for this axis')
	# ]
	# Define header list to udpate
	updates = [
		("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"),
		("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]")
	]

	# updates.append(("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"))
	# updates.append(("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]"))

	# RA, Dec Polygons
	for ii, (_ra, _dec) in enumerate(vertices):
		updates.append((f"RAPOLY{ii}", round(_ra, 3), f"RA POLYGON {ii} [deg]"))
		updates.append((f"DEPOLY{ii}", round(_dec, 3), f"DEC POLYGON {ii} [deg]"))

	# Field Rotation
	try:
		if (cd1_1 != 0) and (cd1_2 != 0) and (cd2_1 != 0) and (cd2_2 != 0):
			rotation_angle_1, rotation_angle_2 = tool.calculate_field_rotation(cd1_1, cd1_2, cd2_1, cd2_2)
		else:
			rotation_angle_1, rotation_angle_2 = float('nan'), float('nan')
	except Exception as e:
		print(f'Error: {e}')
		print(f'Image: {calim}')
		rotation_angle_1, rotation_angle_2 = float('nan'), float('nan')

	# Update rotation angle
	updates.append(('ROTANG1', rotation_angle_1, 'Rotation angle from North [deg]'))
	updates.append(('ROTANG2', rotation_angle_2, 'Rotation angle from East [deg]'))

	# FITS header update
	with fits.open(calim, mode='update') as hdul:
		for key, value, comment in updates:
			hdul[0].header[key] = (value, comment)
		hdul.flush()  # 변경 사항을 디스크에 저장

delt_get_polygon_info = time.time() - t0_get_polygon_info
timetbl['status'][timetbl['process']=='get_polygon_info'] = True
timetbl['time'][timetbl['process']=='get_polygon_info'] = delt_get_polygon_info

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
from astropy.wcs import WCS
import numpy as np
from astropy.io import fits
from astropy.time import Time

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
# ## Photometry for combined images
# %%
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
def create_mask_images(input_image, mask_suffix="mask.fits"):
	data = fits.getdata(input_image)
	mask = np.zeros_like(data, dtype=int)
	mask[data == 0] = 1
	mask[data != 0] = 0
	mask_filename = input_image.replace("fits", mask_suffix)
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
	_reference_images = _reference_images_7dt = _reference_images_ps1

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
		#	Transient Search Command
		search_com = f"python {path_find} {inim} {refim} {hcim} {hdim}"
		print(search_com)
		os.system(search_com)

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

from concurrent.futures import ThreadPoolExecutor
import shutil
from functools import partial

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

channel = '#pipeline'
text = f'[`gpPy`+GPU/{project}-{obsmode}] Processing Complete {obs} {os.path.basename(path_new)} Data ({nobj} objects) with {ncore} cores taking {delt_total/60.:.1f} mins'

param_slack = dict(
	token = OAuth_Token,
	channel = channel,
	text = text,
)

tool.slack_bot(**param_slack)
# %%
