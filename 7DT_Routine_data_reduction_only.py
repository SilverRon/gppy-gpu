#	7DT_Routine.py
# - Automatically Process the image data from 7DT facilities and search for transients
# - This is an advanced version of `gpPy`
# - Author: Gregory S.H. Paek (23.10.10)
#------------------------------------------------------------
#	Library
#------------------------------------------------------------
from __future__ import print_function, division, absolute_import
import os, sys, glob, subprocess
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
plt.ioff()
# from astropy.nddata import CCDData
from preprocess import calib
from util import tool
#	Astropy
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.coordinates import SkyCoord
from astropy.time import Time

from ccdproc import ImageFileCollection
from datetime import datetime, timezone, timedelta
import warnings
warnings.filterwarnings(action='ignore')
from itertools import repeat
import multiprocessing
import time
#------------------------------------------------------------
os.environ['TZ'] = 'Asia/Seoul'
time.tzset()
start_localtime = time.strftime('%Y-%m-%d_%H:%M:%S_(%Z)', time.localtime())
#------------------------------------------------------------
#	plot setting
#------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"
#------------------------------------------------------------
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#------------------------------------------------------------
#	Ready
#------------------------------------------------------------
#	OBS
try:
	obs = (sys.argv[1]).upper()
except:
	obs = input(f"7DT## (e.g. 7DT01):").upper()
# obs = '7DT01'
print(f'# Observatory : {obs.upper()}')
#	N cores for Multiprocessing
# try:
# 	ncore = int(sys.argv[2])
# except:
# 	ncore = 2
ncore = 4
print(f"- Number of Cores: {ncore}")
#------------------------------------------------------------
verbose_gpu = False
#------------------------------------------------------------
#	Path
#------------------------------------------------------------
path_base = '/large_data/factory'
path_ref = f'{path_base}/ref_frame/{obs.upper()}'
path_factory = f'{path_base}/{obs.lower()}'
# path_save = f'/data6/bkgdata/{obs.upper()}'
path_log = f'{path_base}/log/{obs.lower()}.log'
if not os.path.exists(path_log):
	print(f"Create {path_log}")
	f = open(path_log, 'w')
	f.write('date\n19941026')
	f.close()
path_keys = f'./config'
#------------------------------------------------------------
path_gal = f'{path_base}/../processed'
path_refcat = f'{path_base}/ref_cat'
# path_refcat = '/data4/gecko/factory/ref_frames/LOAO'
#------------------------------------------------------------
# path_config = '/home/paek/config'
path_config = './config'
path_default_gphot = f'{path_config}/gphot.{obs.lower()}.config'
path_mframe = f'{path_base}/master_frame'
# path_calib = f'{path_base}/calib'
#------------------------------------------------------------
#	Codes
#------------------------------------------------------------
# path_phot_sg = './phot/gregoryphot_2021.py'
path_phot_mp = './phot/gregoryphot_7DT.py'
path_phot_sub = './phot/gregoryphot_sub_2021.py'
path_find = './phot/gregoryfind_bulk_mp_2021.py'
#------------------------------------------------------------
path_obsdata = f'{path_base}/../obsdata'
path_raw = f'{path_obsdata}/{obs.upper()}'
rawlist = sorted(glob.glob(f'{path_raw}/2*'))
#------------------------------------------------------------
path_obs = f'{path_config}/obs.dat'
path_changehdr = f'{path_config}/changehdr.dat'
path_alltarget = f'{path_config}/alltarget.dat'
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
#------------------------------------------------------------
#	Table
#------------------------------------------------------------
logtbl = ascii.read(path_log)
datalist = np.copy(logtbl['date'])
obstbl = ascii.read(path_obs)
hdrtbl = ascii.read(path_changehdr)
alltbl = ascii.read(path_alltarget)
keytbl = ascii.read(f'{path_keys}/keys.dat')
#------------------------------------------------------------
#	Time Log Table
#------------------------------------------------------------
# %%
timetbl = Table()
timetbl['process'] = [
	'master_frame_bias',
	'master_frame_dark',
	'master_frame_flat',
	'data_reduction',
	'astrometry',
	'cr_removal',
	'photometry',
	'image_stack',
	'photometry_com',
	'subtraction',
	'photometry_sub',
	'transient_search',
	'total',
]
timetbl['status'] = False
timetbl['time'] = 0.0 * u.second
#============================================================
#------------------------------------------------------------
#	Main Body
#------------------------------------------------------------
#============================================================

# %%
newlist = [i for i in rawlist if (i not in datalist) & (i+'/' not in datalist)]
if len(newlist) == 0:
	print('No new data')
	sys.exit()
else:
	for ff, folder in enumerate(newlist):
		print(f"[{ff:0>2}] {folder}")
"""
# path = newlist[-1]
# path = newlist[3]
# path_raw = newlist[2]
# path_raw = newlist[0]

"""
try:
	path_new = sys.argv[2]
except:
	user_input = input("Path or Index To Process:")
	# 입력값이 숫자인 경우
	if user_input.isdigit():
		index = int(user_input)
		path_new = newlist[index]
	# 입력값이 경로 문자열인 경우 (여기서는 간단하게 '/'를 포함하는지만 확인)
	elif '/' in user_input:
		path_new = user_input
	# 그 외의 경우
	else:
		print("Wrong path or index")
		sys.exit()

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
os.makedirs(path_data)

obsinfo = calib.getobsinfo(obs, obstbl)

ic1 = ImageFileCollection(path_new, keywords='*')

#	Count the number of Light Frame
try:
	nobj = len(ic1.filter(imagetyp='LIGHT').summary)
	#	LIGHT FRAME TABLE
	# objtbl = Table()
	# objtbl['raw_image'] = [os.path.basename(inim) for inim in ic1.filter(imagetyp='LIGHT').files]
except:
	nobj = 0

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
text = f'[`gpPy`+GPU/{project}-TEST] Start Processing {obs} {os.path.basename(path_new)} Data ({nobj} objects) with {ncore} cores'

param_slack = dict(
	token = OAuth_Token,
	channel = channel,
	text = text,
)
# tool.slack_bot(**param_slack)
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
mempool = cp.get_default_memory_pool()
if verbose_gpu:
	print(f"Default GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")
#------------------------------------------------------------
#	BIAS
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Bias
#-----------------------------------------------------------
""")
t0_bias = time.time()

try:
	bimlist = list(ic1.filter(imagetyp='Bias').summary['file'])
	biasnumb = len(bimlist)
	print(f"{biasnumb} Bias Frames Found")
except:
	biasnumb = 0


if biasnumb != 0:
	#   Stacking with GPU
	bfc = FitsContainer(bimlist)
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
try:
    darkexptimelist = sorted(list(set(ic1.filter(imagetyp='dark').summary['exptime'])))
    darknumb = len(darkexptimelist)
except:
    darknumb = 0
darkdict = dict()
if darknumb != 0:
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
	objexptimelist = list(set(ic1.filter(imagetyp='Light').summary['exptime']))
	exptime = np.max(objexptimelist)
	pastdark = np.array(glob.glob(f'{path_mframe}/{obs}/dark/{int(exptime)}*dark.fits'))

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

	indx_closet = np.where(
		(deltime == np.min(deltime)) &
		(darkexptimes == np.max(darkexptimes))
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
#	Check Flat Frames 
try:
	filterlist = list(np.unique(ic1.filter(imagetyp='FLAT').summary['filter']))
	print(f"{len(filterlist)} filters found")
	print(f"Filters: {filterlist}")
	flatnumb = len(filterlist)
except:
	print(f"There is no flat frames")
	flatnumb = 0
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

			print(f"[{i}] BATCH")

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
# objtbl['reduced_image'] = [f"fdz{inim}" if os.path.exists(f"{path_data}/fdz{inim}") else None for inim in objtbl['raw_image']]

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
from concurrent.futures import ProcessPoolExecutor

fnamelist = [f"{path_data}/{_fname}" for _fname in ic_fdzobj.summary['file']]
objectlist = [obsinfo['pixscale']]*len(fnamelist)
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

def modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple):

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

	pattern_to_find_list = [
		pattern_cat_to_find,
		# pattern_param_to_find,
		pattern_conv_to_find,
		pattern_nnw_to_find,
	]

	pattern_to_replace_list = [
		pattern_cat_to_replace,
		# pattern_param_to_replace,
		pattern_conv_to_replace,
		pattern_nnw_to_replace
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
	modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple)

st_ = time.time()
with ProcessPoolExecutor(max_workers=ncore) as executor:
	# results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, cfglist, _))
	results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, _, _))
delt = time.time() - st_

print(f"Astrometry was done: {delt:.3f} sec/{len(fdzimlist)}")
#------------------------------------------------------------


astrometry_suffix_list = ['axy', 'corr', 'xyls', 'match', 'rdls', 'solved', 'wcs']
for suffix in astrometry_suffix_list:
	rmcom = f"rm {path_data}/*.{suffix}"
	print(rmcom)

timetbl['status'][timetbl['process']=='astrometry'] = True
timetbl['time'][timetbl['process']=='astrometry'] = int(time.time() - st_)

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
def run_pre_sextractor(inim, outcat, param_simple, conv_simple, nnw_simple):
	# outhead = inim.replace('fits', 'head')
	# outcat = inim.replace('fits', 'pre.cat')

	#	Pre-Source EXtractor
	sexcom = f"source-extractor -c {conf_simple} {inim} -CATALOG_NAME {outcat} -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME {param_simple} -FILTER_NAME {conv_simple} -STARNNW_NAME {nnw_simple}"
	print(sexcom)
	os.system(sexcom)

outcatlist = []
outheadlist = []

# for inim in calimlist:
for inim in afdzimlist:
	outcat = inim.replace('fits', 'cat')
	outhead = inim.replace('fits', 'head')

	outcatlist.append(outcat)
	outheadlist.append(outhead)

#	Pre-Source EXtractor
st_ = time.time()
with ProcessPoolExecutor(max_workers=ncore) as executor:
	# results = list(executor.map(run_pre_sextractor, calimlist, outcatlist, [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist)))
	results = list(executor.map(run_pre_sextractor, afdzimlist, outcatlist, [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist)))
delt = time.time() - st_
# print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(calimlist)} (ncroe={ncore})")
print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(afdzimlist)} (ncroe={ncore})")
#

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
scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list}"
# scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} -AHEADER_GLOBAL {path_config}/{obs.lower()}.ahead"
print(scampcom)
os.system(scampcom)

#	Rename afdz*.head --> fdz*.head
for inhead in outheadlist: os.rename(inhead, inhead.replace('afdz', 'fdz'))

#	MissFits
##	Single-Thread MissFits Compile
# missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list}"
missfitscom = f"missfits @{path_image_missfits_list}"
##	Multi-Threads MissFits Compile
# missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list} -NTHREADS {ncore}"
print(missfitscom)
os.system(missfitscom)

#	Rename fdz*.fits (scamp astrometry) --> calib*.fits
for inim in fdzimlist: calib.fnamechange(inim, obs)
calimlist = sorted(glob.glob(f"{path_data}/calib*.fits"))

for inim , _inhead in zip(calimlist, outheadlist):
	inhead = _inhead.replace('afdz', 'fdz')
	outhead = inim.replace('fits', 'head')
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

#	Correct CTYPE (TAN --> TPV)
# FITS 파일 열기
for inim in calimlist:
	with fits.open(inim, mode='update') as hdul:
		# 헤더 데이터 불러오기
		hdr = hdul[0].header

		# 헤더 정보 변경 또는 추가
		hdr['CTYPE1'] = ('RA---TPV', 'WCS projection type for this axis')
		hdr['CTYPE2'] = ('DEC--TPV', 'WCS projection type for this axis')
		# 변경된 내용 저장
		hdul.flush()



#	Update Coordinate on the Image
##	Center RA & Dec
##	RA, Dec Polygons
print(f"Update Center & Polygon Info ...")
t0_wcs = time.time()
for cc, calim in enumerate(calimlist):
	center, vertices = tool.get_wcs_coordinates(calim)

	fits.setval(calim, "RACENT", value=round(center[0].item(), 3), comment="RA CENTER [deg]")	
	fits.setval(calim, "DECCENT", value=round(center[1].item(), 3), comment="DEC CENTER [deg]")	
	
	for ii, (_ra, _dec) in enumerate(vertices):
		fits.setval(calim, f"RAPOLY{ii}", value=round(_ra, 3), comment=f"RA POLYGON {ii} [deg]")	
		fits.setval(calim, f"DEPOLY{ii}", value=round(_dec, 3), comment=f"DEC POLYGON {ii} [deg]")
		
delt_wcs = time.time() - t0_wcs
print(f'Done ({delt_wcs:.1f}s)')

#------------------------------------------------------------
#	Rotation Angle
#------------------------------------------------------------
def calculate_field_rotation(cd1_1, cd1_2, cd2_1, cd2_2):
    """
    Calculate the field rotation angle based on the given CD matrix elements.
    The field rotation angles indicate how much the image is rotated with respect
    to the North and East directions in the celestial coordinate system.

    Parameters:
    - cd1_1: CD1_1 value from the FITS header
    - cd1_2: CD1_2 value from the FITS header
    - cd2_1: CD2_1 value from the FITS header
    - cd2_2: CD2_2 value from the FITS header

    Returns:
    - rotation_angle_1: The rotation angle of the image's x-axis (typically Right Ascension)
      from the North in degrees. A positive value indicates a clockwise rotation from North.
    - rotation_angle_2: The rotation angle of the image's y-axis (typically Declination)
      from the East in degrees. A positive value indicates a counterclockwise rotation from East.

    The rotation angles help in understanding how the image is aligned with the celestial coordinate system,
    which is crucial for accurate star positioning and data alignment in astronomical observations.
    """
    rotation_angle_1 = np.degrees(np.arctan(cd1_2 / cd1_1))
    rotation_angle_2 = np.degrees(np.arctan(cd2_1 / cd2_2))

    return rotation_angle_1, rotation_angle_2

for inim in calimlist:
	with fits.open(inim, mode='update') as hdul:
		header = hdul[0].header

		# CD 행렬 값 추출
		cd1_1 = header.get('CD1_1', 0)
		cd1_2 = header.get('CD1_2', 0)
		cd2_1 = header.get('CD2_1', 0)
		cd2_2 = header.get('CD2_2', 0)

		if cd1_1 != 0 and cd1_2 != 0 and cd2_1 != 0 and cd2_2 != 0:
			# 필드 회전 계산
			rotation_angle_1, rotation_angle_2 = calculate_field_rotation(cd1_1, cd1_2, cd2_1, cd2_2)
		else:
			rotation_angle_1, rotation_angle_2 = None, None

		# 헤더 업데이트
		header.set('ROTANG1', rotation_angle_1, 'Rotation angle from North [deg]')
		header.set('ROTANG2', rotation_angle_2, 'Rotation angle from East [deg]')



#======================================================================
#	Slack message
#======================================================================
# delt_total = round(timetbl['time'][timetbl['process']=='total'].item()/60., 1)
delt_total = (time.time() - starttime)/60.
timetbl['status'][timetbl['process']=='total'] = True
timetbl['time'][timetbl['process']=='total'] = delt_total

channel = '#pipeline'
text = f'[`gpPy`+GPU/{project}-TEST] Processing Complete {obs} {os.path.basename(path_new)} Data ({nobj} objects) with {ncore} cores taking {delt_total:.1f} mins'

param_slack = dict(
	token = OAuth_Token,
	channel = channel,
	text = text,
)

tool.slack_bot(**param_slack)







# stop


# ic_com_phot = ImageFileCollection(path_data, glob_include='Calib*com.fits', keywords='*')	
#	Summary
# print('Draw observation summary plots')
# for filte in list(set(ic_cal_phot.summary['filter'])):
# for filte in filterlist:
# 	try:
# 		tool.obs_summary(filte, ic_cal_phot, ic_com_phot, path_save=path_data)
# 	except:
# 		print('Fail to make summary plots.')
# 		pass
# 	plt.close('all')


# #	Image subtraction
# 

# %%
"""
print('IMAGE SUBTRACTION')
subtracted_images = []
ds9comlist = []
for inim in combined_images:
	hdr = fits.getheader(inim)
	# obs = os.path.basename(inim).split('-')[1]
	# obs = 'LOAO'
	obj = hdr['object']
	filte = hdr['filter']
	path_refim = '/data3/paek/factory/ref_frames/{}'.format(obs)
	refimlist = glob.glob('{}/Ref*{}*{}*.fits'.format(path_refim, obj, filte))
	if len(refimlist) > 0:
		refim = refimlist[0]

		# subim, ds9com = tool.subtraction_routine3(inim, refim)

		# if False:
		if obs not in ['LSGT', 'DOAO', 'RASA36', 'SAO_C361K',]:
			subim, ds9com = tool.subtraction_routine(inim, refim)
		else:
			subim, ds9com = tool.subtraction_routine2(inim, refim)
			if os.path.getsize(subim) != 0:
				rmcom = f"rm {subim}"
				print(rmcom)
				os.system(rmcom)
				subim, ds9com = tool.subtraction_routine(inim, refim)
			else:
				pass
		if subim != None:
			subtracted_images.append(subim)
			ds9comlist.append(ds9com)
	else:
		print('There is no reference image for {}'.format(os.path.basename(inim)))
		pass
rmcom = 'rm {}/*Ref*gregister.fits'.format(path_data)
print(rmcom)
os.system(rmcom)
# tdict['subtraction'] = time.time() - st - tdict[list(tdict.keys())[-1]]
timetbl['status'][timetbl['process']=='subtraction'] = True
timetbl['time'][timetbl['process']=='subtraction'] = int(time.time() - st_)


# ##	Photometry for subtracted images
# 

# %%
st_ = time.time()
#	Write photometry configuration
s = open(path_new_gphot, 'w')
for line in lines:
	if 'imkey' in line:
		# line = '{}\t{}/hd*com.fits'.format('imkey', path_data)
		line = '{}\t{}/hd*.fits'.format('imkey', path_data)
	else:
		pass
	if 'photfraction' in line:
		line = '{}\t{}'.format('photfraction', 1.0)
	else:
		pass
	if 'DETECT_MINAREA' in line:
		line = '{}\t{}'.format('DETECT_MINAREA', 10)
	else:
		pass
	if 'DETECT_THRESH' in line:
		line = '{}\t{}'.format('DETECT_THRESH', 1.25)
	else:
		pass
	s.write(line+'\n')
s.close()
#	Execute
hdimlist = sorted(glob.glob('{}/hd*.fits'.format(path_data)))
if len(hdimlist) > 0:
	com = 'python {} {}'.format(path_phot_sub, path_data)
	print(com)
	os.system(com)
	# tdict['photometry_sub'] = time.time() - st - tdict[list(tdict.keys())[-1]]
else:
	print('No subtracted image.')
	pass
timetbl['status'][timetbl['process']=='photometry_sub'] = True
timetbl['time'][timetbl['process']=='photometry_sub'] = int(time.time() - st_)


# ##	Transient Search
# 

# %%
st_ = time.time()
fovval = fov.value
#	Input table for transient search
tstbl = Table()
# hdimlist = sorted(glob.glob(f'{path_data}/hd*com.fits'))
hdimlist = sorted(glob.glob(f'{path_data}/hd*.fits'))
if len(hdimlist) != 0:
	tstbl['hdim'] = hdimlist
	tskeys = ['hdcat', 'hcim', 'inim', 'scicat', 'refim']
	for key in tskeys:
		tstbl[key] = ' '*300
	tstbl['fovval'] = fovval

	for i, hdim in enumerate(hdimlist):
		hdcat = hdim.replace('.fits','.phot_sub.cat')
		hcim = hdim.replace('hdCalib', 'hcCalib')
		inim = hdim.replace('hdCalib', 'Calib')
		scicat = inim.replace('.fits', '.phot.cat')

		hdr = fits.getheader(hdim)
		obj = hdr['object']
		filte = hdr['filter']
		path_refim = f'/data3/paek/factory/ref_frames/{obs}'
		refimlist = glob.glob(f'{path_refim}/Ref*{obj}*{filte}*.fits')
		refim = refimlist[0]


		for key, im in zip(tskeys, [hdcat, hcim, inim, scicat, refim]):
			tstbl[key][i] = im

	out_tstbl = f'{path_data}/transient_search.txt'
	tstbl.write(out_tstbl, format='ascii.tab', overwrite=True)

	com = f'python {path_find} {out_tstbl} {ncore}'
	print(com)
	subprocess.call(com, shell=True)		


timetbl['status'][timetbl['process']=='transient_search'] = True
timetbl['time'][timetbl['process']=='transient_search'] = int(time.time() - st_)


# #	Summary file

# %%
#------------------------------------------------------------
#------------------------------------------------------------
timetbl['status'][timetbl['process']=='total'] = True
timetbl['time'][timetbl['process']=='total'] = int(time.time() - st)	
timetbl.write('{}/obs.summary.log'.format(path_data), format='ascii.tab', overwrite=True)
print(timetbl)
#	Write data summary
f = open(path_data+'/obs.summary.log', 'a')
end_localtime = time.strftime('%Y-%m-%d %H:%M:%S (%Z)', time.localtime())
f.write('Pipelne start\t: {}\n'.format(start_localtime))
f.write('Pipelne end\t: {}\n'.format(end_localtime))
try:
	f.write('='*60+'\n')
	f.write('PATH :{}\n'.format(path))
	f.write('OBJECT NUMBER # :{}\n'.format(len(ic_cal.summary)))
	objkind = sorted(set(ic_cal.summary['object']))
	f.write('OBJECTS # : {}\n'.format(objkind))
	for obj in objkind:
		f.write('-'*60+'\n')
		for filte in list(set(ic_cal.summary['filter'])):
			indx_tmp = ic_cal.files_filtered(filter=filte, object=obj)
			if len(indx_tmp) > 0:
				f.write('{}\t{}\n'.format(obj, filte))
except:
	pass
f.close()


# ## File Transfer

# %%
rmcom = 'rm {}/inv*.*'.format(path_data, path_data)
print(rmcom)
os.system(rmcom)
tails = ['.transients.', '.new.', '.ref.', '.sub.', '']
for obj in objlist:
	for filte in filterlist:
		for tail in tails:
			# tail = 'transients'
			# obj = 'NGC3147'
			# filte = 'B'

			pathto = f'{path_gal}/{obj}/{obs}/{filte}'
			files = f'{path_data}/*Calib*-{obj}-*-{filte}-*{tail}*'
			nfiles = len(glob.glob(files))
			# print(files, nfiles)
			# if nfiles >0:
			# 	print(obj, filte, pathto, files, glob.glob(files)[-1])
			if nfiles !=0:
				#	Save path
				if tail == '':
					pathto = f'{path_gal}/{obj}/{obs}/{filte}'
				else:
					pathto = f'{path_gal}/{obj}/{obs}/{filte}/transients'
				#	Make path
				if (not os.path.exists(pathto)):
					os.makedirs(pathto)
				mvcom = f'mv {files} {pathto}'
				print(mvcom)
				os.system(mvcom)
#	Image transfer
mvcom = f'mv {path_data} {path_save}'
os.system(mvcom)
#	WRITE LOG
f = open(path_log, 'a')
# f.write(path_raw+'/'+os.path.basename(path_data)+'\n')
# f.write('{}/{}\n'.format(path_raw, os.path.basename(path_data)))
f.write(f'{path_raw}/{os.path.basename(path_data)}\n')
f.close()


# ##	Slack message

# %%
delt_total = round(timetbl['time'][timetbl['process']=='total'].item()/60., 1)

channel = '#pipeline'
text = f'[`gpPy`/{project}-{obsmode}] Processing Complete {obs} {os.path.basename(path)} Data ({nobj} objects) with {ncore} cores taking {delt_total} mins'

param_slack = dict(
	token = OAuth_Token,
	channel = channel,
	text = text,
)

tool.slack_bot(**param_slack)


"""
