#============================================================
#	Library
#------------------------------------------------------------
from ccdproc import ImageFileCollection
#------------------------------------------------------------
import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
import time
# time.sleep(60*60*2)
from datetime import datetime, timezone, timedelta
#------------------------------------------------------------
from astropy import units as u
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
#------------------------------------------------------------
from preprocess import calib
from util import tool
#------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
#------------------------------------------------------------
# Plot presetting
#------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#------------------------------------------------------------
#	Function
#------------------------------------------------------------
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
#------------------------------------------------------------
#	Path
#------------------------------------------------------------
path_config = './config'
path_calib = '/large_data/processed'
#------------------------------------------------------------
#	Setting
#------------------------------------------------------------
keys = ['imagetyp', 'telescop', 'object', 'filter', 'exptime', 'ul5_1', 'seeing', 'elong', 'ellip']
#------------------------------------------------------------
#	Input
#------------------------------------------------------------
#	Gain
# gain_default = 0.779809474945068
#	ZeroPoint Key
zpkey = f"ZP_AUTO"
#	Universal Facility Name
obs = '7DT'
#	
keys_to_remember = [
	"EGAIN",
	"TELESCOP",
	'EGAIN',
	"FILTER",
	"OBJECT",
	"OBJCTRA",
	"OBJCTDEC",
	"JD",
	"MJD",
	'SKYVAL',
	'EXPTIME',
	zpkey
]
#	Hard coding for the test
# imagelist_file_to_stack = f"/large_data/Commission/UDS/T03_m725_filelist.part.dat"
# imagelist_file_to_stack = f"/large_data/Commission/UDS/T03_m725_filelist.dat"
# imagelist_file_to_stack = f"/large_data/Commission/EP240408a/r.txt"
imagelist_file_to_stack = input(f"Image List to Stack (/data/data.txt):")
if os.path.exists(imagelist_file_to_stack):
	print(f"{imagelist_file_to_stack} found!")
else:
	print(f"Not Found {imagelist_file_to_stack}!")
	sys.exit()
input_table = Table.read(imagelist_file_to_stack, format='ascii')
_files = [f for f in input_table['file'].data]
#	Get Image Collection (take some times)
print(f"Read Images... (take a few mins)")
ic = ImageFileCollection(filenames=_files, keywords=keys_to_remember)
files = ic.files
n_stack = len(files)
zpvalues = ic.summary[zpkey].data
skyvalues = ic.summary['SKYVAL'].data
objra = ic.summary['OBJCTRA'].data[0].replace(' ', ':')
objdec = ic.summary['OBJCTDEC'].data[0].replace(' ', ':')
#
objs = np.unique(ic.summary['OBJECT'].data)
filters = np.unique(ic.summary['FILTER'].data)
egains = np.unique(ic.summary['EGAIN'].data)
print(f"OBJECT(s): {objs} (N={len(objs)})")
print(f"FILTER(s): {filters} (N={len(filters)})")
print(f"EGAIN(s): {egains} (N={len(egains)})")
#	OBJECT
if len(objs) != 1:
	print(f"There are more than {len(objs)} objects")
	obj = input(f"Type OBJECT name (e.g. {objs}):")
else:
	obj = objs[0]
#	FILTER
if len(filters) != 1:
	print(f"There are more than {len(filters)} filters")
	filte = input(f"Type FILTER name (e.g. m650):")
else:
	filte = filters[0]
#	EGAIN
if len(egains) != 1:
	print(f"There are more than {len(egains)} egains")
	gain_default = input(f"Type EGAIN name (e.g. 0.256190478801727):")
else:
	gain_default = egains[0]
#	Total Exposure Time [sec]
total_exptime = np.sum(ic.summary['EXPTIME'])
#------------------------------------------------------------
#	Summary Input
#------------------------------------------------------------
print(f"Input Images to Stack ({len(files):_}):")
for ii, inim in enumerate(files):
	print(f"[{ii:>6}] {os.path.basename(inim)}")
	if ii > 10:
		print("...")
		break
#------------------------------------------------------------
path_save = f'/large_data/Commission/{obj}/{filte}'
#	Image List for SWarp
path_imagelist = f'{path_save}/images_to_stack.txt'
#	Background Subtracted
path_bkgsub = f"{path_save}/bkgsub"
if not os.path.exists(path_bkgsub): os.makedirs(path_bkgsub)
#	Scaled
path_scaled = f"{path_save}/scaled"
if not os.path.exists(path_scaled): os.makedirs(path_scaled)
#	Resampled (temp. files from SWarp)
path_resamp = f"{path_save}/resamp"
if not os.path.exists(path_resamp): os.makedirs(path_resamp)
#------------------------------------------------------------
#	Keywords
#------------------------------------------------------------
keywords_to_add = [
    "IMAGETYP",
    # "EXPOSURE",
    # "EXPTIME",
    "DATE-LOC",
    # "DATE-OBS",
    "XBINNING",
    "YBINNING",
    # "GAIN",
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
    "CENTALT",
    "CENTAZ",
    "AIRMASS",
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
#------------------------------------------------------------
#	Base Image for the Alignment
#------------------------------------------------------------
baseim = files[0]
zp_base = ic.summary[zpkey][0]
#------------------------------------------------------------
#	Global Background Subtraction
#------------------------------------------------------------
print("BACKGROUND Subtraction...")
_st = time.time()
bkg_subtracted_images = []
for ii, (inim, _bkg) in enumerate(zip(files, skyvalues)):
	print(f"[{ii:>6}] {os.path.basename(inim)}", end='')
	nim = f"{path_bkgsub}/{os.path.basename(inim).replace('fits', 'bkgsub.fits')}"
	if not os.path.exists(nim):
		with fits.open(inim, memmap=True) as hdul:  # 파일 열기
			_data = hdul[0].data  # 데이터 접근
			_hdr = hdul[0].header  # 헤더 접근
			# _bkg = np.median(_data)
			_data -= _bkg
			print(f"- {_bkg:.3f}",)
			fits.writeto(nim, _data, header=_hdr, overwrite=True)
	bkg_subtracted_images.append(nim)
_delt = time.time() - _st
print(f"--> Done ({_delt:.1f}sec)")
#------------------------------------------------------------
#	ZP Scale
#------------------------------------------------------------
print(f"Flux Scale to ZP={zp_base}")
zpscaled_images = []
_st = time.time()
for ii, (inim, _zp) in enumerate(zip(bkg_subtracted_images, zpvalues)):
	print(f"[{ii:>6}] {os.path.basename(inim)}", end=' ')
	_fscaled_image = f"{path_scaled}/{os.path.basename(inim).replace('fits', 'zpscaled.fits')}"
	if not os.path.exists(_fscaled_image):
		with fits.open(inim, memmap=True) as hdul:  # 파일 열기
			_data = hdul[0].data  # 데이터 접근
			_hdr = hdul[0].header  # 헤더 접근
			_fscale = 10**(0.4*(zp_base-_zp))
			_fscaled_data = _data * _fscale
			print(f"x {_fscale:.3f}",)
			fits.writeto(_fscaled_image, _fscaled_data, _hdr, overwrite=True)
	zpscaled_images.append(_fscaled_image)
_delt = time.time() - _st
print(f"--> Done ({_delt:.1f}sec)")
#------------------------------------------------------------
#	Images to Combine for SWarp
#------------------------------------------------------------
f = open(path_imagelist, 'w')
for inim in zpscaled_images: f.write(f"{inim}\n")
f.close()
#	Get Header info
exptime_stacked = total_exptime
mjd_stacked = np.mean(ic.summary['MJD'].data)
jd_stacked = Time(mjd_stacked, format='mjd').jd
dateobs_stacked = Time(mjd_stacked, format='mjd').isot
# airmass_stacked = np.mean(airmasslist)
# dateloc_stacked = calc_mean_dateloc(dateloclist)
# alt_stacked = np.mean(altlist)
# az_stacked = np.mean(azlist)

center = f"{objra},{objdec}"
datestr, timestr = calib.extract_date_and_time(dateobs_stacked)
comim = f"{path_save}/calib_{obs}_{obj}_{datestr}_{timestr}_{filte}_{exptime_stacked:g}.com.fits"
weightim = comim.replace("com", "weight")
#------------------------------------------------------------
#	Image Combine
#------------------------------------------------------------
t0_stack = time.time()
print(f"Total Exptime: {total_exptime}")
gain = (2/3)*n_stack*gain_default
#	SWarp
# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} -GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} -FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
swarpcom = (
	f"swarp -c {path_config}/7dt.swarp @{path_imagelist} "
	f"-IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} "
	f"-SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} "
	f"-GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} "
	f"-FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
)

print(swarpcom)
os.system(swarpcom)

# t0_stack = time.time()
# swarpcom = f"swarp -c {path_config}/7dt.nocom.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center}"
# print(swarpcom)
# os.system(swarpcom)
# delt_stack = time.time()-t0_stack
# print(delt_stack)

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

#	Additional Header Information
keywords_to_update = {
	'DATE-OBS': (dateobs_stacked, 'Time of observation (UTC) for combined image'),
	# 'DATE-LOC': (dateloc_stacked, 'Time of observation (local) for combined image'),
	'EXPTIME' : (exptime_stacked, '[s] Total exposure duration for combined image'),
	'EXPOSURE': (exptime_stacked, '[s] Total exposure duration for combined image'),
	# 'CENTALT' : (alt_stacked,     '[deg] Average altitude of telescope for combined image'),
	# 'CENTAZ'  : (az_stacked,      '[deg] Average azimuth of telescope for combined image'),
	# 'AIRMASS' : (airmass_stacked, 'Average airmass at frame center for combined image (Gueymard 1993)'),
	'MJD'     : (mjd_stacked,     'Modified Julian Date at start of observations for combined image'),
	'JD'      : (jd_stacked,      'Julian Date at start of observations for combined image'),
	'SKYVAL'  : (0,               'SKY MEDIAN VALUE (Subtracted)'),
	'GAIN'    : (gain,            'Sensor gain'),
}

#	Header Update
with fits.open(comim, mode='update') as hdul:
	# 헤더 정보 가져오기
	header = hdul[0].header

	# 여러 헤더 항목 업데이트
	for key, (value, comment) in keywords_to_update.items():
		header[key] = (value, comment)

	#	Stacked Single images
	# for nn, inim in enumerate(files):
	# 	header[f"IMG{nn:0>5}"] = (os.path.basename(inim), "")

	# 변경 사항 저장
	hdul.flush()

delt_stack = time.time() - t0_stack
print(f"Time to stack {n_stack} images: {delt_stack:.3f} sec")
