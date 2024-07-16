#============================================================
#	Library
#------------------------------------------------------------
from ccdproc import ImageFileCollection
#------------------------------------------------------------
import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime, timezone, timedelta
#------------------------------------------------------------
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
#------------------------------------------------------------
# from preprocess import calib
# from util import tool
#------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")
#------------------------------------------------------------
# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#------------------------------------------------------------
path_config = './config'
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
path_calib = '/large_data/processed'
path_save = '/large_data/Test'
keys = ['imagetyp', 'telescop', 'object', 'filter', 'exptime', 'ul5_1', 'seeing', 'elong', 'ellip']
obj = 'UDS'
filterlist = sorted(glob.glob(f'{path_calib}/{obj}/*/*'))
# filte = sys.argv[1]
filte = 'm600'
obslist = [f"7DT{n:0>2}" for n in np.arange(1, 21)]
obs = obslist[0]
#------------------------------------------------------------
path_to_table = f'/large_data/Test/UDS/{filte}/filtered_image_header.csv'
comtbl = Table.read(path_to_table)
for ff, file in enumerate(comtbl['file']):
	if '20231105' not in file:
		n_cut = ff
		break
comtbl = comtbl[:n_cut]
#------------------------------------------------------------
comtbl['image'] = comtbl['file']
n_table = len(comtbl)
print(f"{obs}: {n_table:>6} frames")
path_output = f"{path_save}/{obj}/{filte}"
if not os.path.exists(path_output):
	os.makedirs(path_output)
seltbl = comtbl.copy()
#------------------------------------------------------------
#	Keywords
#------------------------------------------------------------
keywords_to_add = [
    "IMAGETYP",
    # "EXPOSURE",
    # "EXPTIME",
    # "DATE-LOC",
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
#------------------------------------------------------------
# number_to_test = 60
number_to_test = 10
# suffix = 'zpscale30_localbkgsub'
bkgval = 8192
suffix = f'localbkgsub{bkgval:g}'
grouped_images = list(seltbl['image'])[:number_to_test]
print(f"{len(grouped_images)} images to stack")
for ii, inim in enumerate(grouped_images):
	if ii == 0:	
		print(f"- {ii:0>4}: {inim} <-- Base Image")
	else:
		print(f"- {ii:0>4}: {inim}")
#------------------------------------------------------------
#	Base Image for the Alignment
baseim = grouped_images[0]
basecat = baseim.replace('fits', 'cat')
# path_imagelist = f"{path_output}/{os.path.basename(baseim).replace('fits', 'image.list')}"
path_imagelist = f"{path_output}/images_to_test.list"

#	Global Background Subtraction
print("BACKGROUND Subtraction...")
bkgsub_grouped_images = []
for ii, inim in enumerate(grouped_images):
	if filte != 'm525':
		# print(f"[{ii:0>4}] {os.path.basename(inim)}", end='')
		print(f"[{ii:0>4}] {os.path.basename(inim)}",)
		nim = f"{path_output}/{os.path.basename(inim).replace('fits', 'subbkg.fits')}"
		if not os.path.exists(nim):
			_data, _hdr = fits.getdata(inim, header=True)
			try:
				_bkg = _hdr['SKYVAL']
			except:
				_bkg = np.median(_data)
			_data -= _bkg
			print(f"\t(-{_bkg:.3f})")
			fits.writeto(nim, _data, header=_hdr, overwrite=True)
			del _data, _hdr
		bkgsub_grouped_images.append(nim)
	else:
		if '7DT03' in inim:
			# print(f"[{ii:0>4}] {os.path.basename(inim)}", end='')
			print(f"[{ii:0>4}] {os.path.basename(inim)}",)
			nim = f"{path_output}/{os.path.basename(inim).replace('fits', 'subbkg.fits')}"
			if not os.path.exists(nim):
				_data, _hdr = fits.getdata(inim, header=True)
				_bkg = _hdr['SKYVAL']
				_data -= _bkg
				print(f"\t(-{_bkg:.3f})")
				fits.writeto(nim, _data, header=_hdr, overwrite=True)
				del _data, _hdr
			bkgsub_grouped_images.append(nim)

#	ZP Scale
# base_data, base_hdr = fits.getdata(baseim, header=True)
# base_zp_auto = base_hdr['ZP_AUTO']
base_zp_auto = 30

fscaled_images = []

print(f"Flux Scale to ZP={base_zp_auto}")
for ii, inim in enumerate(grouped_images):
	print(f"[{ii:0>4}] {os.path.basename(inim)}", end=' ')
	#	New Image
	_fscaled_image = f"{path_save}/{obj}/{filte}/{os.path.basename(inim).replace('fits', 'scaled.fits')}"
	if not os.path.exists(_fscaled_image):
		_st = time.time()
		_data, _hdr = fits.getdata(inim, header=True)
		_zp_auto = _hdr['ZP_AUTO']
		# _fscale = 10**(-0.4*(base_zp_auto-_zp_auto))
		_fscale = 10**(0.4*(base_zp_auto-_zp_auto))
		_fscaled_data = _data*_fscale
		print(f"x {_fscale:.3f}", end=' ')
		fits.writeto(_fscaled_image, _fscaled_data, _hdr, overwrite=True)
		_delt = time.time() - _st
		print(f"--> Done ({_delt:.1f}sec)")
	fscaled_images.append(_fscaled_image)
	# break

#	Images to Combine for SWarp
f = open(path_imagelist, 'w')
for inim in fscaled_images:
# for inim in grouped_images:
# for inim in bkgsub_grouped_images:
# for inim in comtbl['image']:
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
# for _inim in grouped_images:
# for _inim in seltbl['image'][:number_to_test]:
# for _inim in bkgsub_grouped_images:
	#	Open Image Header
	with fits.open(_inim) as hdulist:
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

objra = '02:17:37.0'
objdec = '-05:03:12.0'

center = f"{objra},{objdec}"
datestr, timestr = calib.extract_date_and_time(dateobs_combined)
# comim = f"{path_output}/calib_7DT00_{obj}_{datestr}_{timestr}_{filte}_{exptime_combined}.com.fits"
comim = f"{path_output}/calib_7DT00_{obj}_{datestr}_{timestr}_{filte}_{exptime_combined}.{suffix}.com.fits"
#	Image Combine
t0_com = time.time()

n_image = len(seltbl[:number_to_test])
total_exptime = 60.*n_image
print(f"Total Exptime: {total_exptime}")

gain = 0.779809474945068

if not os.path.exists(comim):
	#	FSCALE_KEYWORD FAKE, Default Gain
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain} -FSCALE_KEYWORD FAKE"
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain} -FSCALE_KEYWORD FAKE"
	swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE {bkgval:g} -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain} -FSCALE_KEYWORD FAKE"

	#	Same...
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {total_exptime}"

	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 1024 -RESAMPLE N -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain}"
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 1024 -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain}"
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 2048 -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain}"
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 4096 -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain}"
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 8192 -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {gain}"
	print(swarpcom)
	os.system(swarpcom)

	# t0_com = time.time()
	# swarpcom = f"swarp -c {path_config}/7dt.nocom.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center}"
	# print(swarpcom)
	# os.system(swarpcom)
	# delt_com = time.time()-t0_com
	# print(delt_com)

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
		'DATE-OBS': (dateobs_combined, 'Time of observation (UTC) for combined image'),
		'DATE-LOC': (dateloc_combined, 'Time of observation (local) for combined image'),
		'EXPTIME' : (exptime_combined, '[s] Total exposure duration for combined image'),
		'EXPOSURE': (exptime_combined, '[s] Total exposure duration for combined image'),
		'CENTALT' : (alt_combined,     '[deg] Average altitude of telescope for combined image'),
		'CENTAZ'  : (az_combined,      '[deg] Average azimuth of telescope for combined image'),
		'AIRMASS' : (airmass_combined, 'Average airmass at frame center for combined image (Gueymard 1993)'),
		'MJD'     : (mjd_combined,     'Modified Julian Date at start of observations for combined image'),
		'JD'      : (jd_combined,      'Julian Date at start of observations for combined image'),
		'SKYVAL'  : (0,                'SKY MEDIAN VALUE (Subtracted)'),
		'GAIN'    : (1.282364516115225,'Sensor gain'),
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

	delt_com = time.time() - t0_com
	print(f"Combied Time: {delt_com:.3f} sec")
