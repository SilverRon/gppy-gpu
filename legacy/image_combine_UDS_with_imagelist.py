from ccdproc import ImageFileCollection
#
# Python Library
import os, glob, sys
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime, timezone, timedelta
#
from astropy import units as u
from astropy.table import Table, vstack, hstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
#
from preprocess import calib
from util import tool
#
import warnings
warnings.filterwarnings("ignore")

# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#

path_config = './config'

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

#

path_calib = '/large_data/processed'
path_save = '/large_data/Commission'
keys = ['imagetyp', 'telescop', 'object', 'filter', 'exptime', 'ul5_1', 'seeing', 'elong', 'ellip']

obj = 'UDS'
filterlist = sorted(glob.glob(f'{path_calib}/{obj}/*/*'))
# for filterdir in filterlist:
# 	print(filterdir)
filte = sys.argv[1]

obslist = [f"7DT{n:0>2}" for n in np.arange(1, 21)]

obs = obslist[0]


path_to_table = f'/large_data/Commission/UDS/{filte}/filtered_image_header.csv'
comtbl = Table.read(path_to_table)
for ff, file in enumerate(comtbl['file']):
	if '20231105' not in file:
		n_cut = ff
		break
comtbl = comtbl[:n_cut]

# comtbl['image'] = [f"{pattern}/{inim}" for inim in comtbl['file']]
comtbl['image'] = comtbl['file']
n_table = len(comtbl)

print(f"{obs}: {n_table:>6} frames")

path_output = f"{path_save}/{obj}/{filte}"

if not os.path.exists(path_output):
	os.makedirs(path_output)

seltbl = comtbl.copy()

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


grouped_images = list(seltbl['image'])
print(f"{len(grouped_images)} images to stack")
for ii, inim in enumerate(grouped_images):
	if ii == 0:	
		print(f"- {ii:0>4}: {inim} <-- Base Image")
	else:
		print(f"- {ii:0>4}: {inim}")

#	Base Image for the Alignment
baseim = grouped_images[0]
# print(f"BASE IMAGE: {baseim}")
basecat = baseim.replace('fits', 'cat')
path_imagelist = f"{path_output}/{os.path.basename(baseim).replace('fits', 'image.list')}"



#	Global Background Subtraction
import glob

# imlist = sorted(glob.glob('calib*0.fits'))

print("BACKGROUND Subtraction...")
bkgsub_grouped_images = []
for ii, inim in enumerate(grouped_images):
	if filte != 'm525':
		# print(f"[{ii:0>4}] {os.path.basename(inim)}", end='')
		print(f"[{ii:0>4}] {os.path.basename(inim)}",)
		ninim = f"{path_output}/{os.path.basename(inim).replace('fits', 'subbkg.fits')}"
		if not os.path.exists(ninim):
			_data, _hdr = fits.getdata(inim, header=True)
			try:
				_bkg = _hdr['SKYVAL']
			except:
				_bkg = np.median(_data)
			_data -= _bkg
			print(f"\t(-{_bkg:.3f})")
			fits.writeto(ninim, _data, header=_hdr, overwrite=True)
			del _data, _hdr
		bkgsub_grouped_images.append(ninim)
	else:
		if '7DT03' in inim:
			# print(f"[{ii:0>4}] {os.path.basename(inim)}", end='')
			print(f"[{ii:0>4}] {os.path.basename(inim)}",)
			ninim = f"{path_output}/{os.path.basename(inim).replace('fits', 'subbkg.fits')}"
			if not os.path.exists(ninim):
				_data, _hdr = fits.getdata(inim, header=True)
				_bkg = _hdr['SKYVAL']
				_data -= _bkg
				print(f"\t(-{_bkg:.3f})")
				fits.writeto(ninim, _data, header=_hdr, overwrite=True)
				del _data, _hdr
			bkgsub_grouped_images.append(ninim)

#	Images to Combine for SWarp
f = open(path_imagelist, 'w')
# for inim in grouped_images:
for inim in bkgsub_grouped_images:
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
# for _inim in grouped_images:
# for _inim in bkgsub_grouped_images:
# for _inim in grouped_images:
for _inim in seltbl['image']:
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

#	Center Coordinate
objra = header['OBJCTRA']
objdec = header['OBJCTDEC']

# objra = objra.replace(' ', ':')
# objdec = objdec.replace(' ', ':')

# objra = '02:17:37.69'
# objdec = '-05:13:13.41'

objra = '02:17:37.0'
objdec = '-05:03:12.0'

center = f"{objra},{objdec}"

datestr, timestr = calib.extract_date_and_time(dateobs_combined)
comim = f"{path_output}/calib_7DT00_{obj}_{datestr}_{timestr}_{filte}_{exptime_combined}.com.fits"

#	Image Combine
t0_com = time.time()

# if input('BKG Subtraction? (y/n)')=='y':
#     swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -BACK_SIZE 1024"
# else:
#	swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N"

n_image = len(seltbl)
total_exptime = 60.*n_image
print(f"Total Exptime: {total_exptime}")

if not os.path.exists(comim):
	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output}"
	swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output} -GAIN_KEYWORD FAKE -GAIN_DEFAULT {total_exptime}"


	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK Y -BACK_SIZE 1024 -RESAMPLE_DIR {path_output}"
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
		'SKYVAL'  : (0,                'SKY MEDIAN VALUE (Subtracted)')
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
