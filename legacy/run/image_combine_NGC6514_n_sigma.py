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

path_calib = '/large_data/processed.231213.bkg'
path_save = '/large_data/factory'
keys = ['imagetyp', 'telescop', 'object', 'filter', 'exptime', 'ul5_1', 'seeing', 'elong', 'ellip']

obj = 'NGC6514'
# obj = 'NGC0253'
# filte = 'm400'
# obj = input("WHICH OBJECT?")
filterlist = sorted(glob.glob(f'{path_calib}/{obj}/*/*'))
for filterdir in filterlist:
	print(filterdir)
# filte = input("WHICH FILTER?")
filte = sys.argv[1]

obslist = [f"7DT{n:0>2}" for n in np.arange(1, 21)]

obs = obslist[0]
tablelist = []
for obs in obslist:
	pattern = f"{path_calib}/{obj}/{obs}/{filte}"
	try:
		ic_cal = ImageFileCollection(pattern, glob_include='calib*0.fits', keywords=keys)
		comtbl = ic_cal.summary
		comtbl['image'] = [f"{pattern}/{inim}" for inim in comtbl['file']]
		tablelist.append(comtbl)
		n_table = len(comtbl)
	except FileNotFoundError as e:
		n_table = 0
		pass

	print(f"{obs}: {n_table:>6} frames")

comtbl = vstack(tablelist)

comtbl = comtbl[
	(~comtbl['seeing'].mask) &
	(~comtbl['ul5_1'].mask)
]
#------------------------------------------------------------
seeingarr = np.array([val for val in comtbl['seeing']])
ularr = np.array([val for val in comtbl['ul5_1']])

# subplot_mosaic을 사용하여 레이아웃을 설정합니다.
mosaic = """
Ax
CD
"""
fig, axd = plt.subplot_mosaic(mosaic, figsize=(6, 6),
                              empty_sentinel = "x",
                              gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4], 'wspace':0., 'hspace':0.},
                              constrained_layout=True)

# 원래의 산점도
axd['C'].scatter(seeingarr, ularr, marker='.', alpha=0.75)

# 3시그마 선 추가
n_sigma = 1
seeing_med = np.median(seeingarr)
seeing_std = np.std(seeingarr)
seeing_cut = np.mean(seeingarr) + n_sigma * seeing_std

depth_med = np.median(ularr)
depth_std = np.std(ularr)
depth_cut = np.mean(ularr) - n_sigma * depth_std

print(f"# Depth")
print(f"- {n_sigma} Depth Cut: {depth_cut:.3f}")
print(f"- Median: {depth_med:.3f}")
print(f"- STD   : {depth_std:.3f}")

print(f"# Seeing")
print(f"- {n_sigma} Sigma Cut: {seeing_cut:.3f}")
print(f"- Median: {seeing_med:.3f}")
print(f"- STD   : {seeing_std:.3f}")

#	Seeing
axd['C'].axvline(seeing_cut, ls='-', c='tomato', alpha=0.75, zorder=0)
axd['C'].axvline(seeing_med, ls='--', c='dodgerblue', zorder=0,)
# axd['C'].axvline(np.mean(seeingarr) - 3 * seeing_std, ls='--', c='red')

#	Depth
# axd['C'].axhline(np.mean(ularr) + 3 * depth_std, ls='--', c='tomato')
axd['C'].axhline(depth_cut, ls='-', c='tomato', alpha=0.75, zorder=0, label=f'{n_sigma} sigma')
axd['C'].axhline(depth_med, ls='--', c='dodgerblue', zorder=0, label='Median')

# 히스토그램 추가
axd['A'].hist(seeingarr, bins=30, alpha=0.5, orientation='vertical')
axd['A'].axvline(np.mean(seeingarr) + n_sigma * seeing_std, ls='-', c='tomato')
axd['A'].axvline(seeing_med, ls='--', c='dodgerblue', zorder=0,)

axd['D'].hist(ularr, bins=30, alpha=0.5, orientation='horizontal')
axd['D'].axhline(np.mean(ularr) - n_sigma * depth_std, ls='-', c='tomato')
axd['D'].axhline(depth_med, ls='--', c='dodgerblue', zorder=0, label='Median')


# 레이블 추가 및 불필요한 레이블 제거
axd['C'].set_xlabel('Seeing [arcsec]')
axd['C'].set_ylabel('5 Sigma Depth')

axd['A'].xaxis.set_tick_params(labelbottom=False)
axd['D'].yaxis.set_tick_params(labelleft=False)

axd['C'].legend(loc='upper right')
plt.tight_layout()
#------------------------------------------------------------

path_output = f"{path_save}/{obj}/{filte}"

if not os.path.exists(path_output):
	os.makedirs(path_output)
	plt.savefig(f"{path_output}/depth_seeing.png")
	print(f"Seeing-Depth Figure Saved at:")
	print(f"{path_output}/depth_seeing_{obj}_{filte}.png")

# answer = input("Are you ready to combine? (y/n/s):")
answer = 's'
if answer == 'n':
	print(f"Exit the process. Please Check the {path_output}/depth_seeing.png to find the selection criteria.")
	sys.exit()
elif answer == 'y':
	# answer2 = input('Choose Mode (n_sigma, median, manual):')
	answer2 = 'n_sigma'
	if answer2 == 'manual':
		seeing_criteria = float(input("Upper Limit for Seeing (float):"))
		depth_criteria = float(input("Lower Limit for 5sig Depth (float):"))
	elif answer2 == 'n_sigma':
		seeing_criteria = seeing_cut
		depth_criteria = depth_cut
	elif answer2 == 'median':
		seeing_criteria = seeing_med
		depth_criteria = depth_cut

	print(f"Total Frames: {len(comtbl)}")
	print(f"- Seeing Criteria: {len(comtbl[seeingarr<seeing_criteria])} (seeing<{seeing_criteria:.3f} arcsec)")
	print(f"- Depth  Criteria: {len(comtbl[ularr<depth_criteria])} (depth>{depth_criteria:.3f} ABmag)")

	seltbl = comtbl[
		(seeingarr<seeing_criteria) &
		(ularr>depth_criteria)
	]

	print(f"--> Final Frames: {len(seltbl)}")

elif answer == 's':
	seltbl = comtbl.copy()	
	print(f"--> Final Frames: {len(seltbl)}")



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
	print(f"[{ii:0>4}] {os.path.basename(inim)}", end='')
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
for _inim in bkgsub_grouped_images:
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
# objra = header['OBJCTRA']
# objdec = header['OBJCTDEC']

# objra = objra.replace(' ', ':')
# objdec = objdec.replace(' ', ':')

# objra = '18:02:23'
# objdec = '-23:01:48'
objra = '18:02:42'
objdec = '-22:58:18'
center = f"{objra},{objdec}"

datestr, timestr = calib.extract_date_and_time(dateobs_combined)
comim = f"{path_output}/calib_7DT00_{obj}_{datestr}_{timestr}_{filte}_{exptime_combined}.com.fits"

#	Image Combine
t0_com = time.time()

# if input('BKG Subtraction? (y/n)')=='y':
#     swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -BACK_SIZE 1024"
# else:
#	swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N"


if not os.path.exists(comim):

	# swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output}"
	swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_output} -CELESTIAL_TYPE EQUATORIAL"
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
