#	PHOTOMETRY CODE FOR PYTHON 3.X
#	CREATED	2020.12.10	Gregory S.H. Paek
#============================================================
import os, glob, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
#	Astropy
from astropy.table import Table, vstack, hstack
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
#
import warnings
warnings.filterwarnings('ignore', message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.")
warnings.filterwarnings('ignore', message="Warning: 'partition' will ignore the 'mask' of the MaskedColumn.")

# from astropy.utils.exceptions import FITSFixedWarning
# warnings.filterwarnings('ignore', category=FITSFixedWarning, append=True)
from datetime import date
import time
import multiprocessing
#	gpPy
sys.path.append('..')
sys.path.append('/home/gp/gppy')
from phot import gpphot
from util import query
from util import tool
from phot import gcurve
from preprocess import calib
starttime = time.time()
#============================================================
#	FUNCTION
#============================================================
def file2dict(path_infile):
	out_dict = dict()
	f = open(path_infile)
	for line in f:
		key, val = line.split()
		out_dict[key] = val
	return out_dict
#------------------------------------------------------------
def is_within_ellipse(x, y, center_x, center_y, a, b):
	term1 = ((x - center_x) ** 2) / (a ** 2)
	term2 = ((y - center_y) ** 2) / (b ** 2)
	return term1 + term2 <= 1
#------------------------------------------------------------
def weighted_median(values, errors):
	# 오차를 역수로 사용하여 가중치 계산
	weights = 1.0 / np.array(errors)
	# 중앙값 계산
	median = np.median(values)
	# 중앙값과 각 데이터 포인트의 차이 계산
	deviations = np.abs(values - median)
	# 가중치를 곱하여 가중 중앙값 계산
	weighted_median = np.median(deviations * weights)
	return median, weighted_median
#------------------------------------------------------------
def compute_median_mad(values):
	median = np.median(values)
	mad = np.median(np.abs(values - median))
	return median, mad

#------------------------------------------------------------
def compute_flux_density_error(magerr, flux_density):
    flux_density_error = (2.5/np.log(10))*(flux_density)*magerr
    # flux_density_error = (1.086)*(flux_density)*magerr
    
    return flux_density_error
#------------------------------------------------------------
def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """
    Calculate the corrected flux excess factor for the input Gaia EDR3 data.
    
    Parameters
    ----------
    
    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    phot_bp_rp_excess_factor: float, numpy.ndarray
        The flux excess factor listed in the Gaia EDR3 archive.
        
    Returns
    -------
    
    The corrected value for the flux excess factor, which is zero for "normal" stars.
    
    Example
    -------
    
    phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
    """
    
    if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
        bp_rp = np.float64(bp_rp)
        phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
    
    if bp_rp.shape != phot_bp_rp_excess_factor.shape:
        raise ValueError('Function parameters must be of the same shape!')
        
    do_not_correct = np.isnan(bp_rp)
    bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
    greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange = np.logical_not(do_not_correct) & (bp_rp >= 4.0)
    
    correction = np.zeros_like(bp_rp)
    correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
    correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) \
        - 0.005879*np.power(bp_rp[greenrange], 3)
    correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
    
    return phot_bp_rp_excess_factor - correction
#------------------------------------------------------------
# def phot_routine(inim):

#============================================================
#	USER SETTING
#============================================================
#	PATH
#------------------------------------------------------------
try:
	# obs = (sys.argv[1]).upper()
	path_base = sys.argv[1]
except:
	path_base = '.'

path_refcat	= f'/large_data/factory/ref_cat'
path_config = '/home/gp/gppy/config'
path_to_filterset = f"{path_config}/filterset"
path_obs = f'{path_config}/obs.dat'
# path_target = './transient.dat'
path_gphot = f'{path_base}/gphot.config'
path_default_gphot = f'{path_config}/gphot.config'
# path_calibration_field = "/large_data/Calibration/7DT-Calibration/output/Calibration_Field"
path_calibration_field = "/large_data/Calibration/7DT-Calibration/output/Calibration_Tile"

#------------------------------------------------------------
print(path_gphot)
if os.path.exists(path_gphot) == True:
	gphot_dict = file2dict(path_gphot)
else:
	gphot_dict = file2dict(path_default_gphot)
	print('There is no gregoryphot configuration. Use default.')
#------------------------------------------------------------
# inim = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/hdcalib_7DT03_T09614_20240423_020757_r_360.com.fits"
# mask_image = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/hdcalib_7DT03_T09614_20240423_020757_r_360.com.mask.fits"
inim = sys.argv[1]
mask_image = sys.argv[2]

# refqueryradius = float(gphot_dict['refqueryradius'])# *u.degree
# frac = float(gphot_dict['photfraction'])
# refcatname = gphot_dict['refcatname']
# refmaglower = float(gphot_dict['refmaglower'])
# refmagupper = float(gphot_dict['refmagupper'])
# refmagerupper = float(gphot_dict['refmagerupper'])
# inmagerupper = float(gphot_dict['inmagerupper'])
# flagcut = int(gphot_dict['flagcut'])
# check = (gphot_dict['check'] == 'True')

check = False
n_binning = 1

DETECT_MINAREA = gphot_dict['DETECT_MINAREA']
DETECT_THRESH = gphot_dict['DETECT_THRESH']
DEBLEND_NTHRESH = gphot_dict['DEBLEND_NTHRESH']
DEBLEND_MINCONT = gphot_dict['DEBLEND_MINCONT']
BACK_SIZE = gphot_dict['BACK_SIZE']
BACK_FILTERSIZE = gphot_dict['BACK_FILTERSIZE']
BACKPHOTO_TYPE = gphot_dict['BACKPHOTO_TYPE']
#------------------------------------------------------------
seeing_assume = 2.0 * u.arcsecond
#------------------------------------------------------------
# imlist = sorted(glob.glob(inim))

# ncore = 8
# ncore = 4
try:
	ncore = int(sys.argv[2])
except:
	ncore = 1

# print(f'#\t{len(imlist)} images to do photometry')
# print('='*60)
# for i, img in enumerate(imlist):
# 	print(f'{i:0>4} {img}')
# print('='*60)

try:
	n_binning = int(sys.argv[3])
except:
	n_binning = 1
#------------------------------------------------------------
fail_image_list = []
# for ii, inim in enumerate(imlist):
# 	try:
# 		phot_routine(inim)
# 	except Exception as e:
# 		print(f"\nPhotometry for {os.path.basename(inim)} was failed!\n")
# 		print(f"Error:\n{e}")
# 		fail_image_list.append(inim)
#------------------------------------------------------------
#	Logging the Failed Images
#------------------------------------------------------------
# if len(fail_image_list) > 0:
# 	if f"{os.path.dirname(fail_image_list[0])}"!='':
# 		f = open(f"{os.path.dirname(fail_image_list[0])}/phot.fail.list", 'w')
# 	else:
# 		f = open(f"./phot.fail.list", 'w')
# 	for finim in fail_image_list:
# 		f.write(f"{os.path.basename(finim)}\n")
# 	f.close()
#------------------------------------------------------------
#	INFO. from file name
#------------------------------------------------------------
hdr = fits.getheader(inim)
part = os.path.basename(inim).split('_')
head = inim.replace('.fits', '')

obs = part[1]
obj = hdr['OBJECT']
# refmagkey = f"{hdr['FILTER']}_mag"
# refmagerkey = f"{hdr['FILTER']}_magerr"
# refsnrkey = f"{hdr['FILTER']}_snr"
refmagkey = f"mag_{hdr['FILTER']}"
refmagerkey = f"magerr_{hdr['FILTER']}"
refsnrkey = f"snr_{hdr['FILTER']}"
#------------------------------------------------------------
print(inim, obs, obj, refmagkey, refmagerkey)
obsdict = tool.getccdinfo(obs, path_obs)
# gain = obsdict['gain']
gain = hdr['EGAIN']
pixscale = obsdict['pixelscale'] * n_binning
fov = obsdict['fov']
#------------------------------------------------------------
#	OUTPUT NAMES
#------------------------------------------------------------
cat = f"{head}.cat"
cat_gc = f"{head}.gcurve.cat"
seg = f"{head}.seg.fits"
bkg = f"{head}.bkg.fits"
sub = f"{head}.sub.fits"
psf = f"{head}.psf"
aper = f"{head}.aper.fits"
#------------------------------------------------------------
#	Growth Curve Names
#------------------------------------------------------------
# param_gc = f"{path_config}/growthcurve.param"
# conv_gc = f"{path_config}/growthcurve.conv"
# nnw_gc = f"{path_config}/growthcurve.nnw"
# conf_gc = f"{path_config}/growthcurve.sex"
#------------------------------------------------------------
#	Photometry Names
#------------------------------------------------------------
conf_simple = f"{path_config}/simple_{n_binning}x{n_binning}.sex"
param_simple = f"{path_config}/simple.param"
nnw_simple = f"{path_config}/simple.nnw"
conv_simple = f"{path_config}/simple.conv"
#------------------------------------------------------------
param = f"{path_config}/gregorydet.param"
conv = f"{path_config}/gregoryphot.conv"
nnw = f"{path_config}/gregoryphot.nnw"
# conf = f"{path_config}/gregoryphot_{n_binning}x{n_binning}.sex"
conf = f"{path_config}/gregorydet.sex"
#------------------------------------------------------------
#	Aperture determine (diameter for SE input)
#------------------------------------------------------------
# aper_lower = 1.0 * u.arcsecond/pixscale
# aper_upper = 10.0 * u.arcsecond/pixscale
# apertures = np.linspace(aper_lower, aper_upper, 32)
# aper_input = ''
# for i in apertures.value: aper_input = aper_input+'{},'.format(i)
# aper_input = aper_input[:-1]

print('-'*60)
print(inim)
# print(f'{obs}\t{obj} in {refmagkey}-band'.format(obs, obj, refmagkey))
print(f'{obs}\t{obj} in {refmagkey}'.format(obs, obj, refmagkey))
print('-'*60)
#------------------------------------------------------------
#	INFO. from file header
#------------------------------------------------------------
hdul = fits.open(inim)
hdr = hdul[0].header
# hdr = fits.getheader(inim)
#	RA, Dec center for reference catalog query
xcent, ycent= hdr['NAXIS1']/2., hdr['NAXIS2']/2.
w = WCS(inim)
racent, decent = w.all_pix2world(xcent, ycent, 1)
racent, decent = racent.item(), decent.item()
# print(racent, decent)
# print('BAD WCS INFORMATION?')
# racent, decent = hdr['CRVAL1'], hdr['CRVAL2']
#------------------------------------------------------------
dateobs = hdr['DATE-OBS']
timeobj = Time(dateobs, format='isot')
jd = timeobj.jd
mjd = timeobj.mjd

seeing = hdr['SEEING']

#------------------------------------------------------------
#	DATE-OBS, JD
#------------------------------------------------------------

#------------------------------------------------------------
#	APERTURE SETTING
#------------------------------------------------------------
# seeing = hdr['SEEING']
# seeing_input = str(seeing)
# seeing_input = str(2.0)
# peeing = seeing*pixscale.value
peeing = seeing/pixscale.value
#	Aperture Dictionary
aperture_dict = {
	'MAG_AUTO'  : (0., 'MAG_AUTO DIAMETER [pix]'),
	'MAG_APER'  : (2*0.6731*peeing, 'BEST GAUSSIAN APERTURE DIAMETER [pix]'),
	'MAG_APER_1': (2*peeing, '2*SEEING APERTURE DIAMETER [pix]'),
	'MAG_APER_2': (3*peeing, '3*SEEING APERTURE DIAMETER [pix]'),
	'MAG_APER_3': (3/pixscale.value, """FIXED 3" APERTURE DIAMETER [pix]"""),
	'MAG_APER_4': (5/pixscale.value, """FIXED 5" APERTURE DIAMETER [pix]"""),
	'MAG_APER_5': (10/pixscale.value, """FIXED 10" APERTURE DIAMETER [pix]"""),
}

add_aperture_dict = {}
for key in list(aperture_dict.keys()):
	add_aperture_dict[key.replace('MAG_', '')] = (round(aperture_dict[key][0], 3), aperture_dict[key][1])
#	MAG KEY
inmagkeys = list(aperture_dict.keys())
#	MAG ERROR KEY
inmagerkeys = [key.replace('MAG_', 'MAGERR_') for key in inmagkeys]
#	Aperture Sizes
aperlist = [aperture_dict[key][0] for key in inmagkeys[1:]]

PHOT_APERTURES = ','.join(map(str, aperlist))
#------------------------------------------------------------
#	SOURCE EXTRACTOR CONFIGURATION FOR PHOTOMETRY
#------------------------------------------------------------
# optaper = 50
param_insex = dict(	#------------------------------
					#	CATALOG
					#------------------------------
					CATALOG_NAME = cat,
					#------------------------------
					#	CONFIG FILES
					#------------------------------
					CONF_NAME = conf,
					PARAMETERS_NAME = param,
					FILTER_NAME = conv,    
					STARNNW_NAME = nnw,
					#------------------------------
					#	EXTRACTION
					#------------------------------			
					# PSF_NAME = psf,
					DETECT_MINAREA = DETECT_MINAREA,
					DETECT_THRESH = DETECT_THRESH,
					DEBLEND_NTHRESH = DEBLEND_NTHRESH,
					DEBLEND_MINCONT = DEBLEND_MINCONT,
					#------------------------------
					#	PHOTOMETRY
					#------------------------------
					#	DIAMETER
					#	OPT.APER, (SEEING x2), x3, x4, x5
					#	MAG_APER	OPT.APER
					#	MAG_APER_1	OPT.GAUSSIAN.APER
					#	MAG_APER_2	SEEINGx2
					#	...
					PHOT_APERTURES = PHOT_APERTURES,
					SATUR_LEVEL  = '65000.0',
					# GAIN = str(gain.value),
					GAIN = str(gain),
					PIXEL_SCALE = str(pixscale.value),
					#------------------------------
					#	MASKING IMAGE
					#------------------------------
					FLAG_IMAGE = mask_image,
					FLAG_TYPE = "OR",
					#------------------------------
					#	STAR/GALAXY SEPARATION
					#------------------------------
					SEEING_FWHM = str(seeing_assume.value),
					#------------------------------
					#	BACKGROUND
					#------------------------------
					BACK_SIZE = BACK_SIZE,
					BACK_FILTERSIZE = BACK_FILTERSIZE,
					BACKPHOTO_TYPE = BACKPHOTO_TYPE,
					#------------------------------
					#	CHECK IMAGE
					#------------------------------
					# CHECKIMAGE_TYPE = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND',
					# CHECKIMAGE_NAME = '{},{},{},{}'.format(seg, aper, bkg, sub),
					)
#	Add Weight Map from SWarp
weightim = inim.replace("com", "weight")
if "com" in inim:
	if os.path.exists(weightim):
		param_insex['WEIGHT_TYPE'] = "MAP_WEIGHT"
		param_insex['WEIGHT_IMAGE'] = weightim
#	Check Image
if check == True:
	param_insex['CHECKIMAGE_TYPE'] = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND'
	param_insex['CHECKIMAGE_NAME'] = f'{seg},{aper},{bkg},{sub}'
else:
	pass

print('2. SOURCE EXTRACTOR')
com = gpphot.sexcom(inim, param_insex)
t0_sex = time.time()
print(com)
sexout = subprocess.getoutput(com)
delt_sex = time.time() - t0_sex
print(f"SourceEXtractor: {delt_sex:.3f} sec")
line = [s for s in sexout.split('\n') if 'RMS' in s]
skymed, skysig = float(line[0].split('Background:')[1].split('RMS:')[0]), float(line[0].split('RMS:')[1].split('/')[0])
# os.system(f'rm {seg} {aper} {bkg} {sub}'.format(seg, aper, bkg, sub))

setbl = Table.read(cat, format='ascii.sextractor')
setbl['IMAFLAGS_ISO'] -= 128
# setbl = Table.read(cat, format='fits')


#------------------------------------------------------------
#	Header
#------------------------------------------------------------
header_to_add = {
	'AUTHOR': ('Gregory S.H. Paek', 'PHOTOMETRY AUTHOR'),
	'PHOTIME': (date.today().isoformat(), 'PHTOMETRY TIME [KR]'),
	#	Time
	'JD': (jd, 'Julian Date of the observation'),
	'MJD': (mjd, 'Modified Julian Date of the observation'),
	#	Image Definition
	# 'SEEING': (round(seeing, 3), 'SEEING [arcsec]'),
	# 'PEEING': (round(peeing, 3), 'SEEING [pixel]'),
	# 'ELLIP': (round(ellipticity, 3), 'ELLIPTICITY 1-B/A [0-1]'),
	# 'ELONG': (round(elongation, 3), 'ELONGATION A/B [1-]'),
	'SKYSIG': (round(skysig, 3), 'SKY SIGMA VALUE'),
	'SKYVAL': (round(skymed, 3), 'SKY MEDIAN VALUE'),
	#	Reference Source Conditions for ZP
	# 'REFCAT': (refcatname, 'REFERENCE CATALOG NAME'),
	# 'MAGLOW': (refmaglower, 'REF MAG RANGE, LOWER LIMIT'),
	# 'MAGUP': (refmagupper, 'REF MAG RANGE, UPPER LIMIT'),
	# 'STDNUMB': (len(zptbl), '# OF STD STARS TO CALIBRATE ZP'),
}

header_to_add.update(add_aperture_dict)


for nn, inmagkey in enumerate(inmagkeys):
	if inmagkey == 'MAG_AUTO':
		_zpkey = inmagkey.replace("MAG", "ZP")
		_zperrkey = inmagkey.replace("MAG", "EZP")
	elif inmagkey == 'MAG_APER':
		_zpkey = inmagkey.replace("MAG_APER", "ZP")+"_0"
		_zperrkey = inmagkey.replace("MAG_APER", "EZP")+"_0"
	else:
		_zpkey = inmagkey.replace("MAG_APER", "ZP")
		_zperrkey = inmagkey.replace("MAG_APER", "EZP")
	# print(_zpkey, _zperrkey)

	zp = hdr[_zpkey]
	zperr = hdr[_zperrkey]

	inmagerrkey = inmagkey.replace("MAG", 'MAGERR')
	#	Apply ZP
	##	MAG
	_calmagkey = f"{inmagkey}_{hdr['FILTER']}"
	_calmagerrkey = f"{inmagerrkey}_{hdr['FILTER']}"
	##	FLUX
	_calfluxkey = _calmagkey.replace('MAG', 'FLUX')
	_calfluxerrkey = _calmagerrkey.replace('MAG', 'FLUX')
	##  SNR
	_calsnrkey = _calmagkey.replace('MAG', 'SNR')

	setbl[_calmagkey] = setbl[inmagkey]+zp
	setbl[_calmagerrkey] = tool.sqsum(setbl[inmagerrkey], zperr)

	#	Flux [uJy]
	setbl[_calfluxkey] = (setbl[_calmagkey].data*u.ABmag).to(u.uJy).value
	# setbl[_calfluxerrkey] = setbl[_calfluxkey] * (10**(-0.4 * setbl[inmagerrkey]) - 1)
	# setbl[_calfluxerrkey] = compute_flux_density_error(magerr=setbl[_calmagerrkey], flux_density=setbl[_calfluxkey])
	setbl[_calfluxerrkey] = 0.4*np.log(10)*setbl[_calfluxkey]*setbl[_calmagerrkey]

	## SNR
	setbl[_calsnrkey] = setbl[_calfluxkey]/setbl[_calfluxerrkey]


	#	Formatting
	setbl[_calmagkey].format = '.3f'
	setbl[_calmagerrkey].format = '.3f'
	setbl[_calfluxkey].format = '.3f'
	setbl[_calfluxerrkey].format = '.3f'

	#	Depth Calculation
	aperture_size = aperture_dict[inmagkey][0]
	if inmagkey == 'MAG_AUTO':
		ul_3sig = 0.0
		ul_5sig = 0.0
	else:
		ul_3sig = gpphot.limitmag(3, zp, aperture_size, skysig)
		ul_5sig = gpphot.limitmag(5, zp, aperture_size, skysig)

#------------------------------------------------------------
#	ADD HEADER INFO
#------------------------------------------------------------
with fits.open(inim, mode='update') as hdul:
	header = hdul[0].header
	for key, (value, comment) in header_to_add.items():
		header[key] = (value, comment)
	hdul.flush()

# setbl['obs'] = obs
# setbl['obj'] = obj
# setbl['filter'] = refmagkey
# setbl['date-obs'] = date_obs
# setbl['jd'] = jd
# setbl['mjd'] = mjd

meta_dict = {
	'obs': obs,
	'object': obj,
	'filter': hdr['FILTER'],
	'date-obs': hdr['date-obs'],
	'jd': jd,
	'mjd': mjd,
}
setbl.meta = meta_dict
setbl.write(f'{head}.phot.cat', format='ascii.tab', overwrite=True)

#------------------------------------------------------------
#	Time
#------------------------------------------------------------
delt = time.time() - starttime
dimen = 'seconds'
if delt > 60.:
	delt = delt/60.
	dimen = 'mins'
if delt > 60.:
	delt = delt/60.
	dimen = 'hours'
print(f'PHOTOMETRY IS DONE.\t({delt:.3f} {dimen})')

#------------------------------------------------------------
#	Mask check
#------------------------------------------------------------
# indx_0 = setbl['IMAFLAGS_ISO'] == 0 # no-masked
# indx_1 = setbl['IMAFLAGS_ISO'] == 1 # masked

# plt.plot(setbl['X_IMAGE'][indx_0], setbl['Y_IMAGE'][indx_0], '.', label='SOURCES (FLAG==0)')
# plt.plot(setbl['X_IMAGE'][indx_1], setbl['Y_IMAGE'][indx_1], '.', label='MASKED (FLAG==1)')

# plt.legend()
# plt.tight_layout()
# plt.show()

photcat = cat.replace("cat", "phot.cat")
photbl = setbl[setbl["IMAFLAGS_ISO"]==0]

photbl.write(photcat, format='ascii.tab', overwrite=True)