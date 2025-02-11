#	PHOTOMETRY CODE FOR PYTHON 3.X
#	CREATED	2020.12.10	Gregory S.H. Paek
#============================================================
import os, glob, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
#	Astropy
from astropy.table import Table, vstack, hstack
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
#
import warnings
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
def phot_routine(inim):
	#------------------------------------------------------------
	#	INFO. from file name
	#------------------------------------------------------------
	hdr = fits.getheader(inim)
	part = os.path.basename(inim).split('_')
	head = inim.replace('.fits', '')

	obs = part[1]
	obj = hdr['OBJECT']
	refmagkey = f"{hdr['FILTER']}_mag"
	refmagerkey = f"{hdr['FILTER']}_magerr"
	#------------------------------------------------------------
	print(inim, obs, obj, refmagkey, refmagerkey)
	obsdict = tool.getccdinfo(obs, path_obs)
	gain = obsdict['gain']
	pixscale = obsdict['pixelscale']
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
	conf_simple = f"{path_config}/simple.sex"
	param_simple = f"{path_config}/simple.param"
	nnw_simple = f"{path_config}/simple.nnw"
	conv_simple = f"{path_config}/simple.conv"
	#------------------------------------------------------------
	param = f"{path_config}/gregoryphot.param"
	conv = f"{path_config}/gregoryphot.conv"
	nnw = f"{path_config}/gregoryphot.nnw"
	conf = f"{path_config}/gregoryphot.sex"
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

	#------------------------------------------------------------
	#	DATE-OBS, JD
	#------------------------------------------------------------
	ref_gaiaxp_cat = f'{path_refcat}/XP_CONTINUOUS_RAW_{obj}.csv'
	ref_gaiaxp_synphot_cat = f'{path_refcat}/gaiaxp_dr3_synphot_{obj}.csv'
	if not os.path.exists(ref_gaiaxp_synphot_cat):
		reftbl = query.querybox_7dt(racent, decent, refqueryradius, ref_gaiaxp_cat, path_to_filterset)
		reftbl.write(ref_gaiaxp_synphot_cat)
	else:
		print(f"Read {ref_gaiaxp_synphot_cat} as reference catalog")
		reftbl = Table.read(ref_gaiaxp_synphot_cat, format='csv')
	#------------------------------------------------------------
	#	Pre-Source EXtractor
	#------------------------------------------------------------
	precat = f"{head}.pre.cat"
	presexcom = f"source-extractor -c {conf_simple} {inim} -FILTER_NAME {conv_simple} -STARNNW_NAME {nnw_simple} -PARAMETERS_NAME {param_simple} -CATALOG_NAME {precat}"
	print(presexcom)
	os.system(presexcom)
	pretbl = Table.read(precat, format='ascii.sextractor')
	#
	pretbl['within_ellipse'] = is_within_ellipse(pretbl['X_IMAGE'], pretbl['Y_IMAGE'], xcent, ycent, frac*hdr['NAXIS1']/2, frac*hdr['NAXIS2']/2)

	indx_star4seeing = np.where(
		#	Star-like Source
		(pretbl['CLASS_STAR']>0.9) &
		(pretbl['FLAGS']==0) &
		#	Within Ellipse
		(pretbl['within_ellipse'] == True)
	)

	ellipticity = np.median(pretbl['ELLIPTICITY'][indx_star4seeing])
	elongation = np.median(pretbl['ELONGATION'][indx_star4seeing])
	seeing = np.median(pretbl['FWHM_WORLD'][indx_star4seeing]*3600)

	print(f"-"*60)
	print(f"{len(pretbl[indx_star4seeing])} Star-like Sources Found")
	print(f"-"*60)
	print(f"SEEING     : {seeing:.3f} arcsec")
	print(f"ELONGATION : {elongation:.3f}")
	print(f"ELLIPTICITY: {ellipticity:.3f}")

	#------------------------------------------------------------
	#	APERTURE SETTING
	#------------------------------------------------------------
	# seeing = hdr['SEEING']
	# seeing_input = str(seeing)
	# seeing_input = str(2.0)
	peeing = seeing*pixscale.value
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
						GAIN = str(gain.value),
						PIXEL_SCALE = str(pixscale.value),
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
	if check == True:
		param_insex['CHECKIMAGE_TYPE'] = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND'
		param_insex['CHECKIMAGE_NAME'] = f'{seg},{aper},{bkg},{sub}'
	else:
		pass

	print('2. SOURCE EXTRACTOR')
	com = gpphot.sexcom(inim, param_insex)
	t0_sex = time.time()
	sexout = subprocess.getoutput(com)
	delt_sex = time.time() - t0_sex
	print(f"SourceEXtractor: {delt_sex:.3f} sec")
	line = [s for s in sexout.split('\n') if 'RMS' in s]
	skymed, skysig = float(line[0].split('Background:')[1].split('RMS:')[0]), float(line[0].split('RMS:')[1].split('/')[0])
	# os.system(f'rm {seg} {aper} {bkg} {sub}'.format(seg, aper, bkg, sub))

	setbl = Table.read(cat, format='ascii.sextractor')
	# setbl = Table.read(cat, format='fits')

	#------------------------------------------------------------
	#	Matching
	#------------------------------------------------------------
	print('3. MATCHING')
	c_sex = SkyCoord(setbl['ALPHA_J2000'], setbl['DELTA_J2000'], unit='deg')
	c_ref = SkyCoord(reftbl['ra'], reftbl['dec'], unit='deg')
	indx_match, sep, _ = c_sex.match_to_catalog_sky(c_ref)
	_mtbl = hstack([setbl, reftbl[indx_match]])
	_mtbl['sep'] = sep.arcsec
	matching_radius = 1.
	mtbl = _mtbl[_mtbl['sep']<matching_radius]
	mtbl['within_ellipse'] = is_within_ellipse(mtbl['X_IMAGE'], mtbl['Y_IMAGE'], xcent, ycent, frac*hdr['NAXIS1']/2, frac*hdr['NAXIS2']/2)

	# mtbl['dist2center'] = tool.sqsum((xcent-mtbl['X_IMAGE']), (ycent-mtbl['Y_IMAGE']))
	# mtbl['xdist2center'] = np.abs(xcent-mtbl['X_IMAGE'])
	# mtbl['ydist2center'] = np.abs(ycent-mtbl['Y_IMAGE'])
	print(f"""Matched Sources: {len(mtbl)} (r={matching_radius:.3f}")""")

	#

	# dist2center_cut = frac*(xcent+ycent)/2.



	indx_star4zp = np.where(
		#	Star-like Source
		# (mtbl['CLASS_STAR']>0.9) &
		(mtbl['FLAGS']==0) &
		#	Within Ellipse
		(mtbl['within_ellipse'] == True) &
		#	Magnitude in Ref. Cat 
		(mtbl[f'{refmagkey}']<refmagupper) &
		(mtbl[f'{refmagkey}']>refmaglower) &
		(mtbl[f'{refmagerkey}']<refmaglower)
	)

	zptbl = mtbl[indx_star4zp]

	print(f"{len(zptbl)} sources to calibration ZP")


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
		'SEEING': (round(seeing, 3), 'SEEING [arcsec]'),
		'PEEING': (round(peeing, 3), 'SEEING [pixel]'),
		'ELLIP': (round(ellipticity, 3), 'ELLIPTICITY 1-B/A [0-1]'),
		'ELONG': (round(elongation, 3), 'ELONGATION A/B [1-]'),
		'SKYSIG': (round(skysig, 3), 'SKY SIGMA VALUE'),
		'SKYVAL': (round(skymed, 3), 'SKY MEDIAN VALUE'),
		#	Reference Source Conditions for ZP
		'REFCAT': (refcatname, 'REFERENCE CATALOG NAME'),
		'MAGLOW': (refmaglower, 'REF MAG RANGE, LOWER LIMIT'),
		'MAGUP': (refmagupper, 'REF MAG RANGE, UPPER LIMIT'),
		'STDNUMB': (len(zptbl), '# OF STD STARS TO CALIBRATE ZP'),
	}

	header_to_add.update(add_aperture_dict)
	header_to_add
	#------------------------------------------------------------
	#	ZEROPOINT CALCULATION
	#------------------------------------------------------------
	print('4. ZERO POINT CALCULATION')

	# nn = 0
	# nn = 1
	# nn = 2
	# inmagkey = inmagkeys[nn]
	for nn, inmagkey in enumerate(inmagkeys):
		inmagerrkey = inmagkey.replace("MAG", 'MAGERR')

		from astropy.stats import sigma_clip

		sigma=2.0

		zparr = zptbl[refmagkey]-zptbl[inmagkey]
		zperrarr = tool.sqsum(zptbl[inmagerrkey], zptbl[refmagerkey])

		zparr_clipped = sigma_clip(
			zparr,
			sigma=sigma,
			maxiters=None,
			cenfunc=np.median,
			copy=False
			)

		indx_alive = np.where( zparr_clipped.mask == False )
		indx_exile = np.where( zparr_clipped.mask == True )

		#	RE-DEF. ZP LIST AND INDEXING CLIPPED & NON-CLIPPED
		zptbl_alive = zptbl[indx_alive]
		zptbl_exile = zptbl[indx_exile]

		zp, zperr = compute_median_mad(zparr[indx_alive])

		print(f"{inmagkey} ZP: {zp:.3f}+/-{zperr:.3f}")

		#	Apply ZP
		##	MAG
		_calmagkey = f"{inmagkey}_{hdr['FILTER']}"
		_calmagerrkey = f"{inmagerrkey}_{hdr['FILTER']}"
		##	FLUX
		_calfluxkey = _calmagkey.replace('MAG', 'FLUX')
		_calfluxerrkey = _calmagerrkey.replace('MAG', 'FLUX')

		setbl[_calmagkey] = setbl[inmagkey]+zp
		setbl[_calmagerrkey] = tool.sqsum(setbl[inmagerrkey], zperr)

		#	Flux [uJy]
		setbl[_calfluxkey] = (setbl[_calmagkey].data*u.ABmag).to(u.uJy).value
		setbl[_calfluxerrkey] = setbl[_calfluxkey] * (10**(-0.4 * setbl[inmagerrkey]) - 1)

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


		#	Header keyword
		if inmagkey == 'MAG_AUTO':
			_zpkey = inmagkey.replace('MAG', 'ZP')
			_zperrkey = inmagerrkey.replace('MAGERR', 'EZP')
			_ul3key = inmagkey.replace('MAG', 'UL3')
			_ul5key = inmagkey.replace('MAG', 'UL5')
		elif inmagkey == 'MAG_APER':
			_zpkey = inmagkey.replace('MAG', 'ZP').replace('APER', '0')
			_zperrkey = inmagerrkey.replace('MAGERR', 'EZP').replace('APER', '0')
			_ul3key = inmagkey.replace('MAG', 'UL3').replace('APER', '0')
			_ul5key = inmagkey.replace('MAG', 'UL5').replace('APER', '0')
		else:
			_zpkey = inmagkey.replace('MAG', 'ZP').replace('APER_', '')
			_zperrkey = inmagerrkey.replace('MAGERR', 'EZP').replace('APER_', '')
			_ul3key = inmagkey.replace('MAG', 'UL3').replace('APER_', '')
			_ul5key = inmagkey.replace('MAG', 'UL5').replace('APER_', '')


		_zp_dict = {
			_zpkey: (round(zp, 3), f'ZERO POINT for {inmagkey}'),
			_zperrkey: (round(zperr, 3), f'ZERO POINT ERROR for {inmagkey}'),
			_ul3key: (round(ul_3sig, 3), f'3 SIGMA LIMITING MAG FOR {inmagkey}'),
			_ul5key: (round(ul_5sig, 3), f'5 SIGMA LIMITING MAG FOR {inmagkey}'),
		}

		# _zp_dict

		header_to_add.update(_zp_dict)

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

path_refcat	= f'/data/large_data_proton/factory/ref_cat'
path_config = '/home/gp/gppy/config'
path_to_filterset = f"{path_config}/filterset"
path_obs = f'{path_config}/obs.dat'
# path_target = './transient.dat'
path_gphot = f'{path_base}/gphot.config'
path_default_gphot = f'{path_config}/gphot.config'
#------------------------------------------------------------
print(path_gphot)
if os.path.exists(path_gphot) == True:
	gphot_dict = file2dict(path_gphot)
else:
	gphot_dict = file2dict(path_default_gphot)
	print('There is no gregoryphot configuration. Use default.')
#------------------------------------------------------------
imkey = gphot_dict['imkey']
refqueryradius = float(gphot_dict['refqueryradius'])# *u.degree
frac = float(gphot_dict['photfraction'])
refcatname = gphot_dict['refcatname']
refmaglower = float(gphot_dict['refmaglower'])
refmagupper = float(gphot_dict['refmagupper'])
refmagerupper = float(gphot_dict['refmagerupper'])
inmagerupper = float(gphot_dict['inmagerupper'])
flagcut = int(gphot_dict['flagcut'])
check = (gphot_dict['check'] == 'True')

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
imlist = sorted(glob.glob(imkey))
# ncore = 8
# ncore = 4
try:
	ncore = int(sys.argv[2])
except:
	ncore = 1
print(f'#\t{len(imlist)} images to do photometry')
print('='*60)
for i, img in enumerate(imlist):
	print(f'{i:0>4} {img}')
print('='*60)
#------------------------------------------------------------
fail_image_list = []
for ii, inim in enumerate(imlist):
	try:
		phot_routine(inim)
	except Exception as e:
		print(f"\nPhotometry for {os.path.basename(inim)} was failed!\n")
		print(f"Error:\n{e}")
		fail_image_list.append(inim)
#------------------------------------------------------------
#	Logging the Failed Images
#------------------------------------------------------------
if len(fail_image_list) > 0:
	if f"{os.path.dirname(fail_image_list[0])}"!='':
		f = open(f"{os.path.dirname(fail_image_list[0])}/phot.fail.list", 'w')
	else:
		f = open(f"./phot.fail.list", 'w')
	for finim in fail_image_list:
		f.write(f"{os.path.basename(finim)}\n")
	f.close()
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
