#!/home/paek/anaconda3/bin/python3.7
#============================================================
#   Transient Finder
#	
#	21.05.27	Created by Gregory S.H. Paek
#============================================================
import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import time
sys.path.append('..')
sys.path.append('/home/gp/gppy')
from phot import gpphot
from util import query
from util import tool
from phot import gcurve
# from datetime import date
from astropy.visualization.wcsaxes import SphericalCircle
import warnings
from astroquery.imcce import Skybot
warnings.filterwarnings("ignore")
# from itertools import product
from itertools import repeat
import multiprocessing
from matplotlib.patches import Circle, PathPatch
from matplotlib.patches import Rectangle
from astropy.visualization import SqrtStretch, LinearStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import ZScaleInterval, MinMaxInterval
#------------------------------------------------------------
#	Function
#------------------------------------------------------------
def inverse(sciim, outim):
	data, hdr = fits.getdata(sciim, header=True)
	# w = WCS(sciim)
	invdata = data*(-1)
	fits.writeto(outim, invdata, header=hdr, overwrite=True)
#------------------------------------------------------------
def routine_se(sciim, outcat, aperpix, seeing, conf_sex, conf_param, conf_conv, conf_nnw):
	param_insex = dict(
						#------------------------------
						#	CATALOG
						#------------------------------
						CATALOG_NAME = outcat,
						#------------------------------
						#	CONFIG FILES
						#------------------------------
						CONF_NAME = conf_sex,
						PARAMETERS_NAME = conf_param,
						FILTER_NAME = conf_conv,    
						STARNNW_NAME = conf_nnw,
						#------------------------------
						#	PHOTOMETRY
						#------------------------------
						#	DIAMETER
						#	OPT.APER, (SEEING x2), x3, x4, x5
						#	MAG_APER	OPT.APER
						#	MAG_APER_1	OPT.GAUSSIAN.APER
						#	MAG_APER_2	SEEINGx2
						#	...
						PHOT_APERTURES = str(aperpix),
						SATUR_LEVEL  = '65000.0',
						# GAIN = str(gain.value),
						# PIXEL_SCALE = str(pixscale.value),
						#------------------------------
						#	STAR/GALAXY SEPARATION
						#------------------------------
						SEEING_FWHM = str(seeing),
						)
	com = gpphot.sexcom(sciim, param_insex)
	print(com)
	os.system(com)
#------------------------------------------------------------
def get_mad(data):
	return np.median(np.absolute(data - np.median(data, axis=0)), axis=0)
#------------------------------------------------------------
def plot_snapshot(data, wcs, outpng, save=True):
	plt.close('all')
	plt.rc('font', family='serif')
	fig = plt.figure(figsize=(1, 1))
	fig.set_size_inches(1. * data.shape[0] / data.shape[1], 1, forward = False)
	x = 720 / fig.dpi
	y = 720 / fig.dpi
	fig.set_figwidth(x)
	fig.set_figheight(y)
	#	No axes
	# ax = plt.subplot(projection=wcs)
	ax = plt.Axes(fig, [0., 0., 1., 1.])
	ax.set_axis_off()
	fig.add_axes(ax)

	from astropy.visualization.stretch import LinearStretch
	#	Sci
	data[np.isnan(data)] = 0.0
	transform = LinearStretch()+ZScaleInterval()
	bdata = transform(data)
	# pylab.subplot(131)
	ax.imshow(bdata, cmap="gray", origin="lower")

	#	Circle
	# circle = Circle(
	# 	(data.shape[0]/2., data.shape[1]/2.),
	# 	# 2*peeing,
	# 	10,
	# 	edgecolor='yellow',
	# 	lw=1,
	# 	facecolor=None,
	# 	fill=False
	# )
	# ax.add_patch(circle)
	rect_size = 10
	rectangle = Rectangle(
		(data.shape[1]/2 - rect_size/2, data.shape[0]/2 - rect_size/2),
		rect_size, rect_size,
		edgecolor='yellow',
		lw=1,
		facecolor='none',
		fill=False
	)
	ax.add_patch(rectangle)
	plt.tight_layout()

	#	RA, Dec direction
	ra0, dec0 = wcs.all_pix2world(0, 0, 1)
	ra1, dec1 = wcs.all_pix2world(data.shape[0], data.shape[1], 1)
	if ra0>ra1:
		pass
	elif ra0<ra1:
		ax.invert_xaxis()
	if dec0>dec1:
		ax.invert_yaxis()
	elif dec0<dec1:
		pass
	if save:
		plt.savefig(outpng, dpi=100)#, overwrite=True)
	else:
		pass
#------------------------------------------------------------
# def process_transient(candidate, cutsize=1.0):
# 	# 추출할 이미지 이름
# 	sciim, hcim, hdim = candidate['sciim'], candidate['hcim'], candidate['hdim']
# 	# 천체의 위치
# 	position = SkyCoord(candidate['ALPHA_J2000'], candidate['DELTA_J2000'], frame='icrs', unit='deg')
# 	#
# 	# outpng = candidate['outpng']
# 	# 이미지 처리
# 	for image, kind in zip([sciim, hcim, hdim], ['new', 'ref', 'sub']):
# 		hdu = fits.open(image)[0]
# 		wcs = WCS(hdu.header)
# 		size = u.Quantity((cutsize, cutsize), u.arcmin)  # cutsize 변경 가능
# 		cutout = Cutout2D(data=hdu.data, position=position, size=size, wcs=wcs,
# 							mode='partial', fill_value=np.median(hdu.data))
# 		hdu.data = cutout.data
# 		hdu.header.update(cutout.wcs.to_header())
# 		outim = f'{os.path.splitext(image)[0]}.{candidate["NUMBER"]}.{kind}{os.path.splitext(image)[1]}'
# 		outpng = f'{os.path.splitext(image)[0]}.{candidate["NUMBER"]}.{kind}.png'
# 		# 이미지 저장
# 		hdu.writeto(outim, overwrite=True)
# 		plot_snapshot(hdu.data, wcs, outpng, save=True)

def process_transient(candidate, cutsize=1.0):
	# 추출할 이미지 이름
	sciim, hcim, hdim = candidate['sciim'], candidate['hcim'], candidate['hdim']
	# 천체의 위치
	position = SkyCoord(candidate['ALPHA_J2000'], candidate['DELTA_J2000'], frame='icrs', unit='deg')

	for image, kind in zip([sciim, hcim, hdim], ['sci', 'ref', 'sub']):
		# 파일 존재 여부 및 검증
		if os.path.exists(image) and os.path.isfile(image):
			try:
				with fits.open(image, ignore_missing_simple=True) as hdul:
					hdu = hdul[0]
					wcs = WCS(hdu.header)
					size = u.Quantity((cutsize, cutsize), u.arcmin)
					cutout = Cutout2D(data=hdu.data, position=position, size=size, wcs=wcs,
										mode='partial', fill_value=np.median(hdu.data))
					hdu.data = cutout.data
					hdu.header.update(cutout.wcs.to_header())
					outim = f'{os.path.splitext(image)[0]}.{candidate["NUMBER"]}.{kind}{os.path.splitext(image)[1]}'
					outpng = f'{os.path.splitext(image)[0]}.{candidate["NUMBER"]}.{kind}.png'
					# 이미지 저장
					hdu.writeto(outim, overwrite=True)
					plot_snapshot(hdu.data, wcs, outpng, save=True)
			except Exception as e:
				print(f"Error processing {image}: {e}")
		else:
			print(f"File not found or not a file: {image}")
#------------------------------------------------------------
def findloc(sciim):
	path_table = "/home/paek/qsopy/tables"
	loctbl = ascii.read(f'{path_table}/obs.location.smnet.dat')
	#	code == 500 : default
	code = 500
	for i, obs in enumerate(loctbl['obs']):
		if obs in sciim:
			code = loctbl['Code'][i].item()
			break
	return code
#------------------------------------------------------------
def create_ds9_region_file(ra_array, dec_array, radius, filename="ds9_regions.reg"):
	"""
	RA, Dec 배열과 반경을 입력으로 받아 DS9 region 파일을 생성하는 함수.
	
	Parameters:
	- ra_array: RA 좌표 배열
	- dec_array: Dec 좌표 배열
	- radius: 원의 반경 (단위: arcsec)
	- filename: 생성될 DS9 region 파일의 이름
	"""
	# Region 파일 시작 부분에 필요한 헤더
	header = 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5'
	
	# 파일 쓰기 시작
	with open(filename, "w") as file:
		file.write(header + "\n")
		
		# 각 좌표에 대한 원 형태의 region 추가
		for ra, dec in zip(ra_array, dec_array):
			region_line = f"circle({ra},{dec},{radius}\")\n"
			file.write(region_line)

	print(f"DS9 region file '{filename}' has been created.")
#------------------------------------------------------------
#	Path & Configuration
#------------------------------------------------------------
path_config = '/home/gp/gppy/config'
conv, nnw, param, sex = 'transient.conv', 'transient.nnw', 'transient.param', 'transient.sex'
conf_sex = f'{path_config}/{sex}'
conf_param = f'{path_config}/{param}'
conf_nnw = f'{path_config}/{nnw}'
conf_conv = f'{path_config}/{conv}'
#------------------------------------------------------------
#	Setting
#------------------------------------------------------------
cutsize = 2.0
fovval = 1.5
print(f"{'='*60}\n#\tTransient Search Process\n{'-'*60}")
print('='*60)
#------------------------------------------------------------
#	Input	
#------------------------------------------------------------
#	Test data
# hdim = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/hdcalib_7DT03_T09614_20240423_020757_r_360.com.fits"
# hdcat = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/hdcalib_7DT03_T09614_20240423_020757_r_360.com.phot.cat"
# hcim = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/hccalib_7DT03_T09614_20240423_020757_r_360.com.fits"
# refim = "/large_data/factory/ref_frame/r/ref_PS1_T09614_00000000_000000_r_0.fits"
# scicat = "/large_data/processed_1x1_gain2750/T09614/7DT03/r/calib_7DT03_T09614_20240423_020757_r_360.com.phot.cat"
#	Images
sciim = sys.argv[1]
refim = sys.argv[2]
hcim = sys.argv[3]
hdim = sys.argv[4]
#	Catalog with a fixed suffix
scicat = sciim.replace("fits", "phot.cat")
refcat = refim.replace("fits", "phot.cat")
hdcat = hdim.replace("fits", "phot.cat")
hccat = hcim.replace("fits", "phot.cat")
#
path_save = os.path.dirname(hdim)
#
print(f"{'-'*60}\n#\tInput\n{'-'*60}")
print(f'hdim : {os.path.basename(hdim)}')	
print(f'hdcat : {os.path.basename(hdcat)}')	
print(f'hcim : {os.path.basename(hcim)}')	
print(f'scicat : {os.path.basename(scicat)}')	
print(f'fovval : {fovval}')	
print('-'*60)

#	Image information
# sciim = hdim.replace('hd', '')
part = hdim.split('-')
#	Header
data, hdr = fits.getdata(hdim, header=True)
w = WCS(hdim)
filte = hdr['FILTER']
magkey = f"MAG_APER_1_{filte}"
#	Positional information
xcent, ycent = data.shape
c_cent = w.pixel_to_world(xcent, ycent)
seeing = hdr['SEEING']
aperpix = hdr['APER']
epoch = Time(hdr['DATE-OBS'], format='isot')
#	Name
invhdim = hdim.replace('subt', 'inv_subt')
invhcim = hcim.replace('conv', 'inv_conv')
invhdcat = invhdim.replace('fits', 'cat')
invhccat = invhcim.replace('fits', 'cat')
#	Inverse images
inverse(hdim, invhdim)
inverse(hcim, invhcim)
#	SEtractor for inverse images
routine_se(invhdim, invhdcat, aperpix, seeing, conf_sex, conf_param, conf_conv, conf_nnw)
routine_se(invhcim, invhccat, aperpix, seeing, conf_sex, conf_param, conf_conv, conf_nnw)
# 카탈로그 로드 및 초기 변수 설정
hdtbl = Table.read(hdcat, format='ascii.tab')
if 'flag' in hdtbl.keys():
	hdtbl.remove_column('flag')
hdtbl['flag'] = False

hdtbl['sciim'] = sciim
hdtbl['hcim'] = hcim
hdtbl['hdim'] = hdim
hdtbl['seeing'] = seeing
hdtbl['ratio_seeing'] = hdtbl['FWHM_WORLD'] * 3600 / seeing
hdtbl['ratio_ellipticity'] = hdtbl['ELLIPTICITY'] / hdr['ELLIP']
hdtbl['ratio_elongation'] = hdtbl['ELONGATION'] / hdr['ELONG']

#	Coordinate
c_hd = SkyCoord(hdtbl['ALPHA_J2000'], hdtbl['DELTA_J2000'], unit=u.deg)

#------------------------------------------------------------
#	Flagging
#------------------------------------------------------------
#	Generate blank flag
# numbers = np.arange(0, 12+1, 1)  # flag 0-12
numbers = np.arange(0, 13+1, 1)  # flag 0-13
# for num in numbers:
#     hdtbl[f'flag_{num}'] = False

# 플래그 초기화 및 불리언 타입 강제 설정
for num in numbers:
	flag_key = f"flag_{num}"
	if flag_key in hdtbl.keys():
		hdtbl.remove_column(flag_key)
	hdtbl[flag_key] = False  # 초기값을 False로 설정
	# hdtbl[flag_key] = hdtbl[flag_key].astype(bool)  # 불리언 타입으로 강제 변환

result_table = Table()
result_table['flag'] = numbers
result_table['n'] = 0
result_table['ratio'] = 0.
#------------------------------------------------------------
#	flag 0
#------------------------------------------------------------
#	Skybot query
try:
	# https://www.minorplanetcenter.net/iau/lists/ObsCodes.html
	# X08 (ShAO Chile station, El Sauce)
	# code = findloc(sciim)
	code = "X08"
	fovval = 1.5  # 가정: FOV를 1.5도로 설정
	epoch = 2000.0  # 가정: epoch를 J2000.0으로 설정

	sbtbl = Skybot.cone_search(c_hd, fovval * u.deg, epoch, location=code)
	c_sb = SkyCoord(sbtbl['RA'], sbtbl['DEC'], unit='deg')
	sbtbl['sep'] = c_hd.separation(c_sb).to(u.arcmin)
	
	# Skybot 매칭
	indx_sb, sep_sb, _ = c_hd.match_to_catalog_sky(c_sb)
	hdtbl['flag_0'][(sep_sb.arcsec < 5)] = True

except Exception as e:
	print(f"No solar system object was found in the requested FOV ({fovval} deg), error: {e}")

#------------------------------------------------------------
#	flag 1
#------------------------------------------------------------
invhdtbl = ascii.read(invhdcat)
if len(invhdtbl)>0:
	#	Coordinate
	c_invhd = SkyCoord(invhdtbl['ALPHA_J2000'], invhdtbl['DELTA_J2000'], unit=u.deg)
	#	Matching with inverted images
	indx_invhd, sep_invhd, _ = c_hd.match_to_catalog_sky(c_invhd)
	hdtbl['flag_1'][
		(sep_invhd.arcsec<seeing)
		] = True
else:
	print('Inverted subtraction image has no source. ==> pass flag1')
	pass
#------------------------------------------------------------
#	flag 2
#------------------------------------------------------------
invhctbl = ascii.read(invhccat)
if len(invhctbl)>0:
	#	Coordinate
	c_invhc = SkyCoord(invhctbl['ALPHA_J2000'], invhctbl['DELTA_J2000'], unit=u.deg)
	#	Matching with inverted images
	indx_invhc, sep_invhc, _ = c_hd.match_to_catalog_sky(c_invhc)
	hdtbl['flag_2'][
		(sep_invhc.arcsec<seeing)
		] = True
else:
	print('Inverted reference image has no source. ==> pass flag2')
	pass
#------------------------------------------------------------
#	SEtractor criterion
#------------------------------------------------------------
#	flag 3
#------------------------------------------------------------
#	Sources @edge
frac = 0.99
hdtbl['flag_3'][
	((hdtbl['X_IMAGE']<xcent-xcent*frac) |
	(hdtbl['X_IMAGE']>xcent+xcent*frac) |
	(hdtbl['Y_IMAGE']<ycent-ycent*frac) |
	(hdtbl['Y_IMAGE']>ycent+ycent*frac))
	] = True
#------------------------------------------------------------
#	flag 4
#------------------------------------------------------------
#	More than 5 sigma signal
hdtbl['flag_4'][
	(hdtbl[magkey]>hdr['UL5_1'])
	# ((hdtbl['magerr_aper_1']>0.2) |
	# (hdtbl['mag_aper_1']>hdr['ul5_1']) |
	# (hdtbl['mag_aper_2']>hdr['ul5_1']) |
	# (hdtbl['mag_aper_3']>hdr['ul5_1']) |
	# (hdtbl['mag_aper_4']>hdr['ul5_1']) |
	# (hdtbl['mag_aper_5']>hdr['ul5_1']))
	] = True
#	Empirical criterion
#------------------------------------------------------------
#	flag 5
#------------------------------------------------------------
hdtbl['flag_5'][
	(hdtbl['ratio_ellipticity'] > 5)
	] = True
#------------------------------------------------------------
#	flag 6
#------------------------------------------------------------
hdtbl['flag_6'][
	(hdtbl['FLAGS'] > 4.0)
	] = True
#------------------------------------------------------------
#	flag 7
#------------------------------------------------------------
hdtbl['flag_7'][
	(hdtbl['ratio_seeing']>3.0) |
	(hdtbl['ratio_seeing']<0.5)
	] = True
#------------------------------------------------------------
#	flag 8
#------------------------------------------------------------
hdtbl['flag_8'][
	(hdtbl['BACKGROUND']<-50) |
	(hdtbl['BACKGROUND']>+50)
	] = True
#------------------------------------------------------------
#	flag 9
#------------------------------------------------------------
scitbl = ascii.read(scicat)
scitbl = scitbl[
	(scitbl['FLAGS']==0) &
	(scitbl['CLASS_STAR']>0.5)
]

aperdict = {
	'mag_aper':'SNR_curve',
	'mag_aper_1':'Best_Aperture',
	'mag_aper_2':'2seeing',
	'mag_aper_3':'3seeing',
	'mag_aper_4':'3arcsec',
	'mag_aper_5':'5arcsec',	
}
	
key0 = f'MAG_APER_3_{filte}'
key1 = f'MAG_APER_5_{filte}'

try:
	#	Sci. sources magnitude diff.
	indelm = scitbl[key0] - scitbl[key1]
	#	Subt. sources magnitude diff.
	hddelm = hdtbl[key0] - hdtbl[key1]
	hdtbl['del_mag'] = hddelm
	#	MED & MAD
	indelm_med = np.median(indelm)
	indelm_mad = get_mad(indelm)
	hdtbl['del_mag_med'] = indelm_med
	hdtbl['del_mag_mad'] = indelm_mad
	hdtbl['N_del_mag_mad'] = np.abs((hdtbl['del_mag']-hdtbl['del_mag_med'])/hdtbl['del_mag_mad'])
	#	out
	n = 10
	indx_out = np.where(
		(hddelm<indelm_med-indelm_mad*n) |
		(hddelm>indelm_med+indelm_mad*n)
		)
	hdtbl['flag_9'][indx_out] = True
except Exception as e:
	print(f"Error Occured:\n{e}")

#------------------------------------------------------------
#	flag 10+11
#------------------------------------------------------------
peeing = hdr['PEEING']
skysig = hdr['SKYSIG']

nbadlist = []
ratiobadlist = []
nnulllist = []

f = 0.3
for i in range(len(hdtbl)):
	tx, ty = hdtbl['X_IMAGE'][i], hdtbl['Y_IMAGE'][i]
	bkg = hdtbl['BACKGROUND'][i]
	#	Snapshot
	tsize = peeing
	y0, y1 = int(ty-tsize), int(ty+tsize)
	x0, x1 = int(tx-tsize), int(tx+tsize)
	cdata = data[y0:y1, x0:x1]
	# plt.close()
	# plt.imshow(cdata)
	crt = bkg - skysig
	cutline = cdata.size*f
	nbad = len(cdata[cdata<crt])
	try:
		ratiobad = nbad/cdata.size
	except:
		ratiobad = -99.0
	nnull = len(np.where(cdata == 1e-30)[0])
	#	Dipole
	if nbad > cutline:
		hdtbl['flag_10'][i] = True
	#	HOTPANTS Null value
	if nnull != 0:
		hdtbl['flag_11'][i] = True
	nbadlist.append(nbad)
	ratiobadlist.append(ratiobad)
	nnulllist.append(nnull)

hdtbl['n_bad'] = nbadlist
hdtbl['ratio_bad'] = ratiobadlist
hdtbl['n_null'] = nnulllist
#------------------------------------------------------------
#	flag 12
#------------------------------------------------------------
x, y = fits.getdata(refim).shape
w_ref = WCS(refim)
xim, yim = w_ref.world_to_pixel(c_hd)
hdtbl['x_refim'] = xim
hdtbl['y_refim'] = yim
indx_nosci = np.where(
	(xim < 0) | (xim > x) | (yim < 0) | (yim > y)
)
hdtbl['flag_12'][indx_nosci] = True
#------------------------------------------------------------
#	flag 13
#------------------------------------------------------------
hddata = fits.getdata(hdim)
trim_size = 5
zero_pixel_lower_limit = 0.
for ii, (x, y) in enumerate(zip(hdtbl['X_IMAGE'], hdtbl['Y_IMAGE'])):
	sub_data = hddata[int(y-trim_size):int(y+trim_size), int(x-trim_size):int(x+trim_size)]
	if len(sub_data[sub_data == 0]) > zero_pixel_lower_limit:
		hdtbl['flag_13'][ii] = True
	else:
		pass
#------------------------------------------------------------
#	Final flag
#------------------------------------------------------------
# 플래그 합산 및 최종 결과 계산
flag = np.zeros(len(hdtbl), dtype=bool)  # 불리언 배열로 초기화
for n in numbers:
	if not hdtbl[f'flag_{n}'].dtype == bool:
		raise ValueError(f"Flag {n} contains non-boolean values.")
	flag |= hdtbl[f'flag_{n}']  # OR 연산을 사용하여 플래그 적용
hdtbl['flag'] = flag

# 결과 출력
for n in numbers:
	tmptbl = hdtbl[hdtbl[f'flag_{n}']]
	print(f'flag=={n:>2} : {len(tmptbl):>6} [{len(tmptbl)/len(hdtbl):.1%}]')
	#	Update result_table
	result_table['n'][n] = len(tmptbl)
	result_table['ratio'][n] = len(tmptbl)/len(hdtbl)


outcat = hdcat.replace('phot.cat', 'transient.cat')

result_table_name = f"{path_save}/{os.path.basename(hdcat).replace('phot.cat', 'flag.summary.csv')}"
result_table.write(result_table_name, format='csv', overwrite=True)

hdtbl.write(outcat, format='ascii.tab', overwrite=True)
print('-' * 60)
bgstbl = hdtbl[hdtbl['flag']]
tctbl = hdtbl[~hdtbl['flag']]
print(f'Filtered sources\t: {len(bgstbl)} {len(bgstbl)/len(hdtbl):.1%}')
print(f'Transient Candidates\t: {len(tctbl)} {len(tctbl)/len(hdtbl):.1%}')
#	DS9 REGION
ds9reg_name = outcat.replace("cat", "reg")
create_ds9_region_file(tctbl['ALPHA_J2000'], tctbl['DELTA_J2000'], radius=2.5, filename=ds9reg_name)
#------------------------------------------------------------
#	Snapshot maker
#------------------------------------------------------------
nn = 0
t0 = time.time()
for nn, (number, ra, dec) in enumerate(zip(tctbl['NUMBER'], tctbl['ALPHA_J2000'], tctbl['DELTA_J2000'])):
	# print(f"[{nn:>6}] NUMBER={number} (ra,dec = {ra:.3f},{dec:.3f})")
	print(f"[{nn:>6}] NUMBER= {number}")
	candidate = {
		'NUMBER': number,
		'ALPHA_J2000': ra,
		'DELTA_J2000': dec,
		'sciim': sciim,
		'hcim': hcim,
		'hdim': hdim,
		# 'seeing': seeing
	}
	process_transient(candidate)
delt = time.time()
print(f"[  DONE  ] {delt/60:.1f} mins for {len(tctbl)} candidates")