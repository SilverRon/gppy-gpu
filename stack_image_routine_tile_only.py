# %%
import os
import glob
import time
import numpy as np
from astropy.io import fits
from astropy.time import Time
from ccdproc import ImageFileCollection
from astropy.table import Table
from datetime import datetime, timezone, timedelta
from astropy.coordinates import SkyCoord
from astropy import units as u
#
from preprocess import calib
from util import tool
# %%
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

# %%
path_tmp = "/large_data/factory/tmp"
path_config = './config'
path_skygrid = './data/skygrid/7DT/displaycenter.txt'
skygrid_table = Table.read(path_skygrid, format='ascii')
skygrid_table['tile'] = [f"T{idname:0>5}" for idname in skygrid_table['id']]

tiles = ['T08415',
 'T08887',
 'T08888',
 'T08889',
 'T08890',
 'T08891',
 'T08892',
 'T08893',
 'T08894',
 'T08895',
 'T09123',
 'T09125',
 'T09126',
 'T09127',
 'T09128',
 'T09129',
 'T09130',
 'T09131',
 'T09132',
 'T09133',
 'T09134',
 'T09366',
 'T09367',
 'T09368',
 'T09369',
 'T09370',
 'T09371',
 'T09372',
 'T09373',
 'T09374',
 'T09609',
 'T09610',
 'T09611',
 'T09612',
 'T09613',
 'T09614',
 'T09615',
 'T09616',
 'T09617',
 'T09851',
 'T09853',
 'T09854',
 'T09855',
 'T09856',
 'T09857',
 'T09858',
 'T09859',
 'T09860',
 'T09861',
 'T09862',
 'T10100',
 'T10101',
 'T10102',
 'T10103',
 'T10104',
 'T10105',
 'T10106',
 'T10107',
 'T10108',
 'T10109',
 'T10346',
 'T10349',
 'T10350',
 'T10351',
 'T10352',
 'T10353',
 'T10354',
 'T10355',
 'T10356',
 'T10357',
 'T10599',
 'T10600',
 'T10601',
 'T10602',
 'T10603',
 'T10604',
 'T10605',
 'T10606',
 'T10607',
 'T10608',
 'T10851',
 'T10852',
 'T10853',
 'T10854',
 'T10855',
 'T10856',
 'T10857',
 'T10858',
 'T10859',
 'T10860',
 'T11105',
 'T11106',
 'T11107',
 'T11108',
 'T11109',
 'T11110',
 'T11111',
 'T11112',
 'T11113',
 'T11114',
 'T11361',
 'T11362',
 'T11363',
 'T11364',
 'T11365',
 'T11366',
 'T11367',
 'T11368',
 'T11369',
 'T11370',
 'T11618',
 'T11619',
 'T11620',
 'T11621',
 'T11622',
 'T11623',
 'T11624',
 'T11625',
 'T11626',
 'T11627',
 'T11874',
 'T11877',
 'T11878',
 'T11879',
 'T11880',
 'T11881',
 'T11882',
 'T11883',
 'T11884',
 'T11885',
 'T11886',
 'T12137',
 'T12138',
 'T12139',
 'T12140',
 'T12141',
 'T12142',
 'T12143',
 'T12144',
 'T12145',
 'T12146',
 'T12396',
 'T12399',
 'T12400',
 'T12401',
 'T12402',
 'T12403',
 'T12404',
 'T12405',
 'T12406',
 'T12407',
 'T12408',
 'T12662',
 'T12663',
 'T12664',
 'T12665',
 'T12666',
 'T12667',
 'T12668',
 'T12669',
 'T12670',
 'T12671',
 'T12927',
 'T12928',
 'T12929',
 'T12930',
 'T12931',
 'T12932',
 'T12933',
 'T12934',
 'T12935',
 'T12936',
 'T13726']


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

# %%
# wildcard = input(f"Wildcard (./calib*0.fits):")

for tt, tile in enumerate(tiles):
	print(f"[{tt:>4}] {tile}")
	wildcard = f"/large_data/processed_1x1_gain2750/{tile}/7DT*/r/calib*0.fits"
	images = sorted(glob.glob(f"{wildcard}"))
	path_data = os.path.dirname(images[0])

	obslist = np.unique([os.path.basename(inim).split("_")[1] for inim in images])
	if len(obslist) == 1:
		obs = obslist[0]
	else:
		print(f"Various OBS detected: {obslist}")
		obs = input(f"Pick One of them:")
	print(f"OBS: {obs}")

	#
	indx_skygrid = skygrid_table['tile'] == tile
	ra, dec = skygrid_table['ra'][indx_skygrid][0], skygrid_table['dec'][indx_skygrid][0]
	c = SkyCoord(ra, dec, unit=u.deg)
	rahms = c.ra.to_string(unit=u.hourangle, sep=':', pad=True)
	decdms = c.dec.to_string(unit=u.degree, sep=':', pad=True, alwayssign=True)
	radec = f"{rahms} {decdms}"

	print(f"{images} found")
	# break
	# %%
	ic_cal = ImageFileCollection(filenames=images, keywords='*')

	#	Time to group
	threshold = 300./(60*60*24) # [MJD]
	t_group = 0.5/24 # 30 min

	# %%
	grouplist = []
	for obj in np.unique(ic_cal.summary['object']):
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
						swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -RESAMPLE_DIR {path_tmp} -CENTER_TYPE MANUAL -CENTER {center} -GAIN_KEYWORD EGAIN"
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
						gain_default = fits.getheader(baseim)['EGAIN']
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

						delt_com = time.time() - t0_com
						print(f"Combied Time: {delt_com:.3f} sec")

# delt_image_stack = time.time() - t0_image_stack