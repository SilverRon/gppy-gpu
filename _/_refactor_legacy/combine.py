def stack(path_config, path_data, tile_name_pattern, skygrid_table, obs, hdr, timetbl):
    import os
    import sys
    import re
    import time
    from datetime import datetime
    from pathlib import Path
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.table import Table, hstack
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy import units as u
    from ccdproc import ImageFileCollection

    # Path Setup for Custom Packages
    path_thisfile = Path(__file__).resolve()  # /gppy-gpu/src/refactor/preprocessing.py
    Path_root = path_thisfile.parents[2]  # gppu-gpu
    Path_src = Path_root / 'src'
    if Path_src not in map(Path, sys.path):
        sys.path.append(str(Path_src)) 
    from preprocess import calib
    from util import tool

    #------------------------------------------------------------

    #	IMAGE COMBINE

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
        """Unused for now"""
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
    

    return timetbl, stacked_images