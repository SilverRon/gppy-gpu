def run_pre_sextractor(inim, outcat, conf_simple, param_simple, conv_simple, nnw_simple, pixscale, verbose_sex=False):
    import os
    import sys
    from pathlib import Path

    # Path Setup for Custom Packages
    path_thisfile = Path(__file__).resolve()  # /gppy-gpu/src/refactor/preprocessing.py
    Path_root = path_thisfile.parents[2]  # gppu-gpu
    Path_src = Path_root / 'src'
    if Path_src not in map(Path, sys.path):
        sys.path.append(str(Path_src)) 
    from util.path_manager import log2tmp

    # outhead = inim.replace('fits', 'head')
    # outcat = inim.replace('fits', 'pre.cat')

    #	Pre-Source EXtractor
    sexcom = f"source-extractor -c {conf_simple} {inim} -CATALOG_NAME {outcat} -CATALOG_TYPE FITS_LDAC -PARAMETERS_NAME {param_simple} -FILTER_NAME {conv_simple} -STARNNW_NAME {nnw_simple} -PIXEL_SCALE {pixscale}"
    print(sexcom)
    # os.system(sexcom)
    if verbose_sex:
        os.system(sexcom)
    else:
        # Redirect SE output to a tmp log
        os.system(log2tmp(sexcom, "presex"))

def astrom(path_data, ic_fdzobj, obsinfo, n_binning, path_config, memory_threshold,
           ncore, fdzimlist, timetbl, objtbl, verbose_sex, objarr, local_astref,
           tile_name_pattern, path_ref_scamp, upaths, debug, obs):
    import os
    import sys
    import re
    import time
    import glob
    import psutil
    from pathlib import Path
    import subprocess
    from concurrent.futures import ProcessPoolExecutor
    from astropy.io import fits

    # Path Setup for Custom Packages
    path_thisfile = Path(__file__).resolve()  # /gppy-gpu/src/refactor/preprocessing.py
    Path_root = path_thisfile.parents[2]  # gppu-gpu
    Path_src = Path_root / 'src'
    if Path_src not in map(Path, sys.path):
        sys.path.append(str(Path_src)) 
    from preprocess import calib
    from util import tool

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
    t0_astrometry_solve_field = time.time()
    #------------------------------------------------------------

    fnamelist = [f"{path_data}/{_fname}" for _fname in ic_fdzobj.summary['file']]
    pixscale = obsinfo['pixscale']*n_binning
    objectlist = [pixscale]*len(fnamelist)
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

    def modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple, pixscale):

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
        #	Pixel Scale
        pattern_pixscale_to_find = 'PIXEL_SCALE      0.51         # size of pixel in arcsec (0=use FITS WCS info)'
        pattern_pixscale_to_replace = f'PIXEL_SCALE      {pixscale}         # size of pixel in arcsec (0=use FITS WCS info)'


        pattern_to_find_list = [
            pattern_cat_to_find,
            # pattern_param_to_find,
            pattern_conv_to_find,
            pattern_nnw_to_find,
            pattern_pixscale_to_find,
        ]

        pattern_to_replace_list = [
            pattern_cat_to_replace,
            # pattern_param_to_replace,
            pattern_conv_to_replace,
            pattern_nnw_to_replace,
            pattern_pixscale_to_replace,
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
        modify_sex_config(_precat, _outcfg, conf_simple, param_simple, nnw_simple, conv_simple, pixscale)

    st_ = time.time()
    #	Move to the path_data
    original_directory = os.getcwd()
    os.chdir(path_data)
    #	Copy default.sex (High DETECT_THRESH)
    cpcom_default_cfg = f"cp {path_config}/default.sex {path_data}"
    print(cpcom_default_cfg)
    os.system(cpcom_default_cfg)

    #	Astrometry
    while psutil.virtual_memory().percent > memory_threshold:
        print(f"Memory Usage is above {memory_threshold}% ({psutil.virtual_memory().percent}%) - Start the Astrometry!!!")
        time.sleep(10)

    print(f"Memory Usage is below {memory_threshold}% - Start the Astrometry!!!")
    with ProcessPoolExecutor(max_workers=ncore) as executor:
        # results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, cfglist, _))
        results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, _, _))

    # WCS 계산 실패 체크
    failed_wcs_images = []
    for fname, result in zip(fnamelist, results):
        solved_file = fname.replace('.fits', '.solved')  # solve-field가 생성하는 solved 파일
        if not os.path.exists(solved_file):  # solved 파일이 없으면 실패로 간주
            failed_wcs_images.append(fname)

    # 실패 이미지 처리
    if failed_wcs_images:
        fail_wcs_log = open(f"{path_data}/solve_field_fail.txt", 'w')
        print(f"Astrometry failed for {len(failed_wcs_images)} images:")
        for img in failed_wcs_images:
            print(f"  - {img}")
            fail_wcs_log.write(f"{img}\n")
        fail_wcs_log.close()
    else:
        print("All images passed the astrometry step successfully.")



    delt = time.time() - st_
    #	Move back to the original path
    os.chdir(original_directory)
    print(f"Astrometry was done: {delt:.3f} sec/{len(fdzimlist)}")


    #------------------------------------------------------------
    astrometry_suffix_list = ['axy', 'corr', 'xyls', 'match', 'rdls', 'solved', 'wcs']
    for suffix in astrometry_suffix_list:
        rmcom = f"rm {path_data}/*.{suffix}"
        print(rmcom)

    delt_astrometry_solve_field = time.time() - t0_astrometry_solve_field
    timetbl['status'][timetbl['process']=='astrometry_solve_field'] = True
    timetbl['time'][timetbl['process']=='astrometry_solve_field'] = delt_astrometry_solve_field
    # timetbl['status'][timetbl['process']=='astrometry'] = True
    # timetbl['time'][timetbl['process']=='astrometry'] = int(time.time() - st_)

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

    outcatlist = []
    outheadlist = []

    # for inim in calimlist:
    for inim in afdzimlist:
        outcat = inim.replace('fits', 'cat')
        outhead = inim.replace('fits', 'head')

        outcatlist.append(outcat)
        outheadlist.append(outhead)

    t0_pre_source_extractor = time.time()

    #	Pre-Source EXtractor
    st_ = time.time()
    with ProcessPoolExecutor(max_workers=ncore) as executor:
        # results = list(executor.map(run_pre_sextractor, calimlist, outcatlist, [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist)))
        results = list(executor.map(run_pre_sextractor, afdzimlist, outcatlist, [conf_simple]*len(outcatlist), [param_simple]*len(outcatlist), [conv_simple]*len(outcatlist), [nnw_simple]*len(outcatlist), [pixscale]*len(outcatlist), [verbose_sex]*len(outcatlist)))
    delt = time.time() - st_
    # print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(calimlist)} (ncroe={ncore})")
    print(f"Pre-SExtractor Done: {delt:.3f} sec/{len(afdzimlist)} (ncroe={ncore})")

    delt_pre_source_extractor = time.time() - t0_pre_source_extractor
    timetbl['status'][timetbl['process']=='pre_source_extractor'] = True
    timetbl['time'][timetbl['process']=='pre_source_extractor'] = delt_pre_source_extractor

    #
    t0_astrometry_scamp = time.time()

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
    # scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list}"
    # scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} -AHEADER_GLOBAL {path_config}/{obs.lower()}.ahead"
    # print(scampcom)
    # os.system(scampcom)

    #	SCAMP (input CATALOG)
    print(f"= = = = = = = = = = = = Astrometric Correction = = = = = = = = = = = =")
    for oo, obj in enumerate(objarr):
        print(f"[{oo+1}/{len(objarr)}] {obj}")

        path_cat_scamp_list = f"{path_data}/{obj}.cat.scamp.list"
        s = open(path_cat_scamp_list, 'w')
        obj_outcatlist = [incat for incat in outcatlist if obj in os.path.basename(incat)]
        for incat in obj_outcatlist:
            s.write(f"{incat}\n")
        s.close()

        if local_astref and (re.match(tile_name_pattern, obj)) and (obj not in ['T04231', 'T04409', 'T04590']):
            astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
            if debug:
                print('='*79)
                print('astrefcat', astrefcat)
                print('='*79)
            scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
        else:
            scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"

        #	Run
        # scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} {scamp_addcom}"
        # print(scampcom)
        # os.system(scampcom)

        # Run with subprocess
        path_scampconfig = f"{path_config}/7dt.scamp" if 'path_scampconfig' not in upaths or upaths['path_scampconfig'] == '' else upaths['path_scampconfig']
        scampcom = ["scamp", "-c", path_scampconfig, f"@{path_cat_scamp_list}"]
        scampcom += scamp_addcom.split()
        print(" ".join(scampcom))  # Join the command list for printing

        try:
            result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # 명령어 실행 결과 출력
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error code {e.returncode}")
            print(f"stderr output: {e.stderr.decode()}")


    #	Rename afdz*.head --> fdz*.head
    for inhead in outheadlist: os.rename(inhead, inhead.replace('afdz', 'fdz'))

    delt_astrometry_scamp = time.time() - t0_astrometry_scamp
    timetbl['status'][timetbl['process']=='astrometry_scamp'] = True
    timetbl['time'][timetbl['process']=='astrometry_scamp'] = delt_astrometry_scamp

    #	MissFits
    t0_missfits = time.time()

    ##	Single-Thread MissFits Compile
    # missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list}"
    missfitscom = f"missfits -c {path_config}/7dt.missfits @{path_image_missfits_list}"
    ##	Multi-Threads MissFits Compile
    # missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list} -NTHREADS {ncore}"
    print(missfitscom)
    os.system(missfitscom)

    delt_missfits = time.time() - t0_missfits
    timetbl['status'][timetbl['process']=='missfits'] = True
    timetbl['time'][timetbl['process']=='missfits'] = delt_missfits

    #	Rename fdz*.fits (scamp astrometry) --> calib*.fits
    for inim in fdzimlist:
        if inim not in failed_wcs_images:
            calib.fnamechange(inim, obs)
    calimlist = sorted(glob.glob(f"{path_data}/calib*.fits"))

    for inim , _inhead in zip(calimlist, outheadlist):
        inhead = _inhead.replace('afdz', 'fdz')
        outhead = inim.replace('fits', 'head').replace('head', 'head.bkg')
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



    #	Update Coordinate on the Image
    #------------------------------------------------------------
    ##	TAN --> TPV Projection
    ##	Center RA & Dec
    ##	RA, Dec Polygons
    ##	Rotation angle
    #------------------------------------------------------------
    print(f"Update Center & Polygon Info ...")
    t0_get_polygon_info = time.time()

    #	Correct CTYPE (TAN --> TPV)
    for inim in calimlist:
        with fits.open(inim, mode='update') as hdul:
            # 헤더 데이터 불러오기
            hdr = hdul[0].header

            # 헤더 정보 변경 또는 추가
            hdr['CTYPE1'] = ('RA---TPV', 'WCS projection type for this axis')
            hdr['CTYPE2'] = ('DEC--TPV', 'WCS projection type for this axis')
            # 변경된 내용 저장
            hdul.flush()

    # t0_wcs = time.time()
    for cc, calim in enumerate(calimlist):
        # Extract WCS information (center, CD matrix)
        center, vertices, cd_matrixs = tool.get_wcs_coordinates(calim)
        cd1_1, cd1_2, cd2_1, cd2_2 = cd_matrixs

        # updates = [
        # 	("CTYPE1", 'RA---TPV', 'WCS projection type for this axis'),
        # 	("CTYPE2", 'DEC--TPV', 'WCS projection type for this axis')
        # ]
        # Define header list to udpate
        updates = [
            ("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"),
            ("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]")
        ]

        # updates.append(("RACENT", round(center[0].item(), 3), "RA CENTER [deg]"))
        # updates.append(("DECCENT", round(center[1].item(), 3), "DEC CENTER [deg]"))

        # RA, Dec Polygons
        for ii, (_ra, _dec) in enumerate(vertices):
            updates.append((f"RAPOLY{ii}", round(_ra, 3), f"RA POLYGON {ii} [deg]"))
            updates.append((f"DEPOLY{ii}", round(_dec, 3), f"DEC POLYGON {ii} [deg]"))

        # Field Rotation
        try:
            if (cd1_1 != 0) and (cd1_2 != 0) and (cd2_1 != 0) and (cd2_2 != 0):
                rotation_angle_1, rotation_angle_2 = tool.calculate_field_rotation(cd1_1, cd1_2, cd2_1, cd2_2)
            else:
                rotation_angle_1, rotation_angle_2 = float('nan'), float('nan')
        except Exception as e:
            print(f'Error: {e}')
            print(f'Image: {calim}')
            rotation_angle_1, rotation_angle_2 = float('nan'), float('nan')

        # Update rotation angle
        updates.append(('ROTANG1', rotation_angle_1, 'Rotation angle from North [deg]'))
        updates.append(('ROTANG2', rotation_angle_2, 'Rotation angle from East [deg]'))

        # FITS header update
        with fits.open(calim, mode='update') as hdul:
            for key, value, comment in updates:
                hdul[0].header[key] = (value, comment)
            hdul.flush()  # 변경 사항을 디스크에 저장

    delt_get_polygon_info = time.time() - t0_get_polygon_info
    timetbl['status'][timetbl['process']=='get_polygon_info'] = True
    timetbl['time'][timetbl['process']=='get_polygon_info'] = delt_get_polygon_info

    return hdr