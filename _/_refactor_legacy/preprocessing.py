def preproc(
    obs,
    path_data,
    path_mframe,
    path_log,
    verbose_gpu,
    path_new,
    timetbl,
    biasnumb,
    bimlist,
    ic1,
    darknumb,
    darkexptimelist,
    nobj,
    flatnumb,
    filterlist,
    start_localtime,
    objarr,
):
    import os, sys
    import time
    import gc
    import glob
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    from ccdproc import ImageFileCollection

    # ------------------------------------------------------------
    # 	GPU Library
    # ------------------------------------------------------------
    from eclaire import FitsContainer, reduction, fixpix, imalign, imcombine
    import cupy as cp

    # Path Setup for Custom Packages
    path_thisfile = Path(__file__).resolve()  # /gppy-gpu/src/refactor/preprocessing.py
    Path_root = path_thisfile.parents[2]  # gppu-gpu
    Path_src = Path_root / "src"

    if Path_src not in map(Path, sys.path):
        sys.path.append(str(Path_src))
    from preprocess import calib
    from util import tool

    # ============================================================

    t00 = time.time()

    # 	Pick GPU unit
    odd_unit_obs = [
        "7DT01",
        "7DT03",
        "7DT05",
        "7DT07",
        "7DT09",
        "7DT11",
        "7DT13",
        "7DT15",
        "7DT17",
        "7DT19",
    ]
    even_unit_obs = [
        "7DT02",
        "7DT04",
        "7DT06",
        "7DT08",
        "7DT10",
        "7DT12",
        "7DT14",
        "7DT16",
        "7DT18",
        "7DT20",
    ]

    print(f"=" * 60)
    print()
    if obs in odd_unit_obs:
        cp.cuda.Device(0).use()
        print(f"Use the first GPU Unit ({obs})")
    elif obs in even_unit_obs:
        cp.cuda.Device(1).use()
        print(f"Use the second GPU Unit ({obs})")

    mempool = cp.get_default_memory_pool()
    if verbose_gpu:
        print(f"Default GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

    print()
    print(f"=" * 60)
    # ------------------------------------------------------------
    # 	BIAS
    # ------------------------------------------------------------
    print(
        """#------------------------------------------------------------
#	Bias
#-----------------------------------------------------------
    """
    )
    t0_bias = time.time()

    if biasnumb != 0:
        #   Stacking with GPU
        bfc = FitsContainer(bimlist)
        if verbose_gpu:
            print(
                f"Bias fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
            )

        boutput = f"{path_data}/zero.fits"

        # t0 = time.time()
        mbias = imcombine(
            bfc.data,
            name=boutput,
            list=bimlist,
            overwrite=True,
            combine="median",  # specify the co-adding method
            # width=3.0 # specify the clipping width
            # iters=5 # specify the number of iterations
        )
        # delt = time.time() - t0
        fits.writeto(
            f"{path_data}/zero.fits",
            data=cp.asnumpy(mbias),
            header=fits.getheader(bimlist[0]),
            overwrite=True,
        )

        if verbose_gpu:
            print(
                f"bias combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
            )

        dateobs_mzero = tool.calculate_average_date_obs(
            ic1.filter(imagetyp="bias").summary["date-obs"]
        )

        date = dateobs_mzero[:10].replace("-", "")
        zeroim = f"{path_mframe}/{obs}/zero/{date}-zero.fits"
        if not os.path.exists(os.path.dirname(zeroim)):
            os.makedirs(os.path.dirname(zeroim))
        cpcom = f"cp {path_data}/zero.fits {zeroim}"
        print(cpcom)
        os.system(cpcom)
        plt.close("all")

        # 	Clear the momory pool
        del bfc
        mempool.free_all_blocks()
        if verbose_gpu:
            print(
                f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
            )
        timetbl["status"][timetbl["process"] == "master_frame_bias"] = True

    else:
        # 	IF THERE IS NO FLAT FRAMES, BORROW FROM CLOSEST OTHER DATE
        print("\nNO BIAS FRAMES\n")
        pastzero = np.array(glob.glob(f"{path_mframe}/{obs}/zero/*zero.fits"))

        # 	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
        deltime = []

        # 	Zero
        _zeromjd = Time(ic1.summary["date-obs"][0], format="isot").mjd
        for date in pastzero:
            pastzeromjd = calib.isot_to_mjd((os.path.basename(date)).split("-")[0])
            deltime.append(np.abs(_zeromjd - pastzeromjd))
        indx_closest = np.where(deltime == np.min(deltime))
        tmpzero = pastzero[indx_closest][0]
        print(f"Borrow {tmpzero}")
        mbias = cp.asarray(fits.getdata(tmpzero), dtype="float32")

    delt_bias = time.time() - t0_bias
    print(f"Bias Master Frame: {delt_bias:.3f} sec")
    timetbl["time"][timetbl["process"] == "master_frame_bias"] = delt_bias

    # ------------------------------------------------------------
    ##	Dark
    # ------------------------------------------------------------
    print(
        """#------------------------------------------------------------
#	Dark
#------------------------------------------------------------
    """
    )
    t0_dark = time.time()

    darkdict = dict()
    # if (darknumb > 0) & ((nobj > 0) | (flatnumb > 0)):
    if darknumb > 0:
        dark_process = True
        for i, exptime in enumerate(darkexptimelist):
            print(
                f"PRE PROCESS FOR DARK ({exptime} sec)\t[{i+1}/{len(darkexptimelist)}]"
            )
            dimlist = list(ic1.filter(imagetyp="Dark", exptime=exptime).summary["file"])
            print(f"{len(dimlist)} Dark Frames Found")
            dfc = FitsContainer(dimlist)

            if verbose_gpu:
                print(
                    f"Dark fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )

            doutput = f"{path_data}/dark-{int(exptime)}.fits"

            # t0 = time.time()
            mdark = (
                imcombine(
                    # dfc.data, list=dimlist, overwrite=True,
                    dfc.data,
                    name=doutput,
                    list=dimlist,
                    overwrite=True,
                    combine="median",  # specify the co-adding method
                    # width=3.0 # specify the clipping width
                    # iters=5 # specify the number of iterations
                )
                - mbias
            )

            # 	Apply BIAS Image on the Dark Image
            # 기존 FITS 파일 읽기
            with fits.open(doutput) as hdul:
                # 데이터 섹션 가져오기
                data = hdul[0].data

                # 데이터 수정하기 (예: 모든 픽셀 값을 2배로 만들기)
                new_data = cp.asnumpy(cp.array(data) - mbias)

                # 새로운 HDU 생성
                new_hdu = fits.PrimaryHDU(new_data)

                # 기존 헤더 정보 복사
                new_hdu.header = hdul[0].header

                # 새로운 HDU 리스트 생성
                new_hdul = fits.HDUList([new_hdu])

                # 수정된 데이터를 새로운 FITS 파일로 저장
                new_hdul.writeto(doutput, overwrite=True)

            # delt = time.time() - t0

            # t_np = time.time()
            # data - mbias_np
            # print(time.time()-t_np)
            # 0.19986248016357422

            # t_cp = time.time()
            # mdark_cp - mbias
            # print(time.time()-t_cp)
            # 0.0005612373352050781

            if verbose_gpu:
                print(
                    f"dark combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )

            # 	Clear the momory pool
            del dfc
            mempool.free_all_blocks()
            gc.collect()
            if verbose_gpu:
                print(
                    f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )
            darkdict[str(int(exptime))] = mdark

            dateobs_mdark = tool.calculate_average_date_obs(
                ic1.filter(imagetyp="dark").summary["date-obs"]
            )
            date = dateobs_mdark[:10].replace("-", "")

            darkim = f"{path_mframe}/{obs}/dark/{int(exptime)}-{date}-dark.fits"

            if not os.path.exists(os.path.dirname(darkim)):
                os.makedirs(os.path.dirname(darkim))

            cpcom = f"cp {path_data}/dark-{int(exptime)}.fits {darkim}"
            print(cpcom)
            os.system(cpcom)
            plt.close("all")
            del mdark
            mempool.free_all_blocks()
            gc.collect()

        timetbl["status"][timetbl["process"] == "master_frame_dark"] = True
    else:
        # 	Borrow
        print("\nNO DARK FRAMES\n")
        if nobj > 0:
            objexptimelist = list(set(ic1.filter(imagetyp="Light").summary["exptime"]))
            exptime = np.max(objexptimelist)
            pastdark = np.array(glob.glob(f"{path_mframe}/{obs}/dark/*-dark.fits"))

        elif flatnumb > 0:
            objexptimelist = list(set(ic1.filter(imagetyp="Flat").summary["exptime"]))
            exptime = np.max(objexptimelist)
            pastdark = np.array(glob.glob(f"{path_mframe}/{obs}/dark/*-dark.fits"))

        # 	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
        deltime = []
        delexptime = []
        darkexptimes = []
        _darkmjd = Time(ic1.summary["date-obs"][0], format="isot").mjd
        for date in pastdark:
            darkmjd = calib.isot_to_mjd((os.path.basename(date)).split("-")[1])
            darkexptime = int(os.path.basename(date).split("-")[0])
            darkexptimes.append(darkexptime)
            deltime.append(np.abs(_darkmjd - darkmjd))

        deldarkexptime_arr = np.abs(np.array(darkexptimes) - exptime)

        indx_closet = np.where(
            (deltime == np.min(deltime))
            &
            # (darkexptimes == np.max(darkexptimes))
            (deldarkexptime_arr == np.min(deldarkexptime_arr))
        )
        if len(indx_closet[0]) == 0:
            indx_closet = np.where((deltime == np.min(deltime)))
        else:
            pass

        tmpdark = pastdark[indx_closet][-1]
        # exptime = int(fits.getheader(tmpdark)['exptime'])
        exptime = int(np.array(darkexptimes)[indx_closet])

        mdark = cp.asarray(fits.getdata(tmpdark), dtype="float32")
        darkdict[f"{exptime}"] = mdark
        del mdark

    delt_dark = time.time() - t0_dark
    print(f"Dark Master Frame: {delt_dark:.3f} sec")
    timetbl["time"][timetbl["process"] == "master_frame_dark"] = delt_dark

    darkexptimearr = np.array([float(val) for val in list(darkdict.keys())])
    # ------------------------------------------------------------
    # 	Flat
    # ------------------------------------------------------------
    print(
        """#------------------------------------------------------------
#	Flat
#------------------------------------------------------------
    """
    )

    t0_flat = time.time()

    #
    flatdict = dict()
    if flatnumb > 0:

        # 	master flat dictionary
        for filte in filterlist:
            print(f"- {filte}-band")
            fimlist = []

            flat_raw_imlist = list(
                ic1.filter(imagetyp="FLAT", filter=filte).summary["file"]
            )
            flat_raw_exptarr = cp.array(
                ic1.filter(imagetyp="FLAT", filter=filte).summary["exptime"].data.data
            )[:, None, None]
            _ffc = FitsContainer(flat_raw_imlist)

            _exptarr = np.array([int(expt) for expt in list(darkdict.keys())])

            closest_dark_exptime = np.min(_exptarr)
            exptime_scale_arr = flat_raw_exptarr / closest_dark_exptime

            # 	Bias Correction
            _ffc.data -= mbias
            # 	Dark Correction
            _ffc.data -= darkdict[str(int(closest_dark_exptime))] * exptime_scale_arr

            # 	Normalization
            _ffc.data /= cp.median(_ffc.data, axis=(1, 2), keepdims=True)

            if verbose_gpu:
                print(
                    f"Flat fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )

            # 	Generate Master Flat
            foutput = f"{path_data}/n{filte}.fits"
            mflat = imcombine(
                _ffc.data,
                name=foutput,
                list=flat_raw_imlist,
                overwrite=True,
                combine="median",  # specify the co-adding method
                # width=3.0 # specify the clipping width
                # iters=5 # specify the number of iterations
            )
            # --------------------------------------------------------

            dateobs_mflat = tool.calculate_average_date_obs(
                ic1.filter(imagetyp="FLAT", filter=filte).summary["date-obs"]
            )
            date = dateobs_mflat[:10].replace("-", "")
            flatim = f"{path_mframe}/{obs}/flat/{date}-n{filte}.fits"

            # 	Save to the database
            if not os.path.exists(os.path.dirname(flatim)):
                os.makedirs(os.path.dirname(flatim))

            cpcom = f"cp {path_data}/n{filte}.fits {flatim}"
            print(cpcom)
            os.system(cpcom)

            # 	Save to the dictionary
            flatdict[filte] = mflat

            if verbose_gpu:
                print(
                    f"flat combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )

            # 	Clear the momory pool
            del _ffc
            del mflat
            mempool.free_all_blocks()
            gc.collect()

            if verbose_gpu:
                print(
                    f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                )

        timetbl["status"][timetbl["process"] == "master_frame_flat"] = True

    else:
        pass

    delt_flat = time.time() - t0_flat
    print(f"Flat Master Frame: {delt_flat:.3f} sec")
    timetbl["time"][timetbl["process"] == "master_frame_flat"] = delt_flat

    # ------------------------------------------------------------
    # 	Object correction
    # ------------------------------------------------------------
    print(
        """#------------------------------------------------------------
#	Object correction
#------------------------------------------------------------
    """
    )
    if nobj == 0:

        end_localtime = time.strftime("%Y-%m-%d_%H:%M:%S_(%Z)", time.localtime())
        note = "No Light Frame"
        log_text = f"{path_new},{start_localtime},{end_localtime},{note}\n"

        with open(path_log, "a") as file:
            file.write(f"{log_text}")
        print(f"[EXIT!] {note}")
        sys.exit()

    print(f"{nobj} OBJECT: {list(objarr)}")
    t0_data_reduction = time.time()

    # 	OBJECT FRAME filter list

    objtbl = Table()
    objtbl["image"] = list(ic1.filter(imagetyp="LIGHT").files)
    objtbl["object"] = list(ic1.filter(imagetyp="LIGHT").summary["object"])
    objtbl["filter"] = list(ic1.filter(imagetyp="LIGHT").summary["filter"])
    objtbl["exptime"] = list(ic1.filter(imagetyp="LIGHT").summary["exptime"])
    objtbl["data_reduction"] = False
    objtbl["astrometry"] = False
    objtbl["photometry"] = False
    #
    BATCH_SIZE = 10  # 한 번에 처리할 이미지 수, 필요에 따라 조정
    #
    for filte in np.unique(objtbl["filter"]):
        for exptime in np.unique(objtbl["exptime"][objtbl["filter"] == filte]):
            fnamelist = list(
                objtbl["image"][
                    (objtbl["filter"] == filte) & (objtbl["exptime"] == exptime)
                ]
            )
            outfnamelist = [
                f"{path_data}/fdz{os.path.basename(fname)}" for fname in fnamelist
            ]

            print(f"{len(fnamelist)} OBJECT Correction: {exptime}s in {filte}-band")

            # 여기서 fnamelist를 BATCH_SIZE 만큼씩 나눠서 처리합니다.
            for i in range(0, len(fnamelist), BATCH_SIZE):
                batch_fnames = fnamelist[i : i + BATCH_SIZE]
                batch_outfnames = outfnamelist[i : i + BATCH_SIZE]

                print(f"[{i:0>4}] BATCH")

                ofc = FitsContainer(batch_fnames)
                if verbose_gpu:
                    print(
                        f"Object fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                    )

                # 	Take Dark Frame
                indx_closest_dark = np.argmin(np.abs(darkexptimearr - exptime))
                closest_dark_exptime = darkexptimearr[indx_closest_dark]
                exptime_scale = exptime / closest_dark_exptime

                # 	Take Flat Frame
                if filte in list(flatdict.keys()):
                    pass
                else:
                    print("No Master Flat. Let's Borrow")
                    dateobs = fits.getheader(fnamelist[0])["DATE-OBS"]
                    _objmjd = Time(dateobs, format="isot").mjd
                    pastflat = np.array(
                        glob.glob(f"{path_mframe}/{obs}/flat/*n{filte}*.fits")
                    )

                    deltime = []
                    for date in pastflat:

                        flatmjd = calib.isot_to_mjd(
                            (os.path.basename(date)).split("-")[0]
                        )
                        deltime.append(np.abs(_objmjd - flatmjd))

                    indx_closet = np.where(deltime == np.min(deltime))
                    tmpflat = pastflat[indx_closet].item()

                    with fits.open(tmpflat, mode="readonly") as hdul:
                        mflat = cp.asarray(hdul[0].data, dtype="float32")
                    flatdict[filte] = mflat
                    del mflat
                    mempool.free_all_blocks()

                # 	Reduction
                ofc.data = reduction(
                    ofc.data,
                    mbias,
                    darkdict[str(int(closest_dark_exptime))],
                    flatdict[filte],
                )
                ofc.write(batch_outfnames, overwrite=True)
                del ofc
                mempool.free_all_blocks()
                # gc.collect()
                if verbose_gpu:
                    print(
                        f"Object correction GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
                    )

    # 	Check Memory pool
    if verbose_gpu:
        print(
            f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
        )
    # 	Clear all momories pool
    mempool.free_all_blocks()
    # 	Bias
    del mbias
    # 	Dark Dictionary
    for key in list(darkdict.keys()):
        del darkdict[key]
    del darkdict
    # 	Dark Dictionary
    for key in list(flatdict.keys()):
        del flatdict[key]
    del flatdict
    mempool.free_all_blocks()
    if verbose_gpu:
        print(
            f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes"
        )

    # 	Add Reduced LIGHT FRAME
    # objtbl['reduced_image'] = [f"fdz{inim}" if os.path.exists(f"{path_data}/fdz{inim}"s) else None for inim in objtbl['raw_image']]

    # 	Corrected image list
    fdzimlist = sorted(glob.glob(f"{path_data}/fdz*.fits"))

    # 	Logging for the data reduction
    for ii, inim in enumerate(objtbl["image"]):
        fdzim = f"{path_data}/fdz{os.path.basename(inim)}"
        if fdzim in fdzimlist:
            objtbl["data_reduction"][ii] = True

    ic_fdzobj = ImageFileCollection(path_data, keywords="*", filenames=fdzimlist)

    delt_data_reduction = time.time() - t0_data_reduction

    print(
        f"OBJECT Correction: {delt_data_reduction:.3f} sec / {len(ic1.filter(imagetyp='LIGHT').summary)} frames"
    )

    timetbl["status"][timetbl["process"] == "data_reduction"] = True
    timetbl["time"][timetbl["process"] == "data_reduction"] = delt_data_reduction
    #
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.set_allocator(None)

    return ic_fdzobj, fdzimlist, objtbl
