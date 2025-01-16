# %%
# ============================================================
# 	Library
# ------------------------------------------------------------
from ccdproc import ImageFileCollection

# ------------------------------------------------------------
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

# time.sleep(60*60*2)
from datetime import datetime

# ------------------------------------------------------------
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time

# ------------------------------------------------------------
sys.path.append('../../src/preprocess/')
import calib

# from util import tool

# ------------------------------------------------------------
import warnings

warnings.filterwarnings("ignore")
# ------------------------------------------------------------
# Plot preset
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 500
plt.rc("font", family="serif")

class color:
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'


# ------------------------------------------------------------
# testflag = True
testflag = False
# ------------------------------------------------------------


# ------------------------------------------------------------
# 	Functions
# ------------------------------------------------------------
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


def inputlist_gen():
    """
    A makeshift function for testing.
    To be deleted in a real pipeline
    """
    from glob import glob
    from pathlib import Path

    datadir = Path("/data3/dhhyun/7DTStackCode/mytest/data/NGC0253/7DT01/m650")
    filelist = glob(str(datadir / "calib*0.fits"))
    with open(datadir / "inputlist.txt", "w") as f:
        f.write("file\n")
        for filename in filelist:
            f.write(filename + "\n")


def inputlist_parser(imagelist_file_to_stack):
    if os.path.exists(imagelist_file_to_stack):
        print(f"{imagelist_file_to_stack} found!")
    else:
        print(f"Not Found {imagelist_file_to_stack}!")
        sys.exit()
    input_table = Table.read(imagelist_file_to_stack, format="ascii")
    # input_table = Table.read(imagelist_file_to_stack, format="ascii.commented_header")
    _files = [f for f in input_table["file"].data]
    return _files


def unpack(packed, type, ex=None):
    if len(packed) != 1:
        print(f"There are more than one ({len(packed)}) {type}s")
        unpacked = input(
            f"Type {type.upper()} name (e.g. {packed if ex is None else ex}):"
        )
    else:
        unpacked = packed[0]
    return unpacked
    # return float(unpacked)


class SwarpCom:
    def run(self):
        # Setting Keywords
        # - the code should run without it but it allows more control
        # self.update_keys()

        # also you can do
        # self.zpkey = 'ZP'

        # sets miscellaneous variables
        ic = self.set_imcollection()

        # background subtraction
        self.bkgsub()

        # zero point scaling
        self.zpscale()

        # swarp imcombine
        self.swarp_imcom()

    def __init__(self, imagelist_file_to_stack=None) -> None:
        # ------------------------------------------------------------
        # 	Path
        # ------------------------------------------------------------
        self.path_config = "/home/snu/wohylee_gpPy/gppy-gpu/config" # Hard coded

        if imagelist_file_to_stack is None:
            imagelist_file_to_stack = input(f"Image List to Stack (/data/data.txt):")

        # not used?
        # self.path_calib = '/large_data/processed'

        # ------------------------------------------------------------
        # 	Setting
        # ------------------------------------------------------------

        # Unused?
        # self.keys = [
        #     "imagetyp",
        #     "telescop",
        #     "object",
        #     "filter",
        #     "exptime",
        #     "ul5_1",
        #     "seeing",
        #     "elong",
        #     "ellip",
        # ]

        # ------------------------------------------------------------
        # 	Keywords
        # ------------------------------------------------------------
        # 	Gain
        # gain_default = 0.779809474945068
        # 	ZeroPoint Key
        self.zpkey = f"ZP_AUTO"

        # 	Universal Facility Name
        self.obs = "7DT"

        self.keys_to_remember = [
            "EGAIN",
            "TELESCOP",
            "EGAIN",
            "FILTER",
            "OBJECT",
            "OBJCTRA",
            "OBJCTDEC",
            "JD",
            "MJD",
            "SKYVAL",
            "EXPTIME",
            self.zpkey,
        ]

        self.keywords_to_add = [
            "IMAGETYP",
            # "EXPOSURE",
            # "EXPTIME",
            "DATE-LOC",
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
            "CENTALT",
            "CENTAZ",
            "AIRMASS",
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
            "SWCREATE",
        ]

        self._files = inputlist_parser(imagelist_file_to_stack)

        if testflag:
            self._files = self._files[:3]

    def update_keys(self, new_keys):
        self.keys_to_remember = new_keys

    def update_keywords(self, new_keywords):
        self.keywords_to_add = new_keywords

    def set_imcollection(self):
        # ------------------------------------------------------------
        # 	Setting Metadata
        # ------------------------------------------------------------
        print(f"Reading images... (takes a few mins)")

        # 	Get Image Collection (takes some time)
        ic = ImageFileCollection(filenames=self._files, keywords=self.keys_to_remember)

        summary_table = ic.summary
        filtered_table = ic.summary[~ic.summary[self.zpkey].mask]
        print(len(summary_table), len(filtered_table))

        #   Former def
        # self.files = ic.files
        # self.n_stack = len(self.files)
        # self.zpvalues = ic.summary[self.zpkey].data
        # self.skyvalues = ic.summary["SKYVAL"].data
        # self.objra = ic.summary["OBJCTRA"].data[0].replace(" ", ":")
        # self.objdec = ic.summary["OBJCTDEC"].data[0].replace(" ", ":")
        # self.mjd_stacked = np.mean(ic.summary["MJD"].data)

        #   New def
        self.files = [filename for filename in filtered_table['file']]
        self.n_stack = len(self.files)
        self.zpvalues = filtered_table[self.zpkey].data
        self.skyvalues = filtered_table["SKYVAL"].data
        self.objra = filtered_table["OBJCTRA"].data[0].replace(" ", ":")
        self.objdec = filtered_table["OBJCTDEC"].data[0].replace(" ", ":")
        self.mjd_stacked = np.mean(filtered_table["MJD"].data)

        # 	Total Exposure Time [sec]
        self.total_exptime = np.sum(filtered_table["EXPTIME"])

        objs = np.unique(filtered_table["OBJECT"].data)
        filters = np.unique(filtered_table["FILTER"].data)
        egains = np.unique(filtered_table["EGAIN"].data)
        print(f"OBJECT(s): {objs} (N={len(objs)})")
        print(f"FILTER(s): {filters} (N={len(filters)})")
        print(f"EGAIN(s): {egains} (N={len(egains)})")
        self.obj = unpack(objs, "object")
        self.filte = unpack(filters, "filter", ex="m650")
        # self.gain_default = unpack(egains, "egain", ex="0.256190478801727")
        #   Hard coding for the UDS field
        self.gain_default = 0.78

        # ------------------------------------------------------------
        # 	Base Image for Image Alignment
        # ------------------------------------------------------------
        self.baseim = self.files[0]
        # self.zp_base = ic.summary[self.zpkey][0]
        self.zp_base = filtered_table[self.zpkey][0]

        # ------------------------------------------------------------
        # 	Print Input Summary
        # ------------------------------------------------------------
        print(f"Input Images to Stack ({len(self.files):_}):")
        for ii, inim in enumerate(self.files):
            print(f"[{ii:>6}] {os.path.basename(inim)}")
            if ii > 10:
                print("...")
                break

        # ------------------------------------------------------------
        #   Define Paths
        # ------------------------------------------------------------
        if testflag:
            path_save = Path("./out")
            if not path_save.exists():
                path_save.mkdir()
            self.path_save = path_save

            path_imagelist = path_save / "images_to_stack.txt"
            self.path_imagelist = path_imagelist

            path_bkgsub = path_save / "bkgsub"
            if not path_bkgsub.exists():
                path_bkgsub.mkdir()
            self.path_bkgsub = path_bkgsub

            path_scaled = path_save / "scaled"
            if not path_scaled.exists():
                path_scaled.mkdir()
            self.path_scaled = path_scaled

            path_resamp = path_save / "resamp"
            if not path_resamp.exists():
                path_resamp.mkdir()
            self.path_resamp = path_resamp

        else:
            path_save = f"/lyman/data1/factory_whlee/selection/{self.obj}/{self.filte}"
            self.path_save = path_save
            # 	Image List for SWarp
            path_imagelist = f"{path_save}/images_to_stack.txt"
            self.path_imagelist = path_imagelist
            # 	Background Subtracted
            path_bkgsub = f"{path_save}/bkgsub"
            if not os.path.exists(path_bkgsub):
                os.makedirs(path_bkgsub)
            self.path_bkgsub = path_bkgsub
            # 	Scaled
            path_scaled = f"{path_save}/scaled"
            if not os.path.exists(path_scaled):
                os.makedirs(path_scaled)
            self.path_scaled = path_scaled
            # 	Resampled (temp. files from SWarp)
            path_resamp = f"{path_save}/resamp"
            if not os.path.exists(path_resamp):
                os.makedirs(path_resamp)
            self.path_resamp = path_resamp

        return ic

    def bkgsub(self):
        # ------------------------------------------------------------
        # 	Global Background Subtraction
        # ------------------------------------------------------------
        print("BACKGROUND Subtraction...")
        _st = time.time()
        bkg_subtracted_images = []
        for ii, (inim, _bkg) in enumerate(zip(self.files, self.skyvalues)):
            print(f"[{ii:>6}] {os.path.basename(inim)}", end="")
            nim = f"{self.path_bkgsub}/{os.path.basename(inim).replace('fits', 'bkgsub.fits')}"
            if not os.path.exists(nim):
                with fits.open(inim, memmap=True) as hdul:  # 파일 열기
                    _data = hdul[0].data  # 데이터 접근
                    _hdr = hdul[0].header  # 헤더 접근
                    # _bkg = np.median(_data)
                    _data -= _bkg
                    print(f"- {_bkg:.3f}")
                    fits.writeto(nim, _data, header=_hdr, overwrite=True)
            bkg_subtracted_images.append(nim)
        self.bkg_subtracted_images = bkg_subtracted_images
        _delt = time.time() - _st
        print(f"--> Done ({_delt:.1f}sec)")

    def zpscale(self):
        bkg_subtracted_images = self.bkg_subtracted_images
        zpvalues = self.zpvalues

        # ------------------------------------------------------------
        # 	ZP Scale
        # ------------------------------------------------------------
        print(f"Flux Scale to ZP={self.zp_base}")
        zpscaled_images = []
        _st = time.time()
        for ii, (inim, _zp) in enumerate(zip(bkg_subtracted_images, zpvalues)):
            print(f"[{ii:>6}] {os.path.basename(inim)}", end=" ")
            _fscaled_image = f"{self.path_scaled}/{os.path.basename(inim).replace('fits', 'zpscaled.fits')}"
            if not os.path.exists(_fscaled_image):
                with fits.open(inim, memmap=True) as hdul:  # 파일 열기
                    _data = hdul[0].data  # 데이터 접근
                    _hdr = hdul[0].header  # 헤더 접근
                    _fscale = 10 ** (0.4 * (self.zp_base - _zp))
                    _fscaled_data = _data * _fscale
                    print(
                        f"x {_fscale:.3f}",
                    )
                    fits.writeto(_fscaled_image, _fscaled_data, _hdr, overwrite=True)
            zpscaled_images.append(_fscaled_image)
        self.zpscaled_images = zpscaled_images
        _delt = time.time() - _st
        print(f"--> Done ({_delt:.1f}sec)")

    def swarp_imcom(self):
        # ------------------------------------------------------------
        # 	Images to Combine for SWarp
        # ------------------------------------------------------------
        with open(self.path_imagelist, "w") as f:
            for inim in self.zpscaled_images:
                f.write(f"{inim}\n")

        # 	Get Header info
        exptime_stacked = self.total_exptime
        mjd_stacked = self.mjd_stacked
        jd_stacked = Time(mjd_stacked, format="mjd").jd
        dateobs_stacked = Time(mjd_stacked, format="mjd").isot
        # airmass_stacked = np.mean(airmasslist)
        # dateloc_stacked = calc_mean_dateloc(dateloclist)
        # alt_stacked = np.mean(altlist)
        # az_stacked = np.mean(azlist)

        center = f"{self.objra},{self.objdec}"
        datestr, timestr = calib.extract_date_and_time(dateobs_stacked)
        comim = f"{self.path_save}/calib_{self.obs}_{self.obj}_{datestr}_{timestr}_{self.filte}_{exptime_stacked:g}.com.fits"
        weightim = comim.replace("com", "weight")

        # ------------------------------------------------------------
        # 	Image Combine
        # ------------------------------------------------------------
        t0_stack = time.time()
        print(f"Total Exptime: {self.total_exptime}")

        # print(f"self.n_stack: {self.n_stack} (type: {type(self.n_stack)})")
        # print(f"self.gain_default: {self.gain_default} (type: {type(self.gain_default)})")
        
        #   Type
        self.n_stack = int(self.n_stack)
        self.gain_default = float(self.gain_default)

        gain = (2 / 3) * self.n_stack * self.gain_default
        # 	SWarp
        # swarpcom = f"swarp -c {path_config}/7dt.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} -SUBTRACT_BACK N -RESAMPLE_DIR {path_resamp} -GAIN_KEYWORD EGAIN -GAIN_DEFAULT {gain_default} -FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
        swarpcom = (
            f"swarp -c {self.path_config}/7dt.swarp @{self.path_imagelist} "
            f"-IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center} "
            f"-SUBTRACT_BACK N -RESAMPLE_DIR {self.path_resamp} "
            f"-GAIN_KEYWORD EGAIN -GAIN_DEFAULT {self.gain_default} "
            f"-FSCALE_KEYWORD FAKE -WEIGHTOUT_NAME {weightim}"
        )

        print(swarpcom)
        os.system(swarpcom)

        # t0_stack = time.time()
        # swarpcom = f"swarp -c {path_config}/7dt.nocom.swarp @{path_imagelist} -IMAGEOUT_NAME {comim} -CENTER_TYPE MANUAL -CENTER {center}"
        # print(swarpcom)
        # os.system(swarpcom)
        # delt_stack = time.time()-t0_stack
        # print(delt_stack)

        # 	Get Genenral Header from Base Image
        with fits.open(self.baseim) as hdulist:
            header = hdulist[0].header
            chdr = {key: header.get(key, None) for key in self.keywords_to_add}

        # 	Put General Header Infomation on the Combined Image
        with fits.open(comim) as hdulist:
            data = hdulist[0].data
            header = hdulist[0].header
            for key in list(chdr.keys()):
                header[key] = chdr[key]

        # 	Additional Header Information
        keywords_to_update = {
            "DATE-OBS": (
                dateobs_stacked,
                "Time of observation (UTC) for combined image",
            ),
            # 'DATE-LOC': (dateloc_stacked, 'Time of observation (local) for combined image'),
            "EXPTIME": (
                exptime_stacked,
                "[s] Total exposure duration for combined image",
            ),
            "EXPOSURE": (
                exptime_stacked,
                "[s] Total exposure duration for combined image",
            ),
            # 'CENTALT' : (alt_stacked,     '[deg] Average altitude of telescope for combined image'),
            # 'CENTAZ'  : (az_stacked,      '[deg] Average azimuth of telescope for combined image'),
            # 'AIRMASS' : (airmass_stacked, 'Average airmass at frame center for combined image (Gueymard 1993)'),
            "MJD": (
                mjd_stacked,
                "Modified Julian Date at start of observations for combined image",
            ),
            "JD": (
                jd_stacked,
                "Julian Date at start of observations for combined image",
            ),
            "SKYVAL": (0, "SKY MEDIAN VALUE (Subtracted)"),
            "GAIN": (gain, "Sensor gain"),
        }

        # 	Header Update
        with fits.open(comim, mode="update") as hdul:
            # 헤더 정보 가져오기
            header = hdul[0].header

            # 여러 헤더 항목 업데이트
            for key, (value, comment) in keywords_to_update.items():
                header[key] = (value, comment)

            # 	Stacked Single images
            # for nn, inim in enumerate(files):
            # 	header[f"IMG{nn:0>5}"] = (os.path.basename(inim), "")

            # 변경 사항 저장
            hdul.flush()

        delt_stack = time.time() - t0_stack

        print(f"Time to stack {self.n_stack} images: {delt_stack:.3f} sec")


#   Example
"""
if __name__ == "__main__":

    imagelist_file_to_stack = (
        "/large_data/Commission/UDS/T01_m400_filelist.dat"
    )

    # if the path is not given, you will be prompted to give one
    SwarpCom(imagelist_file_to_stack).run()
    # imcom = SwarpCom(imagelist_file_to_stack)
    # imcom.run()
"""

if __name__ == "__main__":
    mid_flt = ['m'+str(int(i)) for i in np.linspace(400, 875, 20)]
    broad_flt = ['u', 'g', 'r', 'i', 'z']

    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", "-s", type=str, nargs='?', help="Standard deviation cut of images\n\t'median': median cut \
                        \n\t'nsigma': +n sigma cut, n for specific float.")
    parser.add_argument("--filter", "-f", type=str, nargs='?', help="Filter type to stack\n\t'all': use all filters \
                        \n\t'medium': use only medium filters\n\t'broad': use only broad filters")
    parser.add_argument("--object", "-o", type=str, nargs='?', help="Target object name")
    parser.add_argument("--verbose", "-v", type=bool, nargs='?', help="Print outputs")

    verbose = False

    args = parser.parse_args()
    if args.object:
        obj = args.object
    else: 
        print(color.RED+' Need object name, Terminate the program'+color.END)
        exit()
    if args.filter:
        if args.filter == 'medium':
            flt = mid_flt
        elif args.filter == 'broad':
            flt = broad_flt
        elif args.filter == 'all':
            flt = mid_flt + broad_flt
        else:
            print(color.RED+' Need filter types, Terminate the program'+color.END)
            exit()
    if args.selection:
        sig_num = float(args.selection)
        type_select = f'{sig_num:.1f}sigma'
        print('Use '+color.YELLOW+f'{type_select}'+color.END+' cut criteria')
    else:
        type_select = 'median'  # default
        print('Use '+color.YELLOW+f'{type_select}'+color.END+' cut criteria')
    if args.verbose:
        verbose = args.verbose


    # 이미지 리스트 파일들의 경로를 리스트로 저장
    avail_image_lists = [f"/lyman/data1/factory_whlee/selection/{obj}/{filters}/select_{type_select}.txt" for filters in flt]
    image_lists = [_file for _file in avail_image_lists if os.path.isfile(_file)]
    print(image_lists)

    # 각 이미지 리스트 파일에 대해 SwarpCom 클래스를 실행
    for image_list in image_lists:
        SwarpCom(image_list).run()