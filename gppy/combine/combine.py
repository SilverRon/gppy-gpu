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

# time.sleep(60*60*2)

# ------------------------------------------------------------
from astropy.io import fits
from astropy.time import Time

# ------------------------------------------------------------
# from preprocess import calib

# from util import tool

# ------------------------------------------------------------
import warnings

# warnings.filterwarnings("ignore")
# ------------------------------------------------------------
# Plot preset
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 500
plt.rc("font", family="serif")


# ------------------------------------------------------------
# testflag = True
testflag = False
# ------------------------------------------------------------
from ..const import REF_DIR
from ..config import Configuration
from .utils import extract_date_and_time, calc_mean_dateloc, inputlist_parser, unpack


class SwarpCom:
    def __init__(self, imagelist_file=None) -> None:
        # ------------------------------------------------------------
        # 	Path
        # ------------------------------------------------------------

        if imagelist_file is None:
            imagelist_file = input(
                f"Text File Containing Images to Stack (/data/data.txt):"
            )

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

        self.ic_keys = [
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

        self._files = inputlist_parser(imagelist_file)

        if testflag:
            self._files = self._files[:3]

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

    def update_keys(self, new_keys):
        self.ic_keys = new_keys

    def update_keywords(self, new_keywords):
        self.keywords_to_add = new_keywords

    def set_imcollection(self):
        # ------------------------------------------------------------
        # 	Setting Metadata
        # ------------------------------------------------------------
        print(f"Reading images... (takes a few mins)")

        # 	Get Image Collection (takes some time)
        ic = ImageFileCollection(filenames=self._files, keywords=self.ic_keys)

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
        self.files = [filename for filename in filtered_table["file"]]
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
            path_save = f"/large_data/Commission/{self.obj}/{self.filte}"
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
        datestr, timestr = extract_date_and_time(dateobs_stacked)
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
            f"swarp -c {REF_DIR}/7dt.swarp @{self.path_imagelist} "
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


class Combine:
    def __init__(self):
        pass

    @classmethod
    def from_list(cls, images):
        image_list = []

        with open(datadir / "inputlist.txt", "w") as f:
            f.write("file\n")
            for filename in filelist:
                f.write(filename + "\n")

        for image in images:
            path = Path(image)
            if not path.is_file():
                print("The file does not exist.")
                return None
            image_list.append(path.parts[-1])
        working_dir = str(path.parent.absolute())
        config = Configuration.base_config(working_dir)
        config.file.processed_files = image_list
        return cls(config=config)

    @classmethod
    def from_file(cls, image):
        return cls.from_list([image])


#   Example
def process_image_list(image_list_file):
    SwarpCom(image_list_file).run()


if __name__ == "__main__":
    # 이미지 리스트 파일들의 경로를 리스트로 저장
    image_lists = [
        #   Broad
        "/large_data/Commission/NGC0253/g/select_median.txt",
        "/large_data/Commission/NGC0253/r/select_median.txt",
        "/large_data/Commission/NGC0253/i/select_median.txt",
        "/large_data/Commission/NGC0253/z/select_median.txt",
        #   Medium
        "/large_data/Commission/NGC0253/m400/select_median.txt",
        "/large_data/Commission/NGC0253/m425/select_median.txt",
        "/large_data/Commission/NGC0253/m450/select_median.txt",
        "/large_data/Commission/NGC0253/m475/select_median.txt",
        "/large_data/Commission/NGC0253/m500/select_median.txt",
        "/large_data/Commission/NGC0253/m525/select_median.txt",
        "/large_data/Commission/NGC0253/m550/select_median.txt",
        "/large_data/Commission/NGC0253/m575/select_median.txt",
        "/large_data/Commission/NGC0253/m600/select_median.txt",
        "/large_data/Commission/NGC0253/m625/select_median.txt",
        "/large_data/Commission/NGC0253/m650/select_median.txt",
        "/large_data/Commission/NGC0253/m675/select_median.txt",
        "/large_data/Commission/NGC0253/m700/select_median.txt",
        "/large_data/Commission/NGC0253/m725/select_median.txt",
        "/large_data/Commission/NGC0253/m750/select_median.txt",
        "/large_data/Commission/NGC0253/m775/select_median.txt",
        "/large_data/Commission/NGC0253/m800/select_median.txt",
        "/large_data/Commission/NGC0253/m825/select_median.txt",
        "/large_data/Commission/NGC0253/m850/select_median.txt",
        "/large_data/Commission/NGC0253/m875/select_median.txt",
    ]

    # 각 이미지 리스트 파일에 대해 SwarpCom 클래스를 실행
    for image_list in image_lists:
        process_image_list(image_list)
