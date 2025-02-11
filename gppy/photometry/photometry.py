import os, subprocess
from typing import Any, Union
from pathlib import Path
from glob import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

# astropy
from astropy.table import Table, hstack
from astropy.table import MaskedColumn
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy moduels
from . import utils as phot_util
from ..utils import get_sex_config, update_padded_header
from ..config import Configuration
from ..services.memory import MemoryMonitor
from ..external import run_sextractor


class Photometry:
    def __init__(self, config: Any = None, logger=None):
        """
        Initialize the Photometry class.
        config: Configuration object or string path to config yaml.
        """
        # Load Configuration
        if isinstance(config, str):  # In case of File Path
            self.config = Configuration(config_source=config)

            # Change working directory to the directory of the first image
            self.working_dir = Path(config).parent
            self.logger.info(self.working_dir)
            os.chdir(self.working_dir)
            # Set Logger
            self.logger.set_output_file(f"{self.working_dir}/phot.log")
        else:
            self.config = config.config

        if logger is None:
            from ..logger import Logger

            self.logger = Logger(name="7DT pipeline logger", slack_channel="pipeline_report")  # fmt:skip
        else:
            self.logger = logger

        # Get image list from config
        self.image = self.config.file.processed_files

        self.name = self.config.name

        self.logger.info("-" * 80)
        self.logger.info(f"Initialize photometry for {self.name}")

    def run(self):
        # Iterate over images
        for i, im in enumerate(self.image):
            im = os.path.join(self.config.path.path_processed, im)
            self.current_frame = i + 1  # f"{i:04}"
            self.logger.info(f"Photometry initiated for {self.name} [{self.current_frame}/{len(self.image)}]")  # fmt:skip

            self.define_info(im)
            reftbl = self.match_refcat(im)
            zptbl, setbl, indx_match, sep = self.do_photometry(im, reftbl)
            self.calculate_zp(zptbl, setbl)
            self.update_header(im, zptbl, reftbl, setbl, indx_match, sep)
            self.write_photcat(reftbl, setbl, indx_match, sep)

            self.logger.info(f"Photometry completed for {self.name} [{self.current_frame}/{len(self.image)}]")  # fmt:skip
            self.logger.debug(MemoryMonitor.log_memory_usage)
            MemoryMonitor.cleanup_memory()

        self.config.flag.single_photometry = True
        MemoryMonitor.cleanup_memory()

        self.logger.info(f"Photometry Done for {self.name}")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def define_info(self, inim):
        self.logger.debug(
            f"Define information for {self.name} [{self.current_frame}/{len(self.image)}]"
        )
        try:
            # ------------------------------------------------------------
            # 	Information from Configuration
            # ------------------------------------------------------------
            self.n_binning = self.config.obs.n_binning
            self.pixscale = self.config.obs.pixscale * self.n_binning
            # ------------------------------------------------------------
            # 	Information from Header
            # ------------------------------------------------------------
            self.hdr = fits.getheader(inim)

            self.obs = self.config.obs.unit
            self.obj = self.hdr["OBJECT"]
            self.filte = self.hdr["FILTER"]
            self.dateobs = self.hdr["DATE-OBS"]
            self.refmagkey = f"mag_{self.filte}"
            self.refmagerkey = f"magerr_{self.filte}"
            timeobj = Time(self.dateobs, format="isot")
            self.jd = timeobj.jd
            self.mjd = timeobj.mjd

            self.gain = self.hdr["EGAIN"]
            self.xcent, self.ycent = self.hdr["NAXIS1"] / 2.0, self.hdr["NAXIS2"] / 2.0
            w = WCS(inim)
            self.racent, self.decent = w.all_pix2world(self.xcent, self.ycent, 1)
            self.racent, self.decent = self.racent.item(), self.decent.item()
            # ------------------------------------------------------------
            # 	SExtractor Configuration
            # ------------------------------------------------------------
            self.head = inim.replace(".fits", "")
            self.cat = f"{self.head}.cat"

            # SExtractor Configuration for photometry
            self.conf, self.param, self.conv, self.nnw = get_sex_config("gregoryphot")

            self.phot_conf = self.config.photometry

            self.logger.info(
                f"Information of {self.name} [{self.current_frame}/{len(self.image)}] defined."
            )
        except Exception as e:
            self.logger.error(f"Error in Defining Information: {e}")

    def match_refcat(self, inim):
        self.logger.info(f"Run Pre-SExtractor for {self.name} [{self.current_frame}/{len(self.image)}]")  # fmt:skip
        try:
            # ------------------------------------------------------------
            # 	Run Pre-SExtractor
            # ------------------------------------------------------------
            precat = f"{self.head}.pre.cat"
            run_sextractor(inim, "simple", outcat=precat, logger=self.logger)

            # ------------------------------------------------------------
            # 	Get Reference Catalogue
            # ------------------------------------------------------------
            ref_gaiaxp_synphot_cat = (
                f"{self.config.path.path_refcat}/gaiaxp_dr3_synphot_{self.obj}.csv"
            )
            if not os.path.exists(ref_gaiaxp_synphot_cat):
                reftbl = phot_util.merge_catalogs(
                    target_coord=SkyCoord(self.racent, self.decent, unit="deg"),
                    path_calibration_field=self.config.path.path_calib_field,
                    matching_radius=self.phot_conf.match_radius * 1.5,
                    path_save=ref_gaiaxp_synphot_cat,
                )
                reftbl.write(ref_gaiaxp_synphot_cat, overwrite=True)
            else:
                reftbl = Table.read(ref_gaiaxp_synphot_cat)

            # ------------------------------------------------------------
            #   Matching
            # ------------------------------------------------------------
            self.logger.debug("Matching sources with reference catalog.")

            pretbl = Table.read(precat, format="ascii.sextractor")

            # select sources within ellipse at the center of image
            pretbl["within_ellipse"] = phot_util.is_within_ellipse(
                pretbl["X_IMAGE"],
                pretbl["Y_IMAGE"],
                self.xcent,
                self.ycent,
                self.phot_conf.photfraction * self.hdr["NAXIS1"] / 2,
                self.phot_conf.photfraction * self.hdr["NAXIS2"] / 2,
            )

            c_pre = SkyCoord(pretbl["ALPHA_J2000"], pretbl["DELTA_J2000"], unit="deg")
            c_ref = SkyCoord(reftbl["ra"], reftbl["dec"], unit="deg")

            indx_match, sep, _ = c_pre.match_to_catalog_sky(c_ref)
            _premtbl = hstack([pretbl, reftbl[indx_match]])
            _premtbl["sep"] = sep.arcsec
            premtbl = _premtbl[_premtbl["sep"] < self.phot_conf.match_radius]
            premtbl["within_ellipse"] = phot_util.is_within_ellipse(
                premtbl["X_IMAGE"],
                premtbl["Y_IMAGE"],
                self.xcent,
                self.ycent,
                self.phot_conf.photfraction * self.hdr["NAXIS1"] / 2,
                self.phot_conf.photfraction * self.hdr["NAXIS2"] / 2,
            )

            indx_star4seeing = np.where(
                (premtbl["FLAGS"] == 0)
                & (premtbl["within_ellipse"] == True)
                & (premtbl[self.refmagkey] > 11.75)
                & (premtbl[self.refmagkey] < 18.0)
            )
            self.ellipticity = np.median(premtbl["ELLIPTICITY"][indx_star4seeing])
            self.elongation = np.median(premtbl["ELONGATION"][indx_star4seeing])
            self.seeing = np.median(premtbl["FWHM_WORLD"][indx_star4seeing] * 3600)

            self.logger.debug(f"-" * 60)
            self.logger.debug(
                f"{len(premtbl[indx_star4seeing])} Star-like Sources Found"
            )
            self.logger.debug(f"SEEING     : {self.seeing:.3f} arcsec")
            self.logger.debug(f"ELONGATION : {self.elongation:.3f}")
            self.logger.debug(f"ELLIPTICITY: {self.ellipticity:.3f}")
            self.logger.debug(f"-" * 60)

            #   Return Reference Catalogue
            return reftbl
        except Exception as e:
            self.logger.error(f"Error in Pre-SExtractor: {e}")
            return

    def do_photometry(self, inim, reftbl):
        self.logger.info(
            f"Start Photometry for {self.name} [{self.current_frame}/{len(self.image)}]"
        )

        # ------------------------------------------------------------
        # 	APERTURE SETTING
        # ------------------------------------------------------------
        self.logger.debug("Setting Aperture for Photometry.")

        self.peeing = self.seeing / self.pixscale
        # 	Aperture Dictionary
        self.aperture_dict = {
            "MAG_AUTO": (0.0, "MAG_AUTO DIAMETER [pix]"),
            "MAG_APER": (
                2 * 0.6731 * self.peeing,
                "BEST GAUSSIAN APERTURE DIAMETER [pix]",
            ),
            "MAG_APER_1": (2 * self.peeing, "2*SEEING APERTURE DIAMETER [pix]"),
            "MAG_APER_2": (3 * self.peeing, "3*SEEING APERTURE DIAMETER [pix]"),
            "MAG_APER_3": (3 / self.pixscale, """FIXED 3" APERTURE DIAMETER [pix]"""),
            "MAG_APER_4": (5 / self.pixscale, """FIXED 5" APERTURE DIAMETER [pix]"""),
            "MAG_APER_5": (10 / self.pixscale, """FIXED 10" APERTURE DIAMETER [pix]"""),
        }

        self.add_aperture_dict = {}
        for key in list(self.aperture_dict.keys()):
            self.add_aperture_dict[key.replace("MAG_", "")] = (
                round(self.aperture_dict[key][0], 3),
                self.aperture_dict[key][1],
            )
        # 	MAG KEY
        self.magkeys = list(self.aperture_dict.keys())
        # 	MAG ERROR KEY
        # magerrkeys = [key.replace("MAG_", "MAGERR_") for key in self.magkeys]
        # 	Aperture Sizes
        aperlist = [self.aperture_dict[key][0] for key in self.magkeys[1:]]

        PHOT_APERTURES = ",".join(map(str, aperlist))
        # ------------------------------------------------------------
        # 	SOURCE EXTRACTOR CONFIGURATION FOR PHOTOMETRY
        # ------------------------------------------------------------

        self.logger.debug("Run Source Extractor for Photometry.")

        param_insex = dict(
            # ------------------------------
            # 	CATALOG
            # ------------------------------
            CATALOG_NAME=self.cat,
            # ------------------------------
            # 	CONFIG FILES
            # ------------------------------
            CONF_NAME=self.conf,
            PARAMETERS_NAME=self.param,
            FILTER_NAME=self.conv,
            STARNNW_NAME=self.nnw,
            # ------------------------------
            # 	EXTRACTION
            # ------------------------------
            # PSF_NAME = psf,
            DETECT_MINAREA=self.phot_conf.DETECT_MINAREA,
            DETECT_THRESH=self.phot_conf.DETECT_THRESH,
            DEBLEND_NTHRESH=self.phot_conf.DEBLEND_NTHRESH,
            DEBLEND_MINCONT=self.phot_conf.DEBLEND_MINCONT,
            # ------------------------------
            # 	PHOTOMETRY
            # ------------------------------
            # 	DIAMETER
            # 	OPT.APER, (SEEING x2), x3, x4, x5
            # 	MAG_APER	OPT.APER
            # 	MAG_APER_1	OPT.GAUSSIAN.APER
            # 	MAG_APER_2	SEEINGx2
            # 	...
            PHOT_APERTURES=PHOT_APERTURES,
            SATUR_LEVEL="65000.0",
            # GAIN = str(gain.value),
            GAIN=str(self.gain),
            PIXEL_SCALE=str(self.pixscale),
            # ------------------------------
            # 	STAR/GALAXY SEPARATION
            # ------------------------------
            SEEING_FWHM=str(2.0),
            # ------------------------------
            # 	BACKGROUND
            # ------------------------------
            BACK_SIZE=self.phot_conf.BACK_SIZE,
            BACK_FILTERSIZE=self.phot_conf.BACK_FILTERSIZE,
            BACKPHOTO_TYPE=self.phot_conf.BACKPHOTO_TYPE,
            # ------------------------------
            # 	CHECK IMAGE
            # ------------------------------
            # CHECKIMAGE_TYPE = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND',
            # CHECKIMAGE_NAME = '{},{},{},{}'.format(seg, aper, bkg, sub),
        )
        # 	Add Weight Map from SWarp
        weightim = inim.replace("com", "weight")
        if "com" in inim:
            if os.path.exists(weightim):
                param_insex["WEIGHT_TYPE"] = "MAP_WEIGHT"
                param_insex["WEIGHT_IMAGE"] = weightim
        # 	Check Image
        if self.phot_conf.check == True:
            param_insex["CHECKIMAGE_TYPE"] = (
                "SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND"
            )
            param_insex["CHECKIMAGE_NAME"] = (
                f"{self.head}.seg.fits,{self.head}.aper.fits,{self.head}.bkg.fits,{self.head}.sub.fits"
            )
        else:
            pass

        # ------------------------------------------------------------
        # 	PHOTOMETRY
        # ------------------------------------------------------------

        com = phot_util.sexcom(inim, param_insex)
        t0_sex = time.time()
        self.logger.debug(com)
        sexout = subprocess.getoutput(com)

        delt_sex = time.time() - t0_sex
        self.logger.debug(f"SourceEXtractor: {delt_sex:.3f} sec")

        line = [s for s in sexout.split("\n") if "RMS" in s]
        self.skymed, self.skysig = float(
            line[0].split("Background:")[1].split("RMS:")[0]
        ), float(line[0].split("RMS:")[1].split("/")[0])
        # os.system(f'rm {seg} {aper} {bkg} {sub}'.format(seg, aper, bkg, sub))

        setbl = Table.read(self.cat, format="ascii.sextractor")

        # ------------------------------------------------------------
        # 	Matching
        # ------------------------------------------------------------

        self.logger.debug("Matching sources with reference catalog.")

        # 	Proper Motion Correction
        #   Convert the Observation Time to an Astropy Time object
        obs_time = Time(self.dateobs, format="isot", scale="utc")

        #   Reference Epoch (Gaia DR3 is J2016.0)
        epoch_gaia = Time(2016.0, format="jyear")

        #   Get the RA, Dec, Proper Motions, and Parallaxes from the Reference Catalogue
        ra = reftbl["ra"]  # Unit: degrees
        dec = reftbl["dec"]  # Unit: degrees
        pmra = reftbl["pmra"]  # Unit: mas/yr
        pmdec = reftbl["pmdec"]  # Unit: mas/yr
        parallax = reftbl["parallax"]  # Unit: mas

        #   Check NaN of None Values for Proper Motions and Parallaxes
        #   If pmra, pmdec, or parallax is NaN, replace it with None
        pmra = np.where(np.isnan(pmra), None, pmra)
        pmdec = np.where(np.isnan(pmdec), None, pmdec)
        parallax = np.where(np.isnan(parallax), None, parallax)

        #   Generate a SkyCoord Object (Sources without proper motions are treated as None)
        c_ref = SkyCoord(
            ra=ra * u.deg,
            dec=dec * u.deg,
            pm_ra_cosdec=pmra * u.mas / u.yr if pmra is not None else None,
            pm_dec=pmdec * u.mas / u.yr if pmdec is not None else None,
            distance=(1 / (parallax * u.mas)) if parallax is not None else None,
            obstime=epoch_gaia,
        )  # Designate the Reference Epoch as J2016.0

        #   Apply Proper Motion Correction to the Reference Epoch
        c_ref_corrected = c_ref.apply_space_motion(new_obstime=obs_time)

        #   Matching with the Reference Catalogue using Corrected Coordinates
        c_sex = SkyCoord(setbl["ALPHA_J2000"], setbl["DELTA_J2000"], unit="deg")

        #   Do Matching
        indx_match, sep, _ = c_sex.match_to_catalog_sky(c_ref_corrected)

        # SourceEXtractor Catalog + Reference Catalog
        _mtbl = hstack([setbl, reftbl[indx_match]])
        _mtbl["sep"] = sep.arcsec
        mtbl = _mtbl[_mtbl["sep"] < self.phot_conf.match_radius]
        mtbl["within_ellipse"] = phot_util.is_within_ellipse(
            mtbl["X_IMAGE"],
            mtbl["Y_IMAGE"],
            self.xcent,
            self.ycent,
            self.phot_conf.photfraction * self.hdr["NAXIS1"] / 2,
            self.phot_conf.photfraction * self.hdr["NAXIS2"] / 2,
        )

        self.logger.info(
            f"""Matched Sources: {len(mtbl):_} (r={self.phot_conf.match_radius:.3f}")"""
        )

        for _, magkey in enumerate(self.magkeys):
            suffix = magkey.replace("MAG_", "")
            mtbl[f"SNR_{suffix}"] = mtbl[f"FLUX_{suffix}"] / mtbl[f"FLUXERR_{suffix}"]

        indx_star4zp = np.where(
            # 	Star-like Source
            # (mtbl['CLASS_STAR']>0.9) &
            (mtbl["FLAGS"] == 0)
            &
            # 	Within Ellipse
            (mtbl["within_ellipse"] == True)
            &
            # 	SNR cut
            (mtbl["SNR_AUTO"] > 20)
            &
            # 	Magnitude in Ref. Cat
            # (mtbl[f'{refmagkey}']<refmagupper) &
            # (mtbl[f'{refmagkey}']>refmaglower) &
            # (mtbl[f'{refmagerkey}']<refmaglower)
            #
            # (mtbl[refmagkey]>11.75) &
            (mtbl[self.refmagkey] > self.phot_conf.refmaglower)  # &
            # (mtbl[refmagkey]<18.0)
        )

        zptbl = mtbl[indx_star4zp]

        self.logger.info(
            f"{len(zptbl)} sources to calibration ZP in {self.name} [{self.current_frame}/{len(self.image)}]"
        )

        # Return ZP Table, Source Extractor Table, Matched Index, Separation
        return zptbl, setbl, indx_match, sep

    def calculate_zp(self, zptbl, setbl):
        # ------------------------------------------------------------
        # 	ZEROPOINT CALCULATION
        # ------------------------------------------------------------
        self.logger.info("Calculating zero points.")

        for _, magkey in enumerate(self.magkeys):
            magerrkey = magkey.replace("MAG", "MAGERR")

            sigma = 2.0

            zparr = zptbl[self.refmagkey] - zptbl[magkey]
            # zperrarr = tool.sqsum(zptbl[inmagerrkey], zptbl[refmagerkey])
            # 	Temperary zeropoint error!!!!!!
            zperrarr = phot_util.sqsum(zptbl[magerrkey], np.zeros_like(len(zptbl)))

            zparr_clipped = sigma_clip(
                zparr, sigma=sigma, maxiters=None, cenfunc=np.median, copy=False
            )

            indx_alive = np.where(zparr_clipped.mask == False)
            indx_exile = np.where(zparr_clipped.mask == True)

            # 	RE-DEF. ZP LIST AND INDEXING CLIPPED & NON-CLIPPED
            zptbl_alive = zptbl[indx_alive]
            zptbl_exile = zptbl[indx_exile]

            zp, zperr = phot_util.compute_median_mad(zparr[indx_alive])

            self.plot_zp(
                magkey, zp, zperr, zptbl, zparr, zperrarr, zptbl_alive, zptbl_exile
            )

            self.apply_zp(magkey, magerrkey, zp, zperr, setbl)

            # self.logger.info(f"{inmagkey} ZP: {zp:.3f}+/-{zperr:.3f}")

    def plot_zp(
        self, magkey, zp, zperr, zptbl, zparr, zperrarr, zptbl_alive, zptbl_exile
    ):
        # plt.errorbar(zptbl[refmagkey], zparr, xerr=zptbl[refmagerkey], yerr=zperrarr, ls='none', c='grey', alpha=0.5)
        plt.errorbar(
            zptbl[self.refmagkey],
            zparr,
            xerr=0,
            yerr=zperrarr,
            ls="none",
            c="grey",
            alpha=0.5,
        )
        plt.plot(
            zptbl_alive[self.refmagkey],
            zptbl_alive[self.refmagkey] - zptbl_alive[magkey],
            ".",
            c="dodgerblue",
            alpha=0.75,
            zorder=999,
            label=f"{len(zptbl_alive)}",
        )
        plt.plot(
            zptbl_exile[self.refmagkey],
            zptbl_exile[self.refmagkey] - zptbl_exile[magkey],
            "x",
            c="tomato",
            alpha=0.75,
            label=f"{len(zptbl_exile)}",
        )
        plt.axhline(
            y=zp, ls="-", lw=1, c="grey", zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}"
        )
        plt.axhspan(
            ymin=zp - zperr, ymax=zp + zperr, color="silver", alpha=0.5, zorder=0
        )
        plt.xlabel(self.refmagkey)
        # plt.xlim([8, 16])
        # plt.xlim([refmaglower-0.5, refmagupper+0.5])
        plt.axvspan(
            xmin=0,
            xmax=self.phot_conf.refmaglower,
            color="silver",
            alpha=0.25,
            zorder=0,
        )
        plt.axvspan(
            xmin=self.phot_conf.refmagupper,
            xmax=25,
            color="silver",
            alpha=0.25,
            zorder=0,
        )
        plt.xlim([10, 20])
        plt.ylim([zp - 0.25, zp + 0.25])
        plt.ylabel(f"ZP_{magkey}")
        plt.legend(loc="upper center", ncol=3)
        plt.tight_layout()
        im_path = Path(f"{self.head}.fits").parent
        if not os.path.exists(os.path.join(im_path, "phot_image")):
            os.makedirs(os.path.join(im_path, "phot_image"))
        im_mag_name = self.head.replace(str(im_path), "")
        plt.savefig(f"{im_path}/phot_image/{im_mag_name}.{magkey}.png", dpi=100)
        plt.close()

    def apply_zp(self, magkey, magerrkey, zp, zperr, setbl):
        # 	Apply ZP
        ##	MAG
        _calmagkey = f"{magkey}_{self.filte}"
        _calmagerrkey = f"{magerrkey}_{self.filte}"
        ##	FLUX
        _calfluxkey = _calmagkey.replace("MAG", "FLUX")
        _calfluxerrkey = _calmagerrkey.replace("MAG", "FLUX")
        ##  SNR
        _calsnrkey = _calmagkey.replace("MAG", "SNR")

        setbl[_calmagkey] = setbl[magkey] + zp
        setbl[_calmagerrkey] = phot_util.sqsum(setbl[magerrkey], zperr)

        # 	Flux [uJy]
        setbl[_calfluxkey] = (setbl[_calmagkey].data * u.ABmag).to(u.uJy).value
        # setbl[_calfluxerrkey] = setbl[_calfluxkey] * (10**(-0.4 * setbl[inmagerrkey]) - 1)
        # setbl[_calfluxerrkey] = compute_flux_density_error(magerr=setbl[_calmagerrkey], flux_density=setbl[_calfluxkey])
        setbl[_calfluxerrkey] = (
            0.4 * np.log(10) * setbl[_calfluxkey] * setbl[_calmagerrkey]
        )

        ## SNR
        setbl[_calsnrkey] = setbl[_calfluxkey] / setbl[_calfluxerrkey]

        # 	Formatting
        setbl[_calmagkey].format = ".3f"
        setbl[_calmagerrkey].format = ".3f"
        setbl[_calfluxkey].format = ".3f"
        setbl[_calfluxerrkey].format = ".3f"

        # 	Depth Calculation
        aperture_size = self.aperture_dict[magkey][0]
        if magkey == "MAG_AUTO":
            ul_3sig = 0.0
            ul_5sig = 0.0
        else:
            ul_3sig = phot_util.limitmag(3, zp, aperture_size, self.skysig)
            ul_5sig = phot_util.limitmag(5, zp, aperture_size, self.skysig)

        # ------------------------------------------------------------
        #   Update ZP Information to Header
        # ------------------------------------------------------------

        # 	Header keyword
        if magkey == "MAG_AUTO":
            _zpkey = magkey.replace("MAG", "ZP")
            _zperrkey = magerrkey.replace("MAGERR", "EZP")
            _ul3key = magkey.replace("MAG", "UL3")
            _ul5key = magkey.replace("MAG", "UL5")
        elif magkey == "MAG_APER":
            _zpkey = magkey.replace("MAG", "ZP").replace("APER", "0")
            _zperrkey = magerrkey.replace("MAGERR", "EZP").replace("APER", "0")
            _ul3key = magkey.replace("MAG", "UL3").replace("APER", "0")
            _ul5key = magkey.replace("MAG", "UL5").replace("APER", "0")
        else:
            _zpkey = magkey.replace("MAG", "ZP").replace("APER_", "")
            _zperrkey = magerrkey.replace("MAGERR", "EZP").replace("APER_", "")
            _ul3key = magkey.replace("MAG", "UL3").replace("APER_", "")
            _ul5key = magkey.replace("MAG", "UL5").replace("APER_", "")

        self.zp_dict = {
            _zpkey: (round(zp, 3), f"ZERO POINT for {magkey}"),
            _zperrkey: (round(zperr, 3), f"ZERO POINT ERROR for {magkey}"),
            _ul3key: (round(ul_3sig, 3), f"3 SIGMA LIMITING MAG FOR {magkey}"),
            _ul5key: (round(ul_5sig, 3), f"5 SIGMA LIMITING MAG FOR {magkey}"),
        }

        # Return Source Extractor Table
        return setbl

    def update_header(self, inim, zptbl, reftbl, setbl, indx_match, sep):
        self.logger.debug(
            f"Updating Header for {self.name} [{self.current_frame}/{len(self.image)}]"
        )

        # ------------------------------------------------------------
        # 	Header
        # ------------------------------------------------------------
        self.header_to_add = {
            "AUTHOR": ("Gregory S.H. Paek", "PHOTOMETRY AUTHOR"),
            "PHOTIME": (datetime.date.today().isoformat(), "PHTOMETRY TIME [KR]"),
            # 	Time
            "JD": (self.jd, "Julian Date of the observation"),
            "MJD": (self.mjd, "Modified Julian Date of the observation"),
            # 	Image Definition
            "SEEING": (round(self.seeing, 3), "SEEING [arcsec]"),
            "PEEING": (round(self.peeing, 3), "SEEING [pixel]"),
            "ELLIP": (round(self.ellipticity, 3), "ELLIPTICITY 1-B/A [0-1]"),
            "ELONG": (round(self.elongation, 3), "ELONGATION A/B [1-]"),
            "SKYSIG": (round(self.skysig, 3), "SKY SIGMA VALUE"),
            "SKYVAL": (round(self.skymed, 3), "SKY MEDIAN VALUE"),
            # 	Reference Source Conditions for ZP
            "REFCAT": (self.phot_conf.refcatname, "REFERENCE CATALOG NAME"),
            "MAGLOW": (self.phot_conf.refmaglower, "REF MAG RANGE, LOWER LIMIT"),
            "MAGUP": (self.phot_conf.refmagupper, "REF MAG RANGE, UPPER LIMIT"),
            "STDNUMB": (len(zptbl), "# OF STD STARS TO CALIBRATE ZP"),
        }

        self.header_to_add.update(self.add_aperture_dict)
        self.header_to_add.update(self.zp_dict)

        # ------------------------------------------------------------
        # 	ADD HEADER INFO
        # ------------------------------------------------------------
        update_padded_header(inim, self.header_to_add)

    def write_photcat(self, reftbl, setbl, indx_match, sep):
        # 	Add Reference Catalog Information
        keys_from_refcat = [
            "source_id",
            "bp_rp",
            "phot_g_mean_mag",
            f"mag_{self.filte}",
        ]

        # 각 키에 대해 매칭된 값들을 처리
        #   Operate Matched Values for Each Key
        for key in keys_from_refcat:
            valuearr = reftbl[key][indx_match].data  # Extract Matched Values

            #  Masking Values Exceeding the Matching Radius
            masked_valuearr = MaskedColumn(
                valuearr, mask=(sep.arcsec > self.phot_conf.match_radius)
            )

            #   Add or Update the Results to Source Extractor Table
            setbl[key] = masked_valuearr

        # 	Meta data
        meta_dict = {
            "obs": self.obs,
            "object": self.obj,
            "filter": self.filte,
            "date-obs": self.hdr["date-obs"],
            "jd": self.jd,
            "mjd": self.mjd,
        }
        setbl.meta = meta_dict
        setbl.write(f"{self.head}.phot.cat", format="ascii.tab", overwrite=True)
        self.logger.info(
            f"Header updated for {self.name} [{self.current_frame}/{len(self.image)}]"
        )


# if __name__ == "__main__":
#     test_phot_loc = "/data/pipeline_reform/processed_test_light/2025-01-01_1x1_gain2750/T00176/7DT11/m850/"
#     if list(glob(f"{test_phot_loc}*.yml")):
#         config = list(glob(f"{test_phot_loc}*.yml"))[0]
#         print(f"Configuration file found: {config}")
#     else:
#         print("No configuration file found.")
#         exit()
#     photo = Photometry(config)
#     photo.run()
