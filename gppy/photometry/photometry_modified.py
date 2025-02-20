import os
from typing import Any
import datetime
import numpy as np
import matplotlib.pyplot as plt

# astropy
from astropy.table import Table, hstack
from astropy.table import MaskedColumn
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy modules
from . import utils as phot_util
from ..utils import update_padded_header
from ..config import Configuration
from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager, Priority
from .. import external


class Photometry:
    def __init__(
        self, config: Any = None, logger=None, queue=False, images=None, ref_catalog="gaia"
    ):
        """
        Initialize the Photometry class.
        config: Configuration object or string path to config yaml.
        """
        # Load Configuration
        if isinstance(config, str):  # In case of File Path
            self.config = Configuration(config_source=config).config
        else:
            # for easy access to config
            self.config = config.config

        # Setup log
        self.logger = logger or self._setup_logger(config)
        
        # Setup queue
        self.queue = self._setup_queue(queue)
        
        self.ref_catalog = ref_catalog
        self.images = images or self.config.file.processed_files
        
        os.makedirs(self.config.path.path_factory, exist_ok=True)

    def _setup_logger(self, config):
        if hasattr(config, "logger") and config.logger is not None:
            return config.logger
        
        from ..logger import Logger
        return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

    def _setup_queue(self, queue):
        if isinstance(queue, QueueManager):
            queue.logger = self.logger
            return queue
        elif queue:
            return QueueManager(logger=self.logger)
        return None

    def run(self):
        self.logger.info("-" * 80)
        self.logger.info(f"Start photometry for {self.config.name}")

        if self.queue:
            self._run_parallel()
        else:
            self._run_sequential()

        self.config.flag.single_photometry = True
        MemoryMonitor.cleanup_memory()
        
        self.logger.info(f"Photometry Done for {self.config.name}")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def _run_parallel(self):
        task_ids = []
        for i, image in enumerate(self._images):
            process_name = f"{self.config.name} [{i+1}/{len(self._images)}]"
            phot_single = PhotometrySingle(
                image,
                self.config,
                self.logger,
                ref_catalog=self._ref_catalog,
            )
            task_id = self.queue.add_task(
                phot_single.run,
                kwargs={"name": process_name},
                priority=Priority.MEDIUM,
                gpu=False,
                task_name=process_name,
            )
            task_ids.append(task_id)
        self.queue.wait_until_task_complete(task_ids)

    def _run_sequential(self):
        for image in self._images:
            PhotometrySingle(
                image,
                self.config,
                self.logger,
                ref_catalog=self._ref_catalog,
            ).run()

class PhotometrySingle:
    def __init__(self, image, config, logger, name=None, ref_catalog="gaia"):
        self.config = config
        self.logger = logger
        self.ref_catalog = ref_catalog        
        self.image = os.path.join(self.config.path.path_processed, image)
        self.image_conf = phot_util.parse_image_config(self.image, pixscale=self.config.obs.pixscale)
        self.phot_conf = self.config.photometry
        self.name = name or self.config.name
        self.header = ImageHeader()

    @property
    def file_prefix(self):
        _tmp = os.path.splitext(self.image)[0]
        return os.path.join(
            self.config.path.path_factory, os.path.basename(_tmp))

    def run(self):
        """im is basename, not full path"""
        self.logger.info(f"Start Single Photometry for {self.name}")  # fmt:skip

        self.refbl = self.read_ref_catalog()
        self.get_seeing()
        self.run_sextractor()
        
        zptbl, setbl, indx_match, sep = self.match_refcat(reftbl)
        self.calculate_zp(zptbl, setbl)
        self.update_header(self.image, zptbl)
        self.write_photcat(reftbl, setbl, indx_match, sep)

        self.logger.debug(MemoryMonitor.log_memory_usage)
        MemoryMonitor.cleanup_memory()

        self.logger.info(f"Photometry completed for {self.config.name}")  # fmt:skip

    def read_ref_catalog(self):
        if self.ref_catalog == "cor_gaia":
            ref_cat = f"{self.config.path.path_refcat}/cor_gaiaxp_dr3_synphot_{self.obj}.csv"
        elif self.ref_catalog == "gaia":
            ref_cat = (
                f"{self.config.path.path_refcat}/gaiaxp_dr3_synphot_{self.obj}.csv"
            )

        if not os.path.exists(ref_cat) and "gaia" in self.ref_catalog:
            reftbl = phot_util.parse_gaia_catalogs(
                target_coord=SkyCoord(self.image_conf.racent, self.image_conf.decent, unit="deg"),
                path_calibration_field=self.config.path.path_calib_field,
                matching_radius=self.phot_conf.match_radius * 1.5,
                path_save=ref_cat,
            )
            reftbl.write(ref_cat, overwrite=True)
        else:
            reftbl = Table.read(ref_cat)

        return reftbl
    

    def get_seeing(self):  #   Return Reference Catalogue
    
        precat = self.file_prefix+".pre.cat"

        #if not(os.path.exist(precat)):  # maybe not need to re-run sextractor with prep?
        if True:
            self.run_sextractor(precat, prefix="prep")
            
        pretbl = Table.read(precat, format="ascii.sextractor")

        self.logger.debug("Matching sources with reference catalog.")

        # select sources within ellipse at the center of image
        pretbl["within_ellipse"] = phot_util.is_within_ellipse(
            pretbl["X_IMAGE"],
            pretbl["Y_IMAGE"],
            self.image_conf.racent,
            self.image_conf.decent,
            self.phot_conf.photfraction * self.hdr["NAXIS1"] / 2,
            self.phot_conf.photfraction * self.hdr["NAXIS2"] / 2,
        )

        c_pre = SkyCoord(pretbl["ALPHA_J2000"], pretbl["DELTA_J2000"], unit="deg")
        c_ref = SkyCoord(self.reftbl["ra"], self.reftbl["dec"], unit="deg")

        indx_match, sep, _ = c_pre.match_to_catalog_sky(c_ref)
        _premtbl = hstack([pretbl, self.reftbl[indx_match]])
        _premtbl["sep"] = sep.arcsec
        premtbl = _premtbl[_premtbl["sep"] < self.phot_conf.match_radius]
        premtbl["within_ellipse"] = phot_util.is_within_ellipse(
            premtbl["X_IMAGE"],
            premtbl["Y_IMAGE"],
            self.image_conf.xcent,
            self.image_conf.ycent,
            self.phot_conf.photfraction * self.hdr["NAXIS1"] / 2,
            self.phot_conf.photfraction * self.hdr["NAXIS2"] / 2,
        )

        indx_star4seeing = np.where(
            (premtbl["FLAGS"] == 0)
            & (premtbl["within_ellipse"] == True)
            & (premtbl[self.refmagkey] > 11.75)
            & (premtbl[self.refmagkey] < 18.0)
        )
        
        self.header.ellipticity = round(np.median(premtbl["ELLIPTICITY"][indx_star4seeing]), 3)
        self.header.elongation = round(np.median(premtbl["ELONGATION"][indx_star4seeing]), 3)
        seeing = np.median(premtbl["FWHM_WORLD"][indx_star4seeing] * 3600)
        peeing = seeing / self.image_conf.pixscale
        self.image_conf.seeing = round(seeing, 3)
        self.image_conf.peeing = round(peeing, 3)


        frame = self.config.name.split(" ")[-1]  # works even though self.config.name is inim
        self.logger.debug(f"-" * 60)
        self.logger.debug(f"{frame} {len(premtbl[indx_star4seeing])} Star-like Sources Found")  # fmt:skip
        self.logger.debug(f"{frame} SEEING     : {self.header.seeing:.3f} arcsec")
        self.logger.debug(f"{frame} ELONGATION : {self.header.elongation:.3f}")
        self.logger.debug(f"{frame} ELLIPTICITY: {self.header.ellipticity:.3f}")
        self.logger.debug(f"-" * 60)

        return seeing, peeing

    def run_sextractor(self):
        try:
            self.logger.info(f"Start Photometry for {self.config.name}")
            
            self.logger.debug("Setting Aperture for Photometry.")
            
            sex_args = phot_util.get_sex_args(self.image, self.phot_conf, self.image_conf)
            
            self.logger.info(f"Run SExtractor ({prefix}) for {self.name}")  # fmt:skip
            outcome = external.sextractor(
                self.image,
                outcat=output,
                prefix=prefix,
                log_file=self.file_prefix+"_sextractor.log",
                logger=self.logger,
                sex_args=sex_args,
            )

            outcome = [s for s in outcome.split("\n") if "RMS" in s][0]
            self.header.skymed = float(outcome.split("Background:")[1].split("RMS:")[0])
            self.header.skysig = float(outcome.split("RMS:")[1].split("/")[0])
            # os.system(f'rm {seg} {aper} {bkg} {sub}'.format(seg, aper, bkg, sub))

        except Exception as e:
            self.logger.error(f"Error in Main SExtractor: {e}")
            return
 
    def match_refcat(self, reftbl):
        setbl = Table.read(self.file_prefix + ".cat", format="ascii.sextractor")

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

        self.logger.info(f"""Matched Sources: {len(mtbl):_} (r={self.phot_conf.match_radius:.3f}")""")  # fmt:skip

        for _, magkey in enumerate(self.magkeys):
            suffix = magkey.replace("MAG_", "")
            mtbl[f"SNR_{suffix}"] = mtbl[f"FLUX_{suffix}"] / mtbl[f"FLUXERR_{suffix}"]

        zp_star_idx = np.where(
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

        zptbl = mtbl[zp_star_idx]

        self.logger.info(f"{len(zptbl)} sources to calibration ZP in {self.config.name}")

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
            # zperrarr = tool.sqsum(zptbl[magerrkey], zptbl[refmagerrkey])
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
        # im_path = Path(f"{self.head}.fits").parent
        im_path = os.path.join(self.config.path.path_processed, "phot_image")
        if not os.path.exists(im_path):
            os.makedirs(im_path)
        # im_mag_name = self.head.replace(str(im_path), "")
        img_stem = os.path.splitext(os.path.basename(self.im))[0]
        plt.savefig(f"{im_path}/{img_stem}.{magkey}.png", dpi=100)
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

    def update_header(self, inim, zptbl):
        self.logger.debug(f"Updating Header for {self.config.name}")

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

        # add to header
        for key in list(self.aperture_dict.keys()):
            self.header[key.replace("MAG_", "")] = (
                round(self.aperture_dict[key][0], 3),
                self.aperture_dict[key][1],
            )
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
        self.logger.info(f"Header updated for {self.config.name}")


@dataclass
class ImageHeader:
    author: str = "Gregory S.H. Paek"
    photime: str = datetime.date.today().isoformat()
    jd: float
    mjd: float
    seeing: float
    peeing: float
    ellipticity: float
    elongation: float
    skysig: float
    skymed: float
    refcat: str
    maglow: float
    magup: float
    stdnumb: int
