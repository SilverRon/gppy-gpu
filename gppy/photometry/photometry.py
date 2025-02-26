import os
from typing import Any, List, Dict, Tuple, Optional, Union
import datetime
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from pathlib import Path
import glob

# astropy
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.table import Table, hstack, MaskedColumn
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy modules
from . import utils as phot_utils
from ..utils import update_padded_header
from ..config import Configuration
from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager, Priority
from .. import external

from dataclasses import dataclass


class Photometry:
    """
    A class to perform photometric analysis on astronomical images.

    This class handles both single-image and batch photometry processing,
    with support for parallel processing through a queue system.

    Attributes:
        config (Configuration): Configuration settings for photometry
        logger: Logger instance for output messaging
        queue (Optional[QueueManager]): Queue manager for parallel processing
        ref_catalog (str): Name of reference catalog to use
        images (List[str]): List of image files to process
    """

    def __init__(
        self,
        config: Any = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
        images: Optional[List[str]] = None,
        ref_catalog: Optional[str] = None,
    ) -> None:
        """
        Initialize the Photometry class.

        Args:
            config: Configuration object or path to config yaml
            logger: Logger instance for output messaging
            queue: Queue manager for parallel processing or boolean to create one
            images: List of image files to process
            ref_catalog: Name of reference catalog to use
        """
        # Load Configuration
        if isinstance(config, str):
            self.config = Configuration(config_source=config).config
        elif hasattr(config, "config"):
            self.config = config.config
        else:
            self.config = config

        self.logger = logger or self._setup_logger(config)
        self.queue = self._setup_queue(queue)
        self.ref_catalog = ref_catalog or self.config.photometry.refcatname
        self.images = images or self.config.file.processed_files

        os.makedirs(self.config.path.path_factory, exist_ok=True)

    @classmethod
    def from_list(cls, images: List[str]) -> Optional["Photometry"]:
        """
        Create Photometry instance from a list of image paths.

        Args:
            images: List of paths to image files

        Returns:
            Photometry instance or None if files don't exist
        """
        image_list = []
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
    def from_file(cls, image: str) -> Optional["Photometry"]:
        """Create Photometry instance from single image file."""
        return cls.from_list([image])

    @classmethod
    def from_dir(cls, dir_path: str) -> "Photometry":
        """Create Photometry instance from directory of FITS files."""
        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)

    def _setup_logger(self, config: Any) -> Any:
        """Initialize logger instance."""
        if hasattr(config, "logger") and config.logger is not None:
            return config.logger

        from ..logger import Logger

        return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

    def _setup_queue(self, queue: Union[bool, QueueManager]) -> Optional[QueueManager]:
        """Initialize queue manager for parallel processing."""
        if isinstance(queue, QueueManager):
            queue.logger = self.logger
            return queue
        elif queue:
            return QueueManager(logger=self.logger)
        return None

    def run(self) -> None:
        """
        Run photometry on all configured images.

        Processes images either sequentially or in parallel depending on queue configuration.
        Updates configuration flags and performs memory cleanup after completion.
        """
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

    def _run_parallel(self) -> None:
        """Process images in parallel using queue system."""
        task_ids = []
        for i, image in enumerate(self.images):
            process_name = f"{self.config.name}"
            phot_single = PhotometrySingle(
                image,
                self.config,
                self.logger,
                ref_catalog=self.ref_catalog,
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

    def _run_sequential(self) -> None:
        """Process images sequentially."""
        for image in self.images:
            PhotometrySingle(
                image,
                self.config,
                self.logger,
                ref_catalog=self.ref_catalog,
                total_image=len(self.images),
            ).run()


class PhotometrySingle:
    """
    Class for performing photometry on a single astronomical image.

    Handles the complete photometry pipeline for one image, including:
    - Source extraction
    - Reference catalog matching
    - Zero point calculation
    - Header updates
    - Results output

    Attributes:
        config: Configuration settings
        logger: Logger instance
        ref_catalog (str): Reference catalog name
        image (str): Path to image file
        image_info (ImageInfo): Parsed image metadata
        phot_conf: Photometry configuration settings
        name (str): Process name for logging
        header (ImageHeader): Header information container
        _id (int): Unique identifier for this instance
    """

    _id_counter = itertools.count(1)

    def __init__(
        self,
        image: str,
        config: Any,
        logger: Any = None,
        name: Optional[str] = None,
        ref_catalog: str = "GaiaXP",
        total_image: int = 1,
    ) -> None:
        """Initialize PhotometrySingle instance."""
        self.config = config
        self.logger = logger or self._setup_logger(config)
        self.ref_catalog = ref_catalog
        self.image = os.path.join(self.config.path.path_processed, image)
        self.image_info = ImageInfo.parse_image_info(
            self.image, pixscale=self.config.obs.pixscale
        )
        self.phot_conf = self.config.photometry
        self.name = name or self.config.name
        self.header = ImageHeader()
        if total_image == 1:
            self._id = next(self._id_counter)
        else:
            self._id = str(next(self._id_counter)) + "/" + str(total_image)

    def _setup_logger(self, config: Any) -> Any:
        """Initialize logger instance."""
        if hasattr(config, "logger") and config.logger is not None:
            return config.logger

        from ..logger import Logger

        return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

    @classmethod
    def from_file(cls, image: str) -> Optional["PhotometrySingle"]:
        """Create instance from single image file."""
        path = Path(image)
        if not path.is_file():
            print("The file does not exist.")
            return None
        working_dir = str(path.parent.absolute())
        config = Configuration.base_config(working_dir)
        image = path.parts[-1]
        return cls(image, config, name="user-input")

    @property
    def file_prefix(self) -> str:
        """Get file prefix for output files."""
        _tmp = os.path.splitext(self.image)[0]
        return os.path.join(self.config.path.path_factory, os.path.basename(_tmp))

    def run(self) -> None:
        """
        Execute complete photometry pipeline for single image.

        Performs the following steps:
        1. Loads reference catalog
        2. Calculates seeing conditions
        3. Runs source extraction
        4. Matches detected sources with reference catalog
        5. Calculates zero point corrections
        6. Updates image header
        7. Writes photometry catalog

        Times the complete process and performs memory cleanup after completion.
        """
        start_time = time.time()
        self.logger.info(f"Start Single Photometry for {self.name} [{self._id}]")

        self.load_ref_catalog()
        self.calculate_seeing()
        self.run_sextractor()
        self.zp_src_table = self.match_ref_catalog(
            snr_cut=20,
            low_mag_cut=self.phot_conf.ref_mag_lower,
            high_mag_cut=self.phot_conf.ref_mag_upper,
        )
        dicts = self.calculate_zp()
        self.update_header(*dicts)
        self.write_photcat()

        self.logger.debug(MemoryMonitor.log_memory_usage)
        MemoryMonitor.cleanup_memory()
        end_time = time.time()
        self.logger.info(
            f"Photometry Done for {self.name} [{self._id}] in {end_time - start_time:.2f} seconds"
        )

    def load_ref_catalog(self) -> None:
        """
        Load reference catalog for photometric calibration.

        Handles both standard and corrected GaiaXP catalogs.
        Creates new catalog if it doesn't exist by parsing Gaia data.
        Sets self.ref_src_table with loaded catalog data.
        """
        if self.ref_catalog == "GaiaXP_cor":
            ref_cat = f"{self.config.path.path_refcat}/cor_gaiaxp_dr3_synphot_{self.image_info.obj}.csv"
        elif self.ref_catalog == "GaiaXP":
            ref_cat = f"{self.config.path.path_refcat}/gaiaxp_dr3_synphot_{self.image_info.obj}.csv"

        if not os.path.exists(ref_cat) and "gaia" in self.ref_catalog:
            ref_src_table = phot_utils.parse_gaia_catalogs(
                target_coord=SkyCoord(
                    self.image_info.racent, self.image_info.decent, unit="deg"
                ),
                path_calibration_field=self.config.path.path_calib_field,
                matching_radius=self.phot_conf.match_radius * 1.5,
                path_save=ref_cat,
            )
            ref_src_table.write(ref_cat, overwrite=True)
        else:
            ref_src_table = Table.read(ref_cat)

        self.ref_src_table = ref_src_table

    def calculate_seeing(
        self, low_mag_cut: float = 11.75, high_mag_cut: float = 18.0
    ) -> None:
        """
        Calculate seeing conditions from stellar sources.

        Uses source extraction to identify stars and calculate median FWHM,
        ellipticity, and elongation values.

        Args:
            low_mag_cut: Lower magnitude limit for star selection
            high_mag_cut: Upper magnitude limit for star selection
        """
        precat = self.file_prefix + ".prep.cat"

        if True:  # Always run sextractor for prep
            self._run_sextractor(precat, prefix="prep")
        else:
            self.obs_src_table = Table.read(precat, format="ascii.sextractor")

        self.post_match_table = self.match_ref_catalog(
            snr_cut=False, low_mag_cut=low_mag_cut
        )

        self.header.seeing = np.median(self.post_match_table["FWHM_WORLD"] * 3600)
        self.header.peeing = self.header.seeing / self.image_info.pixscale
        self.header.ellipticity = round(
            np.median(self.post_match_table["ELLIPTICITY"]), 3
        )
        self.header.elongation = round(
            np.median(self.post_match_table["ELONGATION"]), 3
        )

        self.logger.debug(f"{len(self.post_match_table)} Star-like Sources Found")
        self.logger.debug(f"SEEING     : {self.header.seeing:.3f} arcsec")
        self.logger.debug(f"ELONGATION : {self.header.elongation:.3f}")
        self.logger.debug(f"ELLIPTICITY: {self.header.ellipticity:.3f}")

    def run_sextractor(self) -> None:
        """
        Run source extraction on the image.

        Configures and executes SExtractor with appropriate parameters
        based on seeing conditions and image characteristics.
        Updates header with sky background statistics.
        """
        self.logger.info(f"Start Source Extractor for {self.name}")

        self.logger.debug("Setting Aperture for Photometry.")
        sex_args = phot_utils.get_sex_args(
            self.image,
            self.phot_conf,
            self.image_info.gain,
            self.header.peeing,
            self.image_info.pixscale,
        )
        _, outcome = self._run_sextractor(
            self.file_prefix + ".cat",
            prefix="main",
            sex_args=sex_args,
            return_output=True,
        )
        outcome = [s for s in outcome.split("\n") if "RMS" in s][0]
        self.header.skymed = float(outcome.split("Background:")[1].split("RMS:")[0])
        self.header.skysig = float(outcome.split("RMS:")[1].split("/")[0])

    def _run_sextractor(
        self,
        output: str,
        prefix: str = "prep",
        sex_args: Optional[Dict] = None,
        **kwargs,
    ) -> Any:
        """
        Execute SExtractor on the image.

        Args:
            output: Path for output catalog
            prefix: Prefix for temporary files
            sex_args: Additional arguments for SExtractor
            **kwargs: Additional keyword arguments for SExtractor

        Returns:
            SExtractor execution outcome
        """
        self.logger.info(f"Run SExtractor ({prefix}) for {self.name}")
        outcome = external.sextractor(
            self.image,
            outcat=output,
            prefix=prefix,
            log_file=self.file_prefix + "_sextractor.log",
            logger=self.logger,
            sex_args=sex_args,
            **kwargs,
        )

        self.obs_src_table = Table.read(output, format="ascii.sextractor")
        return outcome

    def match_ref_catalog(
        self,
        snr_cut: Union[float, bool] = False,
        low_mag_cut: Union[float, bool] = False,
        high_mag_cut: Union[float, bool] = False,
    ) -> Table:
        """
        Match detected sources with reference catalog.

        Applies proper motion corrections and performs spatial matching.
        Filters matches based on separation, signal-to-noise, and magnitude limits.

        Args:
            snr_cut: Signal-to-noise ratio cut for filtering
            low_mag_cut: Lower magnitude limit
            high_mag_cut: Upper magnitude limit

        Returns:
            Table of matched sources meeting all criteria
        """
        self.logger.debug("Matching sources with reference catalog.")

        coord_ref = SkyCoord(
            ra=self.ref_src_table["ra"] * u.deg,
            dec=self.ref_src_table["dec"] * u.deg,
            pm_ra_cosdec=(
                self.ref_src_table["pmra"] * u.mas / u.yr
                if not np.isnan(self.ref_src_table["pmra"]).any()
                else None
            ),
            pm_dec=(
                self.ref_src_table["pmdec"] * u.mas / u.yr
                if not np.isnan(self.ref_src_table["pmdec"]).any()
                else None
            ),
            distance=(
                (1 / (self.ref_src_table["parallax"] * u.mas))
                if not np.isnan(self.ref_src_table["parallax"]).any()
                else None
            ),
            obstime=Time(2016.0, format="jyear"),
        )

        obs_time = Time(self.image_info.dateobs, format="isot", scale="utc")
        coord_ref = coord_ref.apply_space_motion(new_obstime=obs_time)

        coord_obs = SkyCoord(
            self.obs_src_table["ALPHA_J2000"],
            self.obs_src_table["DELTA_J2000"],
            unit="deg",
        )
        index_match, sep, _ = coord_obs.match_to_catalog_sky(coord_ref)

        _post_match_table = hstack(
            [self.obs_src_table, self.ref_src_table[index_match]]
        )
        _post_match_table["sep"] = sep.arcsec

        post_match_table = _post_match_table[
            _post_match_table["sep"] < self.phot_conf.match_radius
        ]
        post_match_table["within_ellipse"] = phot_utils.is_within_ellipse(
            post_match_table["X_IMAGE"],
            post_match_table["Y_IMAGE"],
            self.image_info.xcent,
            self.image_info.ycent,
            self.phot_conf.photfraction * self.image_info.naxis1 / 2,
            self.phot_conf.photfraction * self.image_info.naxis2 / 2,
        )

        suffixes = [
            key.replace("FLUXERR_", "")
            for key in post_match_table.keys()
            if "FLUXERR" in key
        ]

        for suffix in suffixes:
            post_match_table[f"SNR_{suffix}"] = (
                post_match_table[f"FLUX_{suffix}"]
                / post_match_table[f"FLUXERR_{suffix}"]
            )

        post_match_table = phot_utils.filter_table(post_match_table, "FLAGS", 0)
        post_match_table = phot_utils.filter_table(
            post_match_table, "within_ellipse", True
        )

        if low_mag_cut:
            post_match_table = phot_utils.filter_table(
                post_match_table,
                self.image_info.ref_mag_key,
                low_mag_cut,
                method="lower",
            )

        if high_mag_cut:
            post_match_table = phot_utils.filter_table(
                post_match_table,
                self.image_info.ref_mag_key,
                high_mag_cut,
                method="upper",
            )

        if snr_cut:
            post_match_table = phot_utils.filter_table(
                post_match_table, "SNR_AUTO", snr_cut, method="lower"
            )

        for key in [
            "source_id",
            "bp_rp",
            "phot_g_mean_mag",
            f"mag_{self.image_info.filter}",
        ]:
            valuearr = self.ref_src_table[key][index_match].data
            masked_valuearr = MaskedColumn(
                valuearr, mask=(sep.arcsec > self.phot_conf.match_radius)
            )
            self.obs_src_table[key] = masked_valuearr

        self.logger.info(
            f"""Matched Sources: {len(post_match_table)} (r={self.phot_conf.match_radius:.3f}")"""
        )

        return post_match_table

    def calculate_zp(self) -> Tuple[Dict, Dict]:
        """
        Calculate photometric zero point.

        Computes zero points and their errors for different apertures,
        creates diagnostic plots, and calculates limiting magnitudes.

        Args:
            zp_src_table: Table of matched sources for ZP calculation

        Returns:
            Tuple of dictionaries containing zero point and aperture information
        """

        zp_src_table = self.zp_src_table
        self.logger.info(
            f"{len(zp_src_table)} sources to calibration ZP in {self.name}"
        )
        self.logger.info("Calculating zero points.")

        aperture = phot_utils.get_aperture_dict(
            self.header.peeing, self.image_info.pixscale
        )

        zp_dict = {}
        aper_dict = {}
        for mag_key in aperture.keys():
            magerr_key = mag_key.replace("MAG", "MAGERR")

            zp_arr = zp_src_table[self.image_info.ref_mag_key] - zp_src_table[mag_key]
            zperr_arr = phot_utils.rss(
                zp_src_table[magerr_key], np.zeros_like(len(zp_src_table))
            )

            mask = sigma_clip(zp_arr, sigma=2.0).mask

            zp, zperr = phot_utils.compute_median_mad(np.array(zp_arr[~mask].value))

            keys = phot_utils.keyset(mag_key, self.image_info.filter)
            values = phot_utils.zp_correction(
                self.obs_src_table[mag_key], self.obs_src_table[magerr_key], zp, zperr
            )
            for key, val in zip(keys, values):
                self.obs_src_table[key] = val
                self.obs_src_table[key].format = ".3f"

            if mag_key == "MAG_AUTO":
                ul_3sig, ul_5sig = 0.0, 0.0
            else:
                ul_3sig, ul_5sig = phot_utils.limitmag(
                    np.array([3, 5]), zp, aperture[mag_key][0], self.header.skysig
                )

            suffix = mag_key.replace("MAG_", "")
            aper_dict[suffix] = round(aperture[mag_key][0], 3), aperture[mag_key][1]

            suffix = suffix.replace("APER", "0").replace("0_", "")
            zp_dict.update(
                {
                    f"ZP_{suffix}": (round(zp, 3), f"ZERO POINT for {mag_key}"),
                    f"EZP_{suffix}": (
                        round(zperr, 3),
                        f"ZERO POINT ERROR for {mag_key}",
                    ),
                    f"UL3_{suffix}": (
                        round(ul_3sig, 3),
                        f"3 SIGMA LIMITING MAG FOR {mag_key}",
                    ),
                    f"UL5_{suffix}": (
                        round(ul_5sig, 3),
                        f"5 SIGMA LIMITING MAG FOR {mag_key}",
                    ),
                }
            )

            self.plot_zp(mag_key, zp_src_table, zp_arr, zperr_arr, zp, zperr, mask)

        return (zp_dict, aper_dict)

    def plot_zp(
        self,
        mag_key: str,
        src_table: Table,
        zp_arr: np.ndarray,
        zperr_arr: np.ndarray,
        zp: float,
        zperr: float,
        mask: np.ndarray,
    ) -> None:
        """Generates and saves a zero-point calibration plot.
        The plot shows the zero-point values for each source and the final calibrated zero-point.
        Sources inside and outside the magnitude limits are plotted with different markers.
        Parameters
        ----------
        mag_key : str
            Key for the magnitude column in the source table
        src_table : astropy.table.Table
            Table containing source measurements and reference magnitudes
        zp_arr : array-like
            Array of individual zero-point values for each source
        zperr_arr : array-like
            Array of zero-point uncertainties for each source
        zp : float
            Final calibrated zero-point value
        zperr : float
            Uncertainty in the final zero-point
        mask : numpy.ndarray
            Boolean mask indicating which sources are within magnitude limits
        Returns
        -------
        None
            Saves plot as PNG file in the processed/phot_image directory
        """
        ref_mag = src_table[self.image_info.ref_mag_key]
        obs_mag = src_table[mag_key]

        plt.errorbar(
            ref_mag,
            zp_arr,
            xerr=0,
            yerr=zperr_arr,
            ls="none",
            c="grey",
            alpha=0.5,
        )

        plt.plot(
            ref_mag[~mask],
            ref_mag[~mask] - obs_mag[~mask],
            ".",
            c="dodgerblue",
            alpha=0.75,
            zorder=999,
            label=f"{len(ref_mag[~mask])}",
        )

        plt.plot(
            ref_mag[mask],
            ref_mag[mask] - obs_mag[mask],
            "x",
            c="tomato",
            alpha=0.75,
            label=f"{len(ref_mag[mask])}",
        )

        plt.axhline(
            y=zp, ls="-", lw=1, c="grey", zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}"
        )
        plt.axhspan(
            ymin=zp - zperr, ymax=zp + zperr, color="silver", alpha=0.5, zorder=0
        )
        plt.axvspan(
            xmin=0,
            xmax=self.phot_conf.ref_mag_lower,
            color="silver",
            alpha=0.25,
            zorder=0,
        )
        plt.axvspan(
            xmin=self.phot_conf.ref_mag_upper,
            xmax=25,
            color="silver",
            alpha=0.25,
            zorder=0,
        )

        plt.xlim([10, 20])
        plt.ylim([zp - 0.25, zp + 0.25])

        plt.xlabel(self.image_info.ref_mag_key)
        plt.ylabel(f"ZP_{mag_key}")

        plt.legend(loc="upper center", ncol=3)
        plt.tight_layout()

        im_path = os.path.join(self.config.path.path_processed, "phot_image")

        if not os.path.exists(im_path):
            os.makedirs(im_path)

        img_stem = os.path.splitext(os.path.basename(self.image))[0]
        plt.savefig(f"{im_path}/{img_stem}.{mag_key}.png", dpi=100)
        plt.close()

    def update_header(
        self,
        zp_dict: Dict[str, Tuple[float, str]],
        aper_dict: Dict[str, Tuple[float, str]],
    ) -> None:
        """
        Update the FITS image header with photometry information.

        Combines header information from multiple sources and updates the FITS file header.
        This includes photometry statistics, aperture information, and zero point data.

        Args:
            zp_dict: Dictionary containing zero point values and descriptions
                    Format: {'key': (value, description)}
            aper_dict: Dictionary containing aperture values and descriptions
                    Format: {'key': (value, description)}
        """
        self.logger.debug(f"Updating Header for {self.name}")
        header_to_add = {}
        header_to_add.update(self.header.dict)
        header_to_add.update(aper_dict)
        header_to_add.update(zp_dict)
        update_padded_header(self.image, header_to_add)

    def write_photcat(self) -> None:
        """
        Write photometry catalog to disk.

        Saves the source detection and photometry results to a catalog file.
        The catalog includes all detected sources with their measured properties
        and calculated photometric values.
        """
        metadata = self.image_info.metadata
        metadata["obs"] = self.config.obs.unit
        self.obs_src_table.meta = metadata
        self.obs_src_table.write(
            self.image.replace(".fits", ".phot.cat"), format="ascii.tab", overwrite=True
        )
        self.logger.info(f"Header updated for {self.name}")


@dataclass
class ImageHeader:
    """Stores image header information related to photometry."""

    author: str = "Gregory S.H. Paek"
    photime: str = datetime.date.today().isoformat()
    jd: float = 0.0
    mjd: float = 0.0
    seeing: float = 0.0
    peeing: float = 0.0
    ellipticity: float = 0.0
    elongation: float = 0.0
    skysig: float = 0.0
    skymed: float = 0.0
    refcat: str = "GaiaXP"
    maglow: float = 0.0
    magup: float = 0.0
    stdnumb: int = 0

    def __repr__(self) -> str:
        """Returns a string representation of the ImageHeader."""
        return ",\n".join(
            f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_")
        )

    @property
    def dict(self) -> Dict[str, Tuple[Any, str]]:
        """Generates a dictionary of header information for FITS."""
        return {
            "AUTHOR": (self.author, "PHOTOMETRY AUTHOR"),
            "PHOTIME": (self.photime, "PHTOMETRY TIME [KR]"),
            "JD": (self.jd, "Julian Date of the observation"),
            "MJD": (self.mjd, "Modified Julian Date of the observation"),
            "SEEING": (round(self.seeing, 3), "SEEING [arcsec]"),
            "PEEING": (round(self.peeing, 3), "SEEING [pixel]"),
            "ELLIP": (round(self.ellipticity, 3), "ELLIPTICITY 1-B/A [0-1]"),
            "ELONG": (round(self.elongation, 3), "ELONGATION A/B [1-]"),
            "SKYSIG": (round(self.skysig, 3), "SKY SIGMA VALUE"),
            "SKYVAL": (round(self.skymed, 3), "SKY MEDIAN VALUE"),
            "REFCAT": (self.refcat, "REFERENCE CATALOG NAME"),
            "MAGLOW": (self.maglow, "REF MAG RANGE, LOWER LIMIT"),
            "MAGUP": (self.magup, "REF MAG RANGE, UPPER LIMIT"),
            "STDNUMB": (self.stdnumb, "# OF STD STARS TO CALIBRATE ZP"),
        }


@dataclass
class ImageInfo:
    """Stores information extracted from a FITS image header."""

    obj: str  # Object name
    filter: str  # Filter used
    dateobs: str  # Observation date/time
    gain: float  # Gain value
    naxis1: int  # Image width
    naxis2: int  # Image height
    ref_mag_key: str  # Reference magnitude key
    ref_magerr_key: str  # Reference magnitude error key
    jd: float  # Julian Date
    mjd: float  # Modified Julian Date
    racent: float  # RA of image center
    decent: float  # DEC of image center
    xcent: float  # X coordinate of image center
    ycent: float  # Y coordinate of image center
    n_binning: int  # Binning factor
    pixscale: float  # Pixel scale [arcsec/pix]

    def __repr__(self) -> str:
        """Returns a string representation of the ImageInfo."""
        return ",\n".join(
            f"  {k}: {v}" for k, v in self.__dict__.items() if not k.startswith("_")
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Returns a dictionary of metadata."""
        return {
            "object": self.obj,
            "filter": self.filter,
            "date-obs": self.dateobs,
            "jd": self.jd,
            "mjd": self.mjd,
        }

    @classmethod
    def parse_image_info(cls, image_path: str, pixscale: float = 0.505) -> "ImageInfo":
        """Parses image information from a FITS header."""
        hdr = fits.getheader(image_path)
        w = WCS(image_path)

        xcent, ycent = hdr["NAXIS1"] / 2.0, hdr["NAXIS2"] / 2.0
        racent, decent = w.all_pix2world(xcent, ycent, 1)
        racent = float(racent)
        decent = float(decent)

        time_obj = Time(hdr["DATE-OBS"], format="isot")
        jd = float(time_obj.jd)
        mjd = float(time_obj.mjd)

        return cls(
            obj=hdr["OBJECT"],
            filter=hdr["FILTER"],
            dateobs=hdr["DATE-OBS"],
            gain=float(hdr["EGAIN"]),
            naxis1=int(hdr["NAXIS1"]),
            naxis2=int(hdr["NAXIS2"]),
            ref_mag_key=f"mag_{hdr['FILTER']}",
            ref_magerr_key=f"magerr_{hdr['FILTER']}",
            jd=jd,
            mjd=mjd,
            racent=racent,
            decent=decent,
            xcent=xcent,
            ycent=ycent,
            n_binning=hdr["XBINNING"],
            pixscale=hdr["XBINNING"] * pixscale,
        )
