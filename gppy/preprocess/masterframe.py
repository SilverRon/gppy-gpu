import os
import glob
import numpy as np
import eclaire as ec
from astropy.io import fits

from ..utils import (
    find_raw_path,
    to_datetime_string,
    header_to_dict,
    define_output_dir,
    get_camera,
    parse_exptime,
)

from ..const import MASTER_FRAME_DIR
import cupy as cp
from .utils import *
from ..services.queue import QueueManager, Priority
from ..services.memory import MemoryMonitor


class MasterFrameGenerator:
    """
    Assumes BIAS, DARK, FLAT, SCI frames taken on the same date with the smae
    unit, n_binning and gain have all identical cameras.
    """

    def __init__(self, obs_params, queue=False, logger=None):

        # Initialize variables
        self.initialize(obs_params)

        # Setup log
        self.logger = logger or self._setup_logger()

        # Setup queue
        self.queue = self._setup_queue(queue)
        
        self.logger.debug(f"Masterframe output folder: {self.path_fdz}")

    def initialize(self, obs_params):
        unit = obs_params["unit"]
        date = obs_params["date"]
        n_binning = obs_params["n_binning"]
        gain = obs_params["gain"]

        self.process_name = f"{date}_{n_binning}x{n_binning}_gain{gain}_{unit}_masterframe"
        
        self.path_raw = find_raw_path(unit, date, n_binning, gain)
        self.path_fdz = os.path.join(
            MASTER_FRAME_DIR, define_output_dir(date, n_binning, gain), unit
        )
        os.makedirs(self.path_fdz, exist_ok=True)

        header = self._get_sample_header()
        self.date_fdz = to_datetime_string(header["DATE-OBS"], date_only=True)  # UTC
        self.camera = get_camera(header)

        self._inventory_manifest()

    def _setup_logger(self):
        from ..logger import Logger
        logger = Logger(name="7DT masterframe logger")
        log_file = os.path.join(self.path_fdz, self.process_name + ".log")
        logger.set_output_file(log_file)
        logger.set_pipeline_name(log_file)
        return logger

    def _setup_queue(self, queue):
        if isinstance(queue, QueueManager):
            queue.logger = self.logger
            return queue
        elif queue:
            return QueueManager(logger=self.logger)
        return None

    def run(self):

        self.logger.debug(
            f"Start generating master calibration frames: {MemoryMonitor.log_memory_usage}"
        )

        if self.queue:
            task_id = self.queue.add_task(
                self.generate_mbias,
                priority=Priority.HIGH,
                gpu=True,
                task_name="generate_mbias",
            )
            self.queue.wait_until_task_complete(task_id)
            task_id = self.queue.add_task(
                self.generate_mdark,
                priority=Priority.MEDIUM,
                gpu=True,
                task_name="generate_mdark",
            )
            self.queue.wait_until_task_complete(task_id)
            task_id = self.queue.add_task(
                self.generate_mflat,
                priority=Priority.MEDIUM,
                gpu=True,
                task_name="generate_mflat",
            )
            self.queue.wait_until_task_complete(task_id)
        else:
            self.generate_mbias()
            self.generate_mdark()
            self.generate_mflat()

        self.logger.info(f"MasterFrameGenerator {self.process_name} completed")
        MemoryMonitor.cleanup_memory()

    def _inventory_manifest(self):
        """Parses file names in the raw directory"""
        pattern = os.path.join(self.path_raw, f"*.fits")
        all_fits = sorted(glob.glob(pattern))

        self._bias_input = []
        self._dark_input = {}
        self._flat_input = {}
        self._sci_input = []

        for fpath in all_fits:
            filename = os.path.basename(fpath)

            if "BIAS" in filename:
                self._bias_input.append(fpath)

            elif "DARK" in filename:
                exptime = parse_exptime(fpath, return_type=int)  # e.g., 100
                self._dark_input.setdefault(exptime, []).append(fpath)

            elif "FLAT" in filename:
                filt = filename.split("_")[4]  # Parse filter info
                self._flat_input.setdefault(filt, []).append(fpath)

            else:
                self._sci_input.append(fpath)  # Anything that isn't BIAS, DARK, or FLAT

        # These are DIFFERENT from bias_input and so on!
        # Extract exposures of science frames for current date and unit
        sci_exptimes = {
            int(float(parsed[6][:-1]))
            for parsed in (os.path.basename(s).split("_") for s in self._sci_input)
        }

        # Extract filts of science frames for current date and unit
        sci_filters = {
            parsed[4]
            for parsed in (os.path.basename(s).split("_") for s in self._sci_input)
        }

        # Define paths to links
        # CAVEAT: Links should be made for SCI Frames, not calib outputs

        self._mdark_link = {}
        for exptime in sci_exptimes:
            mdark_link = os.path.join(
                self.path_fdz, f"dark_{self.date_fdz}_{exptime}s_{self.camera}.link"
            )
            self._mdark_link[exptime] = mdark_link

        self._mflat_link = {}
        for filt in sci_filters:
            mdark_link = os.path.join(
                self.path_fdz, f"flat_{self.date_fdz}_{filt}_{self.camera}.link"
            )
            self._mflat_link[filt] = mdark_link

    @property
    def sci_input(self):
        """List of input science frame file paths"""
        # obsdata/7DT11/2001-02-23_gain2750/7DT11_20250102_082301_BIAS_m850_1x1_0.0s_0000.fits
        return self._sci_input

    @property
    def bias_input(self):
        """List of input bias file paths"""
        # obsdata/7DT11/2001-02-23_gain2750/7DT11_20250102_082301_BIAS_m850_1x1_0.0s_0000.fits
        return self._bias_input

    @property
    def dark_input(self):
        """Dict of input dark file paths grouped by integer exptime"""
        # obsdata/7DT11/2001-02-23_gain2750/7DT11_20250102_094601_DARK_m850_1x1_100.0s_0000.fits
        return self._dark_input

    @property
    def flat_input(self):
        """Dict of Input flat file paths grouped by filter"""
        # obsdata/7DT11/2001-02-23_gain2750/7DT11_20250102_092231_FLAT_m425_1x1_7.7s_0000.fits
        return self._flat_input

    @property
    def mbias_output(self):
        # master_frame/2001-02-23_1x1_gain2750/7DT11/bias_20250102_C3.fits
        return os.path.join(self.path_fdz, f"bias_{self.date_fdz}_{self.camera}.fits")

    @property
    def biassig_output(self):
        # master_frame/2001-02-23_1x1_gain2750/7DT11/biassig_20250102_C3.fits
        return os.path.join(
            self.path_fdz, f"biassig_{self.date_fdz}_{self.camera}.fits"
        )

    @property
    def mdark_output(self):
        # master_frame/2001-02-23_1x1_gain2750/7DT11/dark_20250102_100s_C3.fits
        mdarks_out_dict = {}
        for exptime in self.dark_input.keys():  # exptime is int
            fpath = os.path.join(
                self.path_fdz, f"dark_{self.date_fdz}_{exptime}s_{self.camera}.fits"
            )
            mdarks_out_dict[exptime] = fpath
        return mdarks_out_dict

    @property
    def darksig_output(self):
        # master_frame/2001-02-23_1x1_gain2750/7DT11/darksig_20250102_100s_C3.fits
        darksig_out_dict = {}
        for exptime in self.dark_input.keys():  # exptime is int
            darksig_out_dict[exptime] = os.path.join(
                self.path_fdz, f"darksig_{self.date_fdz}_{exptime}s_{self.camera}.fits"
            )
        return darksig_out_dict

    @property
    def bpmask_output(self):
        """Dict. bad pixel mask output path"""
        # master_frame/2001-02-23_1x1_gain2750/7DT11/bpmask_20250102_100s_C3.fits
        bpmasks_out_dict = {}
        for exptime in self.dark_input.keys():  # exptime is int
            bpmasks_out_dict[exptime] = os.path.join(
                self.path_fdz, f"bpmask_{self.date_fdz}_{exptime}s_{self.camera}.fits"
            )
        return bpmasks_out_dict

    @property
    def mflat_output(self):
        """Dict."""
        # master_frame/2001-02-23_1x1_gain2750/7DT11/flat_20250102_m625_C3.fits
        mflats_out_dict = {}
        for filt in self.flat_input.keys():
            mflats_out_dict[filt] = os.path.join(
                self.path_fdz, f"flat_{self.date_fdz}_{filt}_{self.camera}.fits"
            )
        return mflats_out_dict

    @property
    def flatsig_output(self):
        """Dict."""
        # master_frame/2001-02-23_1x1_gain2750/7DT11/flatsig_20250102_m625_C3.fits
        mflats_out_dict = {}
        for filt in self.flat_input.keys():
            mflats_out_dict[filt] = os.path.join(
                self.path_fdz, f"flatsig_{self.date_fdz}_{filt}_{self.camera}.fits"
            )
        return mflats_out_dict

    @property
    def mbias_link(self):
        """str. Always generated even though no SCI frame exists"""
        return os.path.join(self.path_fdz, f"bias_{self.date_fdz}_{self.camera}.link")

    @property
    def mdark_link(self):
        """dict"""
        return self._mdark_link

    @property
    def mflat_link(self):
        """dict"""
        return self._mflat_link

    def _get_sample_header(self):
        """get any .head file in self.path_raw"""
        s = os.path.join(self.path_raw, f"*.head")
        header_file = sorted(glob.glob(s))[0]
        return header_to_dict(header_file)

    def _get_flatdark(self, filt):
        """If no raw DARK for the date, searches 100s mdark of previous dates"""
        flat_raw_exp_arr = [parse_exptime(s) for s in self.flat_input[filt]]  # float
        flat_exp_repr = np.median(flat_raw_exp_arr)
        if len(self.mdark_output) > 0:
            mdarks_used = self.mdark_output
            darkexptimes = [float(i) for i in self.mdark_link.keys()]
            closest_dark_exp = darkexptimes[
                np.argmin(np.abs(darkexptimes - flat_exp_repr))
            ]
            mdark_used = mdarks_used[closest_dark_exp]
            dark_scaler = flat_exp_repr / closest_dark_exp
        else:
            self.logger.warn(
                f"No master dark frame for the date. "
                f"Searching previous dates for 100s mdark."
            )
            # master_frame/2001-02-23_1x1_gain2750/7DT11/dark_20250102_100s_C3.fits
            search_template = os.path.join(
                self.path_fdz, f"dark_{self.date_fdz}_100s_{self.camera}.fits"
            )
            mdark_used = search_with_date_offsets(search_template, future=False)
            dark_scaler = flat_exp_repr / 100
        return mdark_used, dark_scaler

    def _combine_and_save_eclaire(self, dtype, combine="median", **kwargs):

        if dtype == "bias":
            data_list = getattr(self, f"{dtype}_input")
            output = getattr(self, f"m{dtype}_output")
        elif dtype == "dark":
            exptime = kwargs.pop("exptime")
            data_list = getattr(self, f"{dtype}_input")[exptime]
            output = getattr(self, f"m{dtype}_output")[exptime]
        elif dtype == "flat":
            filt = kwargs.pop("filt")
            data_list = getattr(self, f"{dtype}_input")[filt]
            output = getattr(self, f"m{dtype}_output")[filt]
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        # Mind that there are bias, dark, flat sigma maps
        if os.path.exists(output):
            self.logger.debug(f"Master {dtype} already exists: {output}")
            return output

        self.logger.info(f"Start to generate masterframe {dtype.upper()}")
        self.logger.debug(f"{len(data_list)} {dtype.upper()} files found.")

        self.logger.debug(f"Before combining {dtype}: {MemoryMonitor.log_memory_usage}")
        if self.queue:
            self.queue.log_memory_stats(f"Before combining {dtype}")

        bfc = ec.FitsContainer(data_list)

        self.logger.debug(f"After loading {dtype}: {MemoryMonitor.log_memory_usage}")
        if self.queue:
            self.queue.log_memory_stats(f"After loading {dtype} data")

        header = fits.getheader(data_list[0])
        n = len(data_list)

        if dtype == "bias":
            header = write_IMCMB_to_header(header, data_list)
            # rdnoise = self.calculate_rdnoise()
            # header['RDNOISE'] = rdnoise

            # save bias sigma map (~read noise)
            fits.writeto(
                self.biassig_output,
                data=cp.asnumpy(cp.std(bfc.data, axis=0)) * n / (n - 1),
                header=header,
                overwrite=True,
            )
            self.logger.info(f"Generation of dark sigma map completed")

        elif dtype == "dark":
            mbias_used = read_link(self.mbias_link)
            header = write_IMCMB_to_header(header, [mbias_used] + data_list)
            with load_data_gpu(mbias_used) as mbias:
                bfc.data -= mbias

            # save dark sigma map
            fits.writeto(
                self.darksig_output[exptime],
                data=cp.asnumpy(cp.std(bfc.data, axis=0)) * n / (n - 1),
                header=header,
                overwrite=True,
            )
            self.logger.info(f"Generation of dark sigma map completed")
            # self.generate_bpmask(bfc.data)

        elif dtype == "flat":
            mbias_used = read_link(self.mbias_link)
            # dark_scaler, closest_dark_exp = self._get_dark_scalar(filt)
            # mdark_used = read_link(self.mdark_link[closest_dark_exp])
            mdark_used, dark_scaler = self._get_flatdark(filt)
            header = write_IMCMB_to_header(
                header,
                [mbias_used, mdark_used] + self.flat_input[filt],
            )
            with load_data_gpu(mbias_used) as mbias:
                bfc.data -= mbias
            with load_data_gpu(mdark_used) as mdark:
                bfc.data -= mdark * dark_scaler

            # Normalize Flats
            bfc.data /= cp.median(bfc.data, axis=(1, 2), keepdims=True)

            # save flat sigma map
            fits.writeto(
                self.flatsig_output[filt],
                data=cp.asnumpy(cp.std(bfc.data, axis=0)) * n / (n - 1),
                header=header,
                overwrite=True,
            )
            self.logger.info(f"Generation of dark sigma map completed")

        combined_data = ec.imcombine(
            bfc.data,
            combine=combine,
            **kwargs,
        )

        self.logger.debug(f"After combining {dtype}: {MemoryMonitor.log_memory_usage}")
        if self.queue:
            self.queue.log_memory_stats(f"After combining {dtype} data")

        if dtype == "dark":
            self.generate_bpmask(
                combined_data, self.bpmask_output[exptime], header=header
            )

        fits.writeto(
            output,
            data=cp.asnumpy(combined_data),
            header=header,
            overwrite=True,
        )

        self.logger.debug(f"After writing {dtype}: {MemoryMonitor.log_memory_usage}")
        if self.queue:
            self.queue.log_memory_stats(f"After writing {dtype}")

        self.logger.info(f"Generation of masterframe {dtype.upper()} completed")
        MemoryMonitor.cleanup_memory()
        self.logger.debug(f"Masterframe {dtype.upper()} has been created: {output}")
        return output

    def generate_bpmask(self, data, output, n_sigma=5, header=None):
        """
        Expects 2D cupy array as data (master dark)
        Func to generate hot pixel mask.
        Bad pixel criterion set by J.H. Bae
        """
        # from astropy.stats import sigma_clipped_stats
        # from eclaire.stats import sigma_clipped_stats

        # if self.queue:
        #     self.queue.log_memory_stats(f"After loading {dtype} data")

        # mean, median, std = sigma_clipped_stats(data, sigma=3, maxiters=5) # astropy
        # median, std = sigma_clipped_stats(data, reduce="median", width=3, iters=5) # eclaire
        mean, median, std = sigma_clipped_stats_cupy(data, sigma=3, maxiters=5)

        hot_mask = cp.abs(data - median) > n_sigma * std  # 1 for bad, 0 for okay
        hot_mask = cp.asnumpy(hot_mask).astype(int)  # gpu to cpu

        # Save the file
        newhdu = fits.PrimaryHDU(hot_mask)
        if header:
            newhdu.header = header
            # newhdu.header.add_comment("Above header is from the first dark frame")
            newhdu.header["COMMENT"] = "Header inherited first dark frame"
        newhdu.header["NHOTPIX"] = (np.sum(hot_mask), "Number of hot pixels.")
        # newhdu.header['NDARKIMG'] = (len(bias_sub_dark_names), 'Number of bias subtracted dark images used in sigma-clipped mean combine.')
        # newhdu.header['EXPTIME'] = dark_hdr['EXPTIME']
        # newhdu.header['COMMENT'] = 'Stable hot pixel mask image. Algorithm produced by J.H.Bae on 20250117.'
        newhdu.header["SIGMAC"] = (n_sigma, "HP threshold in clipped sigma")

        newhdul = fits.HDUList([newhdu])
        newhdul.writeto(output, overwrite=True)

        # if self.queue:
        #     self.queue.log_memory_stats(f"After saving {dtype}")

        self.logger.info(f"Generation of bad pixel mask completed")

    # --------------- BIAS ---------------
    def generate_mbias(self, **kwargs):
        """
        Generates mbiases if there are raw BIAS files.
        Make mbias link files if there is a SCI frame.
        """

        if len(self.bias_input) > 0:  # if raw biases exist
            mbias_file = self._combine_and_save_eclaire(dtype="bias", **kwargs)
        else:
            # 2001-02-23_1x1_gain2750/7DT11/bias_20250102_C3.fits
            # 2001-02-23_1x1_gain2750/7DT11/bias_20250102_C3.link
            self.logger.info(
                "No raw BIAS files found for the date. Searching for the closest past master BIAS."
            )
            search_template = os.path.splitext(self.mbias_link)[0] + ".fits"
            mbias_file = search_with_date_offsets(search_template, future=False)
        write_link(self.mbias_link, mbias_file)

    # --------------- DARK ---------------
    def generate_mdark(self, **kwargs):
        """
        Generates mdarks if there are raw DARK files.
        Make mdark link files for SCI frame and FLATs for the date & unit.
        """

        if len(self.dark_input) > 0:  # if raw darks exist
            # for multiple exptimes
            for exptime, mdark_link in self.mdark_link.items():  # exp is int
                mdark_file = self._combine_and_save_eclaire(
                    dtype="dark", exptime=exptime, **kwargs
                )
                write_link(mdark_link, mdark_file)
        else:
            # links to what they're used for, not what exists
            self.logger.info(
                "No raw DARK files found for the date. Searching for the closest past master DARK."
            )
            for exptime, mdark_link in self.mdark_link.items():
                search_template = os.path.splitext(mdark_link)[0] + ".fits"
                mdark_file = search_with_date_offsets(search_template, future=False)
                write_link(mdark_link, mdark_file)

    # --------------- FLAT ---------------
    def generate_mflat(self, **kwargs):
        """
        Generates mflats if there are raw FLAT files.
        Make mflat link files for SCI frame for the date & unit.
        """

        if len(self.flat_input) > 0:  # if raw flats exist
            # for multiple filters
            for filt, mflat_link in self.mflat_link.items():
                mflat_file = self._combine_and_save_eclaire(
                    dtype="flat", filt=filt, **kwargs
                )
                write_link(mflat_link, mflat_file)
        else:
            self.logger.info(
                "No raw FLAT files found for the date. Searching for the closest past master FLAT."
            )
            for filt, mflat_link in self.mflat_link.items():
                search_template = os.path.splitext(mflat_link)[0] + ".fits"
                # master_frame/2001-02-23_1x1_gain2750/7DT11/flat_20250102_m625_C3.fits
                mflat_file = search_with_date_offsets(search_template, future=False)
                write_link(mflat_link, mflat_file)


def sigma_clipped_stats_cupy(cp_data, sigma=3, maxiters=5):
    """
    Approximate sigma-clipping using CuPy.
    Computes mean, median, and std after iteratively removing outliers
    beyond 'sigma' standard deviations from the median.

    Parameters
    ----------
    cp_data : cupy.ndarray
        Flattened CuPy array of image pixel values.
    sigma : float
        Clipping threshold in terms of standard deviations.
    maxiters : int
        Maximum number of clipping iterations.

    Returns
    -------
    mean_val : float
        Mean of the clipped data (as a GPU float).
    median_val : float
        Median of the clipped data (as a GPU float).
    std_val : float
        Standard deviation of the clipped data (as a GPU float).
    """
    # Flatten to 1D for global clipping
    cp_data = cp_data.ravel()

    for _ in range(maxiters):
        median_val = cp.median(cp_data)
        std_val = cp.std(cp_data)
        # Keep only pixels within +/- sigma * std of the median
        mask = cp.abs(cp_data - median_val) < (sigma * std_val)
        cp_data = cp_data[mask]

    # Final statistics on the clipped data
    mean_val = cp.mean(cp_data)
    median_val = cp.median(cp_data)
    std_val = cp.std(cp_data)

    # Convert results back to Python floats on the CPU
    # return float(mean_val), float(median_val), float(std_val)
    return mean_val, median_val, std_val
