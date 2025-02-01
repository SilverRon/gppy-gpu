import os
import glob
import numpy as np
import eclaire as ec
from astropy.io import fits

from ..logging import Logger

logger = Logger(name="7DT masterframe logger")

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
from .utils import read_link, write_link, search_with_date_offsets
from ..queue import QueueManager, MemoryPriority


class MasterFrameGenerator:
    """
    Assumes BIAS, DARK, FLAT, SCI frames taken on the same date with the smae
    unit, n_binning and gain have all identical cameras.
    """

    def __init__(self, date, unit, n_binning, gain, device=None, queue=False):

        self.unit = unit
        self.date = date
        self.n_binning = n_binning
        self.gain = gain
        self.device = device

        self.path_raw = find_raw_path(unit, date, n_binning, gain)
        self.path_fdz = os.path.join(
            MASTER_FRAME_DIR, define_output_dir(date, n_binning, gain), unit
        )
        os.makedirs(self.path_fdz, exist_ok=True)

        self._update_logger()

        logger.info(
            f"Preprocessing for {define_output_dir(date, n_binning, gain)}/{unit} initialized."
        )
        logger.debug(f"Master frame output folder: {self.path_fdz}")

        header = self._get_sample_header()
        self.date_fdz = to_datetime_string(header["DATE-OBS"], date_only=True)  # UTC
        self.camera = get_camera(header)

        self._inventory_manifest()

        if queue:
            self.queue = QueueManager(
                logger=logger,
                auto_cleanup=True,  # Enable automatic memory cleanup
            )
            self.queue.start_processing()
        else:
            self.queue = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.queue:
            self.queue.cleanup_memory(force=True)
        else:
            import cupy as cp
            import gc

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            gc.collect()
        pass

    def _inventory_manifest(self):
        """Parses file names in the raw directory"""
        pattern = os.path.join(self.path_raw, f"{self.unit}_*.fits")
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

    def run(self):
        logger.info("Start generating master calibration frames.")

        if self.queue:
            # Add tasks with appropriate memory priorities
            task_id = self.queue.add_task(
                self.generate_mbias,
                priority=MemoryPriority.HIGH,
                gpu=True,
                task_name="generate_mbias",
                device=self.device,
            )
            self.queue.wait_until_all_tasks_complete()
            task_id = self.queue.add_task(
                self.generate_mdark,
                priority=MemoryPriority.MEDIUM,
                gpu=True,
                task_name="generate_mdark",
                device=self.device,
            )
            self.queue.wait_until_all_tasks_complete()
            task_id = self.queue.add_task(
                self.generate_mflat,
                priority=MemoryPriority.MEDIUM,
                gpu=True,
                task_name="generate_mflat",
                device=self.device,
            )
            self.queue.wait_until_all_tasks_complete()

            # Get final reports and cleanup
            self.queue.stop_processing()
            task_stats = self.queue.get_task_statistics()
            logger.debug(f"Task statistics: {task_stats}")
        else:
            self.generate_mbias()
            self.generate_mdark()
            self.generate_mflat()

        logger.info(
            f"MasterFrameGenerator("
            f"{self.date}, {self.unit}, {self.n_binning}, {self.gain}) ended"
        )

    def _update_logger(self):
        filename = f"{self.date}_{self.n_binning}x{self.n_binning}_gain{self.gain}.log"
        log_file = os.path.join(self.path_fdz, filename)
        logger.set_output_file(log_file)
        logger.set_pipeline_name(log_file)

    def _get_sample_header(self):
        """get any .head file in self.path_raw"""
        # """get first BIAS .head file"""
        # s = os.path.join(self.path_raw, f"{self.unit}_*BIAS*0.head")
        s = os.path.join(self.path_raw, f"{self.unit}_*.head")
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
            logger.warn(
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

    # def _borrow_file(self, dtype="bias", **kwargs):
    #     """returns path to master dtype"""
    #     logger.warning(
    #         f"No raw {dtype.upper()} files found for the date. "
    #         f"Fetching the closest past master {dtype}."
    #     )

    #     if dtype == "bias":
    #         output = getattr(self, f"m{dtype}_output")
    #     elif dtype == "dark":
    #         exptime = kwargs.pop("exptime")
    #         output = getattr(self, f"m{dtype}_output")[exptime]
    #     elif dtype == "flat":
    #         filt = kwargs.pop("filt")
    #         output = getattr(self, f"m{dtype}_output")[filt]
    #     else:
    #         raise ValueError(f"Invalid dtype: {dtype}")

    #     # Search past date with the output path as template
    #     return search_with_date_offsets(output)

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

        logger.info(
            f"{len(data_list)} {dtype.upper()} files found. "
            f"Initiate the generation of master frame {dtype.upper()}."
        )

        if self.queue:
            self.queue._update_memory_stats(f"Before combining {dtype}")

        bfc = ec.FitsContainer(data_list)

        if self.queue:
            self.queue._update_memory_stats(f"After loading {dtype} data")

        header = fits.getheader(data_list[0])

        if dtype == "bias":
            header = write_IMCMB_to_header(header, data_list)
            # rdnoise = self.calculate_rdnoise()
            # header['RDNOISE'] = rdnoise
        elif dtype == "dark":
            mbias_used = read_link(self.mbias_link)
            header = write_IMCMB_to_header(header, [mbias_used] + data_list)
            with load_data_gpu(mbias_used) as mbias:
                bfc.data -= mbias
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

        combined_data = ec.imcombine(
            bfc.data,
            combine=combine,
            **kwargs,
        )

        if self.queue:
            self.queue._update_memory_stats(f"After combining {dtype} data")

        fits.writeto(
            output,
            data=cp.asnumpy(combined_data),
            header=header,
            overwrite=True,
        )

        if self.queue:
            self.queue._update_memory_stats(f"After saving {dtype}")

        logger.info(f"Master {dtype.upper()} has been created: {output}")

    # fmt: off
    def generate_bpmask(self, mask, bias_sub_dark_names, 
                            n_sigma = 3, save_hot_pixel_mask = True, hot_pixel_mask_name = 'Stable_hot_pixel_mask2.fits'):
        """Func to generate hot pixel mask by J.H. Bae"""

        # from astropy.stats import sigma_clipped_stats
        from eclaire.stats import sigma_clipped_stats

        if self.queue:
            self.queue._update_memory_stats(f"After loading {dtype} data")

        # Open the bias subtracted dark images.
        bias_sub_darks = []
        for ii in range(len(bias_sub_dark_names)):
            bias_sub_darks.append(fits.getdata(bias_sub_dark_names[ii]))

        bias_sub_darks = np.array(bias_sub_darks)
        bias_sub_mean, _, _ = sigma_clipped_stats(bias_sub_darks, sigma = 3, maxiters = 5, axis = 0)
        
        dark_hdr = fits.open(bias_sub_dark_names[0])[0].header

        # Count hot pixels in the individual image. Used sigma clipping for calculating threshold.
        # hot_count = np.zeros(np.shape(mask))
        data = bias_sub_mean

        _, median, std = sigma_clipped_stats(data, sigma = 3, maxiters = 5)
        
        hot_mask0 = np.abs(data - median) > n_sigma * std
        hot_mask0 = hot_mask0.astype(int)

            # hot_count = hot_count + hot_mask0

        # Find the stable hot pixel.
        mask_stable = hot_mask0 # == len(bias_sub_darks)
        mask_stable = mask_stable.astype(int)

        # Save the file
        if save_hot_pixel_mask:
            newhdu = fits.PrimaryHDU(mask_stable)
            newhdu.header['NHOTPIX'] = (np.sum(mask_stable), 'Number of hot pixels.')
            newhdu.header['NDARKIMG'] = (len(bias_sub_dark_names), 'Number of bias subtracted dark images used in sigma-clipped mean combine.')
            newhdu.header['EXPTIME'] = dark_hdr['EXPTIME']
            newhdu.header['COMMENT'] = 'Stable hot pixel mask image. Algorithm produced by J.H.Bae on 20250117.'
            newhdu.header['SIGMAC'] = (n_sigma, 'The clipped sigma used for hot pixel extraction.')
        
            newhdul = fits.HDUList([newhdu])
            newhdul.writeto(hot_pixel_mask_name, overwrite = True)
            
        if self.queue:
            self.queue._update_memory_stats(f"After saving {dtype}")

        logger.info(f"Bad Pixel Mask has been created: {hot_pixel_mask_name}")

        # Update the input mask
        mask = mask + mask_stable
        return mask#, bias_sub_med
    # fmt: on

    # --------------- BIAS ---------------
    def generate_mbias(self, **kwargs):
        """
        Generates mbiases if there are raw BIAS files.
        Make mbias link files if there is a SCI frame.
        """

        if len(self.bias_input) > 0:  # if raw biases exist
            self._combine_and_save_eclaire(dtype="bias", **kwargs)

        # 2001-02-23_1x1_gain2750/7DT11/bias_20250102_C3.fits
        # 2001-02-23_1x1_gain2750/7DT11/bias_20250102_C3.link
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
            for exptime, mdark_file in self.mdark_output.items():  # exp is int
                self._combine_and_save_eclaire(dtype="dark", exptime=exptime, **kwargs)

        # links to what they're used for, not what exists
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
            for filt, mflat_file in self.mflat_output.items():
                self._combine_and_save_eclaire(dtype="flat", filt=filt, **kwargs)

        for filt, mflat_link in self.mflat_link.items():
            search_template = os.path.splitext(mflat_link)[0] + ".fits"
            # master_frame/2001-02-23_1x1_gain2750/7DT11/flat_20250102_m625_C3.fits
            mflat_file = search_with_date_offsets(search_template, future=False)
            write_link(mflat_link, mflat_file)

    # def _generate_mbias_eclaire(self, combine="median", **kwargs):
    #     """returns path to mbias"""
    #     logger.info(
    #         f"{len(self.bias_input)} BIAS files found."
    #         f"Initiate the generation of master frame BIAS."
    #     )

    #     bfc = ec.FitsContainer(self.bias_input)

    #     # imcombine eclaire
    #     mbias = ec.imcombine(
    #         bfc.data,
    #         combine=combine,
    #         **kwargs,
    #     )

    #     # later revise to use the combined header
    #     header = fits.getheader(self.bias_input[0])
    #     # rdnoise = self.calulate_rdnoise()
    #     # header['RDNOISE'] = rdnoise
    #     header = write_IMCMB_to_header(header, self.bias_input)

    #     fits.writeto(
    #         self.mbias_output,
    #         data=cp.asnumpy(mbias),
    #         header=header,
    #         overwrite=True,
    #     )

    #     logger.info(f"Master BIAS has been created: {self.mbias_output}")
    #     return self.mbias_output

    # def _generate_mdark_eclaire(self, exptime, combine="median", **kwargs):
    #     """
    #     eclaire imcombine does not return sigma map.
    #     to be modified for error map & hpmask
    #     """

    #     dimlist = self.dark_input[exptime]
    #     logger.info(f"{len(dimlist)} DARK files found for exposure {exptime}s. Initiate the generation of master frame DARK.")  # fmt:skip

    #     dfc = ec.FitsContainer(dimlist)

    #     mdark = ec.imcombine(
    #         dfc.data,
    #         combine=combine,
    #         **kwargs,
    #         # width=3.0 # specify the clipping width
    #         # iters=5 # specify the number of iterations
    #     )

    #     header = fits.getheader(self.dark_input[exptime][0])
    #     header = write_IMCMB_to_header(header, [self.mbias] + self.dark_input[exptime])

    #     mdark = mdark - load_data_gpu(self.mbias)

    #     # generate error map & hot pixel mask here
    #     # self._generate_hpmask()

    #     # later revise to use the combined header

    #     fits.writeto(
    #         self.mdark_output[exptime],
    #         data=cp.asnumpy(mdark),
    #         header=header,
    #         overwrite=True,
    #     )

    #     logger.info(f"Master DARK has been created: {self.mdark_output[exptime]}")

    #     return self.mdark_output[exptime]

    # def _generate_mflat_eclaire(self, filt, combine="median", **kwargs):

    #     flat_raw_imlist = self.flat_input[filt]
    #     ffc = ec.FitsContainer(flat_raw_imlist)
    #     logger.info(f"{len(flat_raw_imlist)} FLAT files found for {filt}. Initiate the generation of master frame FLAT.")  # fmt:skip
    #     dark_scaler, closest_dark_exp = self._get_correction_pars(filt)

    #     # 	Bias Correction
    #     ffc.data -= load_data_gpu(self.mbias_output)

    #     # 	Dark Correction
    #     ffc.data -= load_data_gpu(self.mdark_output[closest_dark_exp]) * dark_scaler

    #     # 	Flat Normalization
    #     ffc.data /= cp.median(ffc.data, axis=(1, 2), keepdims=True)

    #     # 	Combine All and Generate Master Flat
    #     mflat = ec.imcombine(ffc.data, combine=combine, **kwargs)

    #     # generate flatsig here

    #     # later revise to use the combined header
    #     header = fits.getheader(self.flat_input[filt][0])
    #     header = write_IMCMB_to_header(
    #         header,
    #         [self.mbias_output, self.mdark_output[closest_dark_exp]]
    #         + self.flat_input[filt],
    #     )

    #     fits.writeto(
    #         self.mflat_output[filt],
    #         data=cp.asnumpy(mflat),
    #         header=header,
    #         overwrite=True,
    #     )

    #     logger.info(f"Master FLAT has been created: {self.mflat_output[filt]}")

    #     return self.mflat_output[filt]

    # def _optimized_preproc(self):
    #     sci = (sci - mdark) / (mflat - mbias)
    #     reduction_kernel = cp.ElementwiseKernel(
    #         in_params='T x, T b, T d, T f',
    #         out_params='F z',
    #         operation='z = (x - b - d) / f',
    #         name='reduction'
    #     )
    #     # eclaire function
    #     def reduction(image,bias,dark,flat,out=None,dtype=None):
    #         dtype = judge_dtype(dtype)
    #         asarray = lambda x : cp.asarray(x,dtype=dtype)
    #         image = asarray(image)
    #         bias  = asarray(bias)
    #         dark  = asarray(dark)
    #         flat  = asarray(flat)
    #         if out is None:
    #             out = cp.empty(
    #                 cp.broadcast(image,bias,dark,flat).shape,
    #                 dtype=dtype
    #             )
    #         reduction_kernel(image,bias,dark,flat,out)
    #         return out
    #     pass
