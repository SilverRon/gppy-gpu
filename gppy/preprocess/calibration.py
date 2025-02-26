import cupy as cp
import eclaire as ec
from typing import Union, Any
from astropy.io import fits
import os
from pathlib import Path

from . import utils as prep_utils
from ..utils import add_padding

from ..services.memory import MemoryMonitor
from ..services.queue import QueueManager
from ..config import Configuration


class Calibration:
    def __init__(
        self,
        config: Union[str, Any] = None,
        logger: Any = None,
        queue: Union[bool, QueueManager] = False,
    ) -> None:
        """Initialize the astrometry module.

        Args:
            config: Configuration object or path to config file
            logger: Custom logger instance (optional)
            queue: QueueManager instance or boolean to enable parallel processing
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

    @classmethod
    def from_list(cls, images):
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
    def from_file(cls, image):
        return cls.from_list([image])

    @classmethod
    def from_dir(cls, dir_path):
        import glob

        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)

    def run(self, use_eclaire=True):

        self.logger.info("-" * 80)
        self.logger.info(f"Start calibration for {self.config.name}")

        self._load_mbdf()

        mbias_file = self.config.preprocess.mbias_file
        mdark_file = self.config.preprocess.mdark_file
        mflat_file = self.config.preprocess.mflat_file

        try:
            if use_eclaire:
                self._calibrate_image_eclaire(mbias_file, mdark_file, mflat_file)
            else:
                self._calibrate_image_cupy(mbias_file, mdark_file, mflat_file)

            self.config.flag.calibration = True
            self.logger.info(f"Calibration Done for {self.config.name}")
            MemoryMonitor.cleanup_memory()
            self.logger.debug(MemoryMonitor.log_memory_usage)

        except Exception as e:
            self.logger.error(f"Error during calibration: {str(e)}")
            raise

    def _load_mbdf(self):
        selection = self.config.preprocess.masterframe

        if selection == "default":  # use links under self.config.preprocess
            mbias_file = prep_utils.read_link(self.config.preprocess.mbias_link)
            self.config.preprocess.mbias_file = mbias_file
            self.logger.debug("Completed reading master BIAS link; 'selection' is 'default'")  # fmt:skip

            mdark_file = prep_utils.read_link(self.config.preprocess.mdark_link)
            self.config.preprocess.mdark_file = mdark_file
            self.logger.debug("Completed reading master DARK link; 'selection' is 'default'")  # fmt:skip

            mflat_file = prep_utils.read_link(self.config.preprocess.mflat_link)
            self.config.preprocess.mflat_file = mflat_file
            self.logger.debug("Completed reading master FLAT link; 'selection' is 'default'")  # fmt:skip

        elif selection == "closest":  # search closest master frames again
            from .utils import link_to_file, search_with_date_offsets

            # looks for real files, not links

            self.config.preprocess.mbias_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mbias_link), future=True
            )
            self.config.preprocess.mdark_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mdark_link), future=True
            )
            self.config.preprocess.mflat_file = search_with_date_offsets(
                link_to_file(self.config.preprocess.mflat_link), future=True
            )

            # if found is the same as link: abort processing
            if (
                self.config.preprocess.mbias_file
                == prep_utils.read_link(self.config.preprocess.mbias_link)
                and self.config.preprocess.mdark_file
                == prep_utils.read_link(self.config.preprocess.mdark_link)
                and self.config.preprocess.mflat_file
                == prep_utils.read_link(self.config.preprocess.mflat_link)
            ):
                self.logger.info("All newly found master frames are the same as existing links")  # fmt: skip
                raise ValueError(
                    "No new closest master calibration frames. Aborting..."
                )
        elif selection == "custom":
            # use self.config.preprocess.m????_file
            if self.config.preprocess.mbias_file is None:
                self.logger.error("No 'mbias_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mbias_file must be specified when selection is 'custom'.")  # fmt:skip

            if self.config.preprocess.mdark_file is None:
                self.logger.error("No 'mdark_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mdark_file must be specified when selection is 'custom'.")  # fmt:skip

            if self.config.preprocess.mflat_file is None:
                self.logger.error("No 'mflat_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mflat_file must be specified when selection is 'custom'.")  # fmt:skip

        # return mbias_file, mdark_file, mflat_file

    def _calibrate_image_eclaire(self, mbias_file, mdark_file, mflat_file):
        raw_files = self.config.file.raw_files
        processed_files = self.config.file.processed_files

        self.logger.debug(f"Calibrating {len(raw_files)} SCI frames: {self.config.obs.filter}, {self.config.obs.exposure}s")  # fmt:skip
        self.logger.debug(f"Current memory usage: {MemoryMonitor.log_memory_usage}")
        # batch processing
        BATCH_SIZE = 10
        for i in range(0, len(raw_files), BATCH_SIZE):
            batch_raw = raw_files[i : min(i + BATCH_SIZE, len(raw_files))]
            processed_batch = processed_files[i : min(i + BATCH_SIZE, len(raw_files))]

            ofc = ec.FitsContainer(batch_raw)

            # 	Reduction
            with prep_utils.load_data_gpu(mbias_file) as mbias, \
                 prep_utils.load_data_gpu(mdark_file) as mdark, \
                 prep_utils.load_data_gpu(mflat_file) as mflat:  # fmt:skip
                ofc.data = ec.reduction(ofc.data, mbias, mdark, mflat)

            # Save each slice of the cube as a separate 2D file
            for idx in range(len(batch_raw)):
                header = fits.getheader(raw_files[i + idx])
                header = prep_utils.write_IMCMB_to_header(
                    header,
                    [mbias_file, mdark_file, mflat_file, raw_files[i + idx]],
                )
                n_head_blocks = self.config.settings.header_pad
                header = add_padding(header, n_head_blocks, copy_header=False)

                path = self.config.path.path_processed
                output_path = os.path.join(path, processed_batch[idx])
                fits.writeto(
                    output_path,
                    data=cp.asnumpy(ofc.data[idx]),
                    header=header,
                    overwrite=True,
                )
            self.logger.debug(
                f"Current memory usage after {i}-th batch: {MemoryMonitor.log_memory_usage}"
            )

        self.logger.debug(
            f"Current memory usage before cleanup: {MemoryMonitor.log_memory_usage}"
        )
        MemoryMonitor.cleanup_memory()
        self.logger.debug(
            f"Current memory usage after cleanup: {MemoryMonitor.log_memory_usage}"
        )

    def _calibrate_image_cupy(self, mbias_file, mdark_file, mflat_file):
        pass
