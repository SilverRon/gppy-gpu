import gc
import cupy as cp
import eclaire as ec

from .utils import *
from .utils import load_data_gpu
from ..logging import logger


class Calibration:
    def __init__(self, config):
        logger.info("Calibration initialized.")
        self.config = config
        self._load_mbdf()

    def _load_mbdf(self):
        selection = self.config.config.preprocess.selection

        if selection == "default":  # use links under self.config.config.preprocess
            mbias_file = read_link(self.config.config.preprocess.mbias_link, timeout=10)
            self.config.config.preprocess.mbias_file = mbias_file
            logger.debug("Completed reading master BIAS link; 'selection' is 'default'")  # fmt:skip

            mdark_file = read_link(self.config.config.preprocess.mdark_link)
            self.config.config.preprocess.mdark_file = mdark_file
            logger.debug("Completed reading master DARK link; 'selection' is 'default'")  # fmt:skip

            mflat_file = read_link(self.config.config.preprocess.mflat_link)
            self.config.config.preprocess.mflat_file = mflat_file
            logger.debug("Completed reading master FLAT link; 'selection' is 'default'")  # fmt:skip

        if selection == "closest":  # search closest master frames again
            # looks for real files, not links

            self.config.config.preprocess.mbias_file = search_with_date_offsets(
                link_to_file(self.config.config.preprocess.mbias_link), future=True
            )
            self.config.config.preprocess.mdark_file = search_with_date_offsets(
                link_to_file(self.config.config.preprocess.mdark_link), future=True
            )
            self.config.config.preprocess.mflat_file = search_with_date_offsets(
                link_to_file(self.config.config.preprocess.mflat_link), future=True
            )

            # if found is the same as link: abort processing
            if (
                self.config.config.preprocess.mbias_file
                == read_link(self.config.config.preprocess.mbias_link)
                and self.config.config.preprocess.mdark_file
                == read_link(self.config.config.preprocess.mdark_link)
                and self.config.config.preprocess.mflat_file
                == read_link(self.config.config.preprocess.mflat_link)
            ):
                logger.info("All newly found master frames are the same as existing links")  # fmt: skip
                raise ValueError(
                    "No new closest master calibration frames. Aborting..."
                )

        if selection == "custom":
            # use self.config.config.preprocess.m????_file
            if self.config.config.preprocess.mbias_file is None:
                logger.error("No 'mbias_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mbias_file must be specified when selection is 'custom'.")  # fmt:skip

            if self.config.config.preprocess.mdark_file is None:
                logger.error("No 'mdark_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mdark_file must be specified when selection is 'custom'.")  # fmt:skip

            if self.config.config.preprocess.mflat_file is None:
                logger.error("No 'mflat_file' given although 'selection' is 'custom'")  # fmt:skip
                raise ValueError("mflat_file must be specified when selection is 'custom'.")  # fmt:skip

        # return mbias_file, mdark_file, mflat_file

    def run(self, use_eclaire=True):
        if not use_eclaire:
            self._calibrate_image_cupy()
        else:
            self._calibrate_image_eclaire()

    def _calibrate_image_eclaire(self, record_files=True):

        mempool = cp.get_default_memory_pool()

        mbias_file = self.config.config.preprocess.mbias_file
        mdark_file = self.config.config.preprocess.mdark_file
        mflat_file = self.config.config.preprocess.mflat_file
        raw_files = self.config.config.file.raw_files
        processed_files = self.config.config.file.processed_files

        logger.info(f"Calibrating {len(raw_files)} SCI frames: {self.config.config.obs.filter}, {self.config.config.obs.exposure}s")  # fmt:skip

        # batch processing
        BATCH_SIZE = 10
        for i in range(0, len(raw_files), BATCH_SIZE):
            batch_raw = raw_files[i : i + BATCH_SIZE]
            batch_processed = processed_files[i : i + BATCH_SIZE]

            ofc = ec.FitsContainer(batch_raw)
            logger.debug(f"Default GPU Memory for SCI FitsContainer Init: {mempool.used_bytes()*1e-6:.1f} Mbytes")  # fmt: skip

            # 	Reduction
            with load_data_gpu(mbias_file) as mbias, \
                 load_data_gpu(mdark_file) as mdark, \
                 load_data_gpu(mflat_file) as mflat:  # fmt:skip
                ofc.data = ec.reduction(ofc.data, mbias, mdark, mflat)
            # ofc.write(processed_files_batch, overwrite=True)

            # Save each slice of the cube as a separate 2D file
            for idx, output_path in enumerate(batch_processed):
                header = fits.getheader(raw_files[idx])
                if record_files:
                    header = write_IMCMB_to_header(
                        header, [mbias_file, mdark_file, mflat_file] + raw_files
                    )
                fits.writeto(
                    os.path.join(self.config.config.path.path_processed, output_path),
                    data=cp.asnumpy(ofc.data[idx]),
                    header=header,
                    overwrite=True,
                )
            logger.debug(f"Default GPU Memory at {i}-th Batch: {mempool.used_bytes()*1e-6:.1f} Mbytes")  # fmt: skip

        logger.debug(f"Default GPU Memory Before FitsContainer Cleanup: {mempool.used_bytes()*1e-6:.1f} Mbytes")  # fmt: skip
        del ofc
        gc.collect()  #  ensures no lingering references to GPU memory objects remain
        mempool.free_all_blocks()
        logger.debug(f"Default GPU Memory After FitsContainer Variable Cleanup: {mempool.used_bytes()*1e-6:.1f} Mbytes")  # fmt: skip
        logger.info(f"Calibration Done")
