import os
import re
from typing import Any, List, Tuple, Union
from pathlib import Path
import glob
import time
from .services.queue import QueueManager, Priority
from . import external
from .services.memory import MemoryMonitor
from .config import Configuration
from .data import ObservationData


class Astrometry:
    """A class to handle astrometric solutions for astronomical images.

    This class manages the complete astrometric pipeline including plate solving,
    source extraction, and WCS header updates for astronomical images. It supports
    both sequential and parallel processing modes.

    Attributes:
        config (Configuration): Configuration object containing pipeline settings
        logger: Logger instance for recording operations and debugging
        queue (Optional[QueueManager]): Queue manager for parallel processing tasks

    Example:
        >>> astro = Astrometry(config="/data/...7DT05/m550/calib_7DT05_T00176_20250102_012738_m550_100.0.yml")
        >>> astro.run(solve_field=True, joint_scamp=True)
    """

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
        elif hasattr(config, "config"):
            self.config = config.config  # for easy access to config
        else:
            self.config = config

        # Setup log
        self.logger = logger or self._setup_logger(config)

        # Setup queue
        self.queue = self._setup_queue(queue)

        os.makedirs(self.config.path.path_factory, exist_ok=True)

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
        image_list = glob.glob(f"{dir_path}/*.fits")
        return cls.from_list(image_list)

    def _setup_logger(self, config):
        if hasattr(config, "logger") and config.logger is not None:
            return config.logger

        from .logger import Logger

        return Logger(name="7DT pipeline logger", slack_channel="pipeline_report")

    def _setup_queue(self, queue):
        if isinstance(queue, QueueManager):
            queue.logger = self.logger
            return queue
        elif queue:
            return QueueManager(logger=self.logger)
        return None

    def run(
        self,
        solve_field: bool = True,
        joint_scamp: bool = True,
        use_missfits: bool = False,
        processes=["sextractor", "scamp", "header_update"],
        prefix: str = "prep",
    ) -> None:
        """Execute the complete astrometry pipeline.

        Performs a sequence of operations including plate solving, source extraction,
        and WCS header updates. Supports both parallel and sequential processing.

        Args:
            solve_field: Whether to run plate-solving
            joint_scamp: Whether to run SCAMP jointly on all images
            use_missfits: Whether to use missfits for header updates
            processes: List of operations to perform
            prefix: Prefix for sextractor
        """
        try:
            self.logger.info("-" * 80)
            start_time = time.time()
            self.logger.info(f"Start astrometry for {self.config.name}")

            solved_files, soft_links, inims = self.define_paths()

            # solve-field
            if solve_field:
                self.run_solve_field(soft_links, solved_files)
            else:
                # add manual WCS update feature
                pass

            # Source Extractor
            if "sextractor" in processes:
                self.run_sextractor(solved_files, prefix=prefix)

            if "scamp" in processes:
                self.run_scamp(solved_files, joint=joint_scamp, prefix=prefix)

            if "header_update" in processes:
                self.update_header(
                    solved_files,
                    inims,
                    soft_links,
                    use_missfits=use_missfits,
                    prefix=prefix,
                )

            self.config.flag.astrometry = True

            end_time = time.time()
            self.logger.info(
                f"Astrometry Done for {self.config.name} in {end_time - start_time:.2f} seconds"
            )
            MemoryMonitor.cleanup_memory()
            self.logger.debug(MemoryMonitor.log_memory_usage)
        except Exception as e:
            self.logger.error(f"Error during astrometry processing: {str(e)}")
            raise

        # polygon - field rotation

    def define_paths(self) -> Tuple[List[str], List[str], List[str]]:
        """Initialize the astrometry processing environment.

        Sets up necessary file paths and creates symbolic links for processing.

        Returns:
            tuple: Contains:
                - solved_files: List of paths where solved FITS will be stored
                - soft_links: List of symbolic link paths in factory directory
                - inims: List of original input image paths
        """
        fnames = self.config.file.processed_files
        inims = [os.path.join(self.config.path.path_processed, s) for s in fnames]
        soft_links = [os.path.join(self.config.path.path_factory, s) for s in fnames]

        for inim, soft_link in zip(inims, soft_links):
            if not os.path.exists(soft_link):
                os.symlink(inim, soft_link)

        solved_files = [os.path.splitext(s)[0] + "_solved.fits" for s in soft_links]
        return solved_files, soft_links, inims

    def run_solve_field(self, inputs: List[str], outputs: List[str]) -> None:
        """Run astrometric plate-solving on input images.

        Uses astrometry.net's solve-field to determine WCS solution for each image.
        Supports parallel processing through queue system if enabled.

        Args:
            inputs: Paths to input FITS files
            outputs: Paths where solved FITS files will be written
        """
        # parallelize if queue=True
        self.logger.info(f"Start solve-field")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        if self.queue:
            self._submit_task(
                external.solve_field,
                zip(inputs, outputs),
                dump_dir=self.config.path.path_factory,
                pixscale=self.config.obs.pixscale,
            )
        else:
            for i, (slink, sfile) in enumerate(zip(inputs, outputs)):
                external.solve_field(
                    slink,
                    outim=sfile,
                    dump_dir=self.config.path.path_factory,
                    pixscale=self.config.obs.pixscale,
                )
                self.logger.info(f"Completed solve-field for {self.config.name} [{i+1}/{len(inputs)}]")  # fmt:skip
                self.logger.debug(f"input: {slink}, output: {sfile}")  # fmt:skip

        self.logger.debug(MemoryMonitor.log_memory_usage)

    def run_sextractor(self, files: List[str], prefix: str = "prep") -> List[str]:
        """Run Source Extractor on solved images.

        Extracts sources from solved images for use in SCAMP calibration.
        Creates FITS_LDAC format catalogs required by SCAMP.

        Args:
            files: Paths to astrometrically solved FITS files

        Returns:
            List of paths to generated source catalogs
        """
        # parallelize if queue=True
        self.logger.info("Start pre-sextractor")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        if self.queue:
            self._submit_task(
                external.sextractor,
                files,
                prefix=prefix,
                logger=self.logger,
                sex_args=["-catalog_type", "fits_ldac"],
            )
        else:
            for i, solved_file in enumerate(files):
                external.sextractor(
                    solved_file,
                    prefix=prefix,
                    logger=self.logger,
                    sex_args=["-catalog_type", "fits_ldac"],
                )
                self.logger.info(f"Completed sextractor (prep) for {self.config.name} [{i+1}/{len(files)}]")  # fmt:skip
                self.logger.debug(f"{solved_file}")  # fmt:skip

        self.logger.debug(MemoryMonitor.log_memory_usage)

    def run_scamp(
        self,
        files: List[str],
        joint: bool = True,
        prefix: str = "prep",
        astrefcat: str = None,
    ) -> None:
        """Run SCAMP for astrometric calibration.

        Performs astrometric calibration using SCAMP, either jointly on all images
        or individually. Supports parallel processing for individual mode.

        Args:
            files: Paths to astrometrically solved FITS files
            joint: Whether to process all images together
        """
        self.logger.info(f"Start {'joint' if joint else 'individual'} scamp")
        self.logger.debug(MemoryMonitor.log_memory_usage)

        presex_cats = [os.path.splitext(s)[0] + f".{prefix}.cat" for s in files]

        # use local astrefcat if tile obs
        obj = ObservationData(files[0]).obj
        if re.match(r"T\d{5}", obj):
            astrefcat = os.path.join(self.config.path.path_astrefcat, f"{obj}.fits")
            self.config.path.path_astrefcat = astrefcat

        # scamp
        if joint:
            # write target files into a text file
            cat_to_scamp = os.path.join(
                self.config.path.path_factory, "scamp_input.cat"
            )
            with open(cat_to_scamp, "w") as f:
                for precat in presex_cats:
                    f.write(f"{precat}\n")

            # @ is astromatic syntax.
            external.scamp(cat_to_scamp, local_astref=astrefcat)

        elif self.queue:
            self._submit_task(external.scamp, presex_cats, local_astref=astrefcat)

        else:
            for precat in presex_cats:
                external.scamp(precat, local_astref=astrefcat)
                self.logger.info(f"Completed scamp for {precat}]")  # fmt:skip
                self.logger.debug(f"{precat}")  # fmt:skip

        self.logger.info(f"Completed scamp for {self.config.name}")
        self.logger.debug(MemoryMonitor.log_memory_usage)

    def update_header(
        self,
        files: List[str],
        inims: List[str],
        links: List[str],
        use_missfits: bool = False,
        prefix: str = "prep",
    ) -> None:
        """Update WCS information in FITS headers.

        Updates WCS solutions in original FITS files using either missfits
        or manual header updates through custom utility functions.

        Args:
            files: Paths to solved FITS files
            inims: Paths to original input images
            links: Paths to symbolic links (need for use_missfits)
            use_missfits: Whether to use missfits for updates
        """
        self.logger.info(
            f"Updating header {'with missfits' if use_missfits else 'manually'}"
        )
        solved_heads = [os.path.splitext(s)[0] + f".{prefix}.head" for s in files]

        # header update
        if use_missfits:
            for solved_head, output, inim in zip(solved_heads, links, inims):
                output_head = "_".join(output.split("_")[:-1]) + ".head"
                os.symlink(solved_head, output_head)  # factory/inim.head
                external.missfits(
                    output
                )  # soft_link changes to a wcs-updated fits file
                os.system(f"mv {output} {inim}")  # overwrite (inefficient)
        else:
            from .utils import read_header, update_padded_header

            # update img in processed directly
            for solved_head, target_fits in zip(solved_heads, inims):
                # update_scamp_head(target_fits, head_file)
                solved_head = read_header(solved_head)
                update_padded_header(target_fits, solved_head)

        self.logger.info("Header WCS Updated.")

    def _submit_task(self, func: callable, items: List[Any], **kwargs: Any) -> None:
        """Submit tasks to the queue manager for parallel processing.

        Handles task submission and monitoring for parallel processing operations.
        Automatically assigns appropriate priorities and resource requirements.

        Args:
            func: Function to execute
            items: Items to process
            **kwargs: Additional arguments for the function
        """

        task_ids = []

        for i, item in enumerate(items):
            if type(item) is not tuple:
                item = (item,)

            task_id = self.queue.add_task(
                func,
                args=item,
                kwargs=kwargs,
                priority=Priority.MEDIUM,
                gpu=False,
                task_name=f"{func.__name__}_{i}",  # Dynamic task name
            )
            task_ids.append(task_id)

        self.queue.wait_until_task_complete(task_ids)


# ad-hoc
def astrometry_single(file, ahead=None):
    from .utils import read_header, update_padded_header

    solved_file = external.solve_field(file)

    outcat = external.sextractor(
        solved_file, prefix="prep", sex_args=["-catalog_type", "fits_ldac"]
    )

    solved_head = external.scamp(outcat, ahead=ahead)

    update_padded_header(file, read_header(solved_head))

    outcat = external.sextractor(solved_file, prefix="main")

    return outcat
