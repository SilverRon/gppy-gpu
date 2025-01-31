from .config import Configuration
from .preprocess import Calibration
from .astrometry import Astrometry
from .photometry import Photometry
from .logging import logger
from .preprocess.masterframe import MasterFrameGenerator

def run_masterframe_generator(unit, date, gain, n_binning, queue=True):
    with MasterFrameGenerator(unit, date, gain, n_binning, queue=queue) as master:
        master.run()

def run_pipeline(unit, date, obj, **kwargs):
    with Configuration(unit=unit, date=date, obj=obj, **kwargs) as config:
        calib = Calibration(config)
        calib.run()

        astrm = Astrometry(config)
        astrm.run()

        phot = Photometry(config)
        phot.run()
