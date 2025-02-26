"""
gpPy: Automated Pipeline for Astronomical Image Processing

gpPy is a multi-threaded pipeline for processing optical and near-infrared (NIR) images from
IMSNG/GECKO and 7DT facilities. It handles data reduction, astrometric calibration, stacking, photometric 
calibration, image subtraction, and automated transient detection using GPU and CPU multiprocessing.

- Developed by Dr. Gregory Peak (2018)  
- First public release: September 1, 2023  
- Major renovation: February 2025 (7DT pipeline integration)  

Current maintainers: Donghwan Hyun, WonHyeong Lee  
Contributors: Dr. Gregory Peak, Dr. Donggeun Tak 

Contact: gregorypaek94_at_g_mail
"""

__package__ = "gppy"
from .version import __version__

import warnings

import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

# Ignore common warnings that are not harmful
warnings.filterwarnings('ignore', message='.*datfix.*')
warnings.filterwarnings('ignore', message='.*pmsafe.*')
warnings.filterwarnings('ignore', message='.*partition.*')