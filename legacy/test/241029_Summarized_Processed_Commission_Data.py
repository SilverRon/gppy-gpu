# %%
# Python Library
import os
import glob
import sys
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.table import vstack
from astropy.table import hstack
from ccdproc import ImageFileCollection
import warnings
warnings.filterwarnings("ignore")

# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Jupyter Setting
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
# %%
{
    "u": "blue",
    "g": "green",
    "r": "tomato",
    "i": "crimson",
    "z": "purple",
    "m400": [
        0.2265282583621684,
        0.4938869665513264,
        0.7224913494809688,
        1.0
    ],
    "m412": [
        0.2080738177623993,
        0.5467128027681664,
        0.7356401384083044,
        1.0
    ],
    "m425": [
        0.2560553633217993,
        0.6002306805074971,
        0.7134948096885814,
        1.0
    ],
    "m437": [
        0.30403690888119955,
        0.6537485582468281,
        0.6913494809688582,
        1.0
    ],
    "m450": [
        0.35201845444059976,
        0.7072664359861591,
        0.669204152249135,
        1.0
    ],
    "m462": [
        0.4,
        0.7607843137254902,
        0.6470588235294118,
        1.0
    ],
    "m475": [
        0.46366782006920426,
        0.7856978085351788,
        0.6461361014994234,
        1.0
    ],
    "m487": [
        0.5273356401384084,
        0.8106113033448674,
        0.6452133794694349,
        1.0
    ],
    "m500": [
        0.5910034602076126,
        0.835524798154556,
        0.6442906574394464,
        1.0
    ],
    "m512": [
        0.6546712802768171,
        0.8604382929642448,
        0.6433679354094579,
        1.0
    ],
    "m525": [
        0.7114186851211075,
        0.8832756632064592,
        0.6348327566320646,
        1.0
    ],
    "m537": [
        0.765859284890427,
        0.9054209919261824,
        0.623760092272203,
        1.0
    ],
    "m550": [
        0.8202998846597465,
        0.9275663206459055,
        0.6126874279123413,
        1.0
    ],
    "m562": [
        0.874740484429066,
        0.9497116493656287,
        0.6016147635524798,
        1.0
    ],
    "m575": [
        0.9134948096885814,
        0.9653979238754326,
        0.6140715109573243,
        1.0
    ],
    "m587": [
        0.9365628604382931,
        0.9746251441753172,
        0.6500576701268744,
        1.0
    ],
    "m600": [
        0.9596309111880047,
        0.9838523644752019,
        0.6860438292964245,
        1.0
    ],
    "m612": [
        0.9826989619377164,
        0.9930795847750865,
        0.7220299884659749,
        1.0
    ],
    "m625": [
        0.9997693194925029,
        0.9928489042675894,
        0.7370242214532872,
        1.0
    ],
    "m637": [
        0.9988465974625145,
        0.9642445213379469,
        0.689042675893887,
        1.0
    ],
    "m650": [
        0.9979238754325259,
        0.9356401384083044,
        0.6410611303344866,
        1.0
    ],
    "m662": [
        0.9970011534025375,
        0.907035755478662,
        0.5930795847750865,
        1.0
    ],
    "m675": [
        0.996078431372549,
        0.8784313725490196,
        0.5450980392156862,
        1.0
    ],
    "m687": [
        0.9951557093425606,
        0.8322952710495963,
        0.5063437139561706,
        1.0
    ],
    "m700": [
        0.9942329873125721,
        0.7861591695501728,
        0.46758938869665495,
        1.0
    ],
    "m712": [
        0.9933102652825836,
        0.7400230680507497,
        0.42883506343713945,
        1.0
    ],
    "m725": [
        0.9923875432525952,
        0.6938869665513263,
        0.3900807381776239,
        1.0
    ],
    "m737": [
        0.9859284890426759,
        0.6373702422145328,
        0.35963091118800455,
        1.0
    ],
    "m750": [
        0.9776239907727797,
        0.5773933102652824,
        0.33194925028835054,
        1.0
    ],
    "m762": [
        0.9693194925028835,
        0.5174163783160323,
        0.30426758938869664,
        1.0
    ],
    "m775": [
        0.9610149942329873,
        0.45743944636678197,
        0.2765859284890427,
        1.0
    ],
    "m787": [
        0.942560553633218,
        0.4057670126874279,
        0.2682814302191465,
        1.0
    ],
    "m800": [
        0.9139561707035756,
        0.36239907727797,
        0.27935409457900806,
        1.0
    ],
    "m812": [
        0.8853517877739331,
        0.3190311418685121,
        0.29042675893886966,
        1.0
    ],
    "m825": [
        0.8567474048442907,
        0.2756632064590542,
        0.30149942329873125,
        1.0
    ],
    "m837": [
        0.8226066897347175,
        0.22906574394463664,
        0.30680507497116494,
        1.0
    ],
    "m850": [
        0.7718569780853518,
        0.17277970011534027,
        0.2948096885813149,
        1.0
    ],
    "m862": [
        0.7211072664359862,
        0.11649365628604381,
        0.28281430219146486,
        1.0
    ],
    "m875": [
        0.6703575547866205,
        0.0602076124567474,
        0.27081891580161477,
        1.0
    ],
    "m887": [
        0.6196078431372549,
        0.00392156862745098,
        0.25882352941176473,
        1.0
    ]
}

sdss_effective_wavelength = {
	"u": 354.3,  # Å
	"g": 477.0,  # Å
	"r": 623.1,  # Å
	"i": 762.5,  # Å
	"z": 913.4   # Å
}

# %%
path_data = '/large_data/Commission'
columns_to_pick = [
    # "file", "naxis", "naxis1", "naxis2", "mjd-obs", "ctype1", "cunit1", "crval1", 
    # "file", "naxis", "naxis1", "naxis2", "ctype1", "cunit1", "crval1", 
    "file",
    # "crpix1", "cd1_1", "cd1_2", "ctype2", "cunit2", "crval2", "crpix2", "cd2_1", 
    # "crpix1", "cd1_1", "cd1_2", 
    # "cd1_1", "cd1_2", 
    "cd1_2", 
    # "cd2_2", "exptime", "gain", "saturate", "date", "object", "egain", "filter", 
    "cd2_2", "exptime", "gain", "date", "object", "egain", "filter", 
    "date-obs", "date-loc", "exposure", "centalt", "centaz", "airmass", "mjd", 
    "jd", "seeing", "peeing", "ellip", "elong", "skysig", "skyval", "refcat", 
    "maglow", "magup", "stdnumb", "auto", "aper", "aper_1", "aper_2", "aper_3", 
    "aper_4", "aper_5", "zp_auto", "ezp_auto", "ul3_auto", "ul5_auto", "zp_0", 
    "ezp_0", "ul3_0", "ul5_0", "zp_1", "ezp_1", "ul3_1", "ul5_1", "zp_2", "ezp_2", 
    "ul3_2", "ul5_2", "zp_3", "ezp_3", "ul3_3", "ul5_3", "zp_4", "ezp_4", "ul3_4", 
    "ul5_4", "zp_5", "ezp_5", "ul3_5", "ul5_5",
]
print(f"{len(columns_to_pick)} columns selected")
# %%
obj = 'PSZ2G227.44-31.24'
images = sorted(glob.glob(f"{path_data}/{obj}/*/calib*com.fits"))
ic = ImageFileCollection(filenames=images)
intbl = ic.summary
# %%
intbl['lam'] = 0.
intbl['filter_type'] = ' '*10
for ff, filte in enumerate(intbl['filter']):
	if 'm' in filte:
		lam = float(filte[1:])
		ftyp = 'med'
	else:
		lam = sdss_effective_wavelength[filte]
		ftyp = 'brd'
	intbl['lam'][ff] = lam
	intbl['filter_type'][ff] = ftyp

medtbl = intbl[intbl['filter_type'] == 'med']
brdtbl = intbl[intbl['filter_type'] == 'brd']

# %%
plt.scatter(medtbl['lam'], medtbl['ul5_1'], c=medtbl['seeing'], vmin=1.0, vmax=2.0, ec='k')
plt.errorbar(medtbl['lam'], medtbl['ul5_1'], xerr=[25/2]*len(medtbl), c='k', ls='none', zorder=0)

plt.scatter(brdtbl['lam'], brdtbl['ul5_1'], c=brdtbl['seeing'], vmin=1.0, vmax=2.0, ec='k', marker='s')

plt.title(obj)

cbar = plt.colorbar()
cbar.set_label("""Seeing ["]""")
plt.xlim(375, 900)
yl, yu = plt.ylim()
plt.ylim([yu, yl])
plt.xlabel("Wavelengh [nm]")
plt.ylabel(r"Depth [5$\sigma$]")
plt.tight_layout()
# %%
