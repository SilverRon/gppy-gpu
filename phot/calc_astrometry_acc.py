import os, glob, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
#	Astropy
from astropy.table import Table, vstack, hstack
from astropy.io import ascii
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
#
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

radius = 0.8 # 0.799 deg

# inim = './calib_7DT01_NGC0253_20231010_082341_g_600.com.fits'
# incat = './calib_7DT01_NGC0253_20231010_082341_g_600.com.phot.cat'
inim = './calib_7DT02_LTT1020_20231028_033909_g_100.com.fits'
incat = './calib_7DT02_LTT1020_20231028_033909_g_100.com.phot.cat'


# obj = incat.split('_')[2]
with fits.open(inim) as hdul:
	header = hdul[0].header

obj = header['object']
ra = header['ra']
dec = header['dec']
# matching_radius = header['seeing']
matching_radius = 5.0

print(f"OBJECT: {obj}")
print(f"RA    : {ra:.3f} deg")
print(f"Dec   : {dec:.3f} deg")
print(f"MatchR: {matching_radius:.3f} deg")

#	Input Catalog
_intbl = Table.read(incat, format='ascii')
intbl = _intbl[_intbl['FLAGS']==0]

#	Reference Catalog for Astrometry Accuracy Calculation
path_refast = '/large_data/factory/ref_cat/astrometry'
refallcat = f"{path_refast}/gaiadr3_lite_radec.fits"

refastcatlist = sorted(glob.glob(f"{path_refast}/*{obj}*.fits"))
if len(refastcatlist) == 0:
	refalltbl = Table.read(refallcat)
else:
	reftbl = Table.read(refastcatlist[0])

reftbl = refalltbl[
	#	RA
	(refalltbl['ra']>ra-radius) &
	(refalltbl['ra']<ra+radius) &
	#	Dec
	(refalltbl['dec']>dec-radius) &
	(refalltbl['dec']<dec+radius) 
]

plt.plot(intbl['ALPHA_J2000'], intbl['DELTA_J2000'], '.')
plt.plot(reftbl['ra'], reftbl['dec'], '.')


c_sex = SkyCoord(intbl['ALPHA_J2000'], intbl['DELTA_J2000'], unit='deg')
c_ref = SkyCoord(reftbl['ra'], reftbl['dec'], unit='deg')
indx_match, sep, _ = c_sex.match_to_catalog_sky(c_ref)
_mtbl = hstack([intbl, reftbl[indx_match]])
_mtbl['sep'] = sep.arcsec

mtbl = _mtbl[_mtbl['sep']<matching_radius]

plt.close()
plt.hist(sep.arcsec, bins=np.arange(0, 10+0.25, 0.25))
plt.axvline(x=matching_radius, ls='--', c='tomato', lw=3)
plt.axvline(x=np.median(mtbl['sep']), ls='-', c='orange', lw=3)
plt.xlabel('sep [arcsec]')
plt.ylabel('#')
# plt.show()

#
fig = plt.figure()
plt.axis('equal')

# radiff = np.abs(mtbl['ALPHA_J2000']-mtbl['ra'])*3600
radiff = (mtbl['ALPHA_J2000']-mtbl['ra'])*3600
rasep_med = np.median(radiff)
# decdiff = np.abs(mtbl['DELTA_J2000']-mtbl['dec'])*3600
decdiff = (mtbl['DELTA_J2000']-mtbl['dec'])*3600
decsep_med = np.median(decdiff)
plt.plot(radiff, decdiff, '.')

plt.axvline(x=rasep_med, c='tomato', ls='--', lw=3, zorder=0, label=f"""RA SEP: {rasep_med:.3f}" """)
plt.axhline(y=decsep_med, c='tomato', ls='--', lw=3, zorder=0, label=f"""DEC SEP: {decsep_med:.3f}" """)
#	0,0 grid
plt.axvline(x=0, c='k', ls='-', lw=0.5, zorder=0)
plt.axhline(y=0, c='k', ls='-', lw=0.5, zorder=0)
#	Setting
plt.xlim([-2.5, 2.5])
plt.ylim([-2.5, 2.5])
plt.xlabel("""RA Difference ["]""")
plt.ylabel("""Dec Difference ["]""")
plt.legend()
plt.tight_layout()

fig = plt.figure(figsize=(8, 6))
plt.axis('equal')
# plt.scatter(mtbl['ALPHA_J2000'], mtbl['DELTA_J2000'], c=mtbl['sep'], vmin=0.0, vmax=5.0, alpha=0.5)
plt.hexbin(mtbl['ALPHA_J2000'], mtbl['DELTA_J2000'], mtbl['sep'], gridsize=10, vmin=0.0, vmax=5.0)
cbar = plt.colorbar()
cbar.set_label("""sep ["]""")
plt.xlabel("""RA [deg]""")
plt.ylabel("""Dec [deg]""")
xl, xr = plt.xlim()
plt.xlim([xr, xl])