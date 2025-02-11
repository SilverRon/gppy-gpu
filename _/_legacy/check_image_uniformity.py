# %%
import os, glob, sys
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
import numpy as np
from astropy.table import Table, vstack, hstack
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
warnings.filterwarnings("ignore")

# %%
# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl
#
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# %%
plt.style.use('dark_background') # Dark 모드 스타일 적용

# %%
# inim = '../data/spss/LTT7987_deep/Calib-7DT01-LTT7987-20231015-022405-r-400.com.fits'
# inim = '../data/spss/LTT7987_deep/Calib-7DT01-LTT7987-20231015-022405-r-400.com.fits'
inim = input('IMAGE:')
data, hdr = fits.getdata(inim, header=True)
data.shape

# %% [markdown]
# # Sky Sigma

# %%
pixscale = 0.505

ysize, xsize = data.shape
xsize/4, ysize/4

# xstep = xsize/8
# ystep = ysize/4

nx = 16
ny = 8

xstep = xsize/nx
ystep = ysize/ny

print(f"{xstep*pixscale/60:.3f} arcmin", f"{ystep*pixscale/60:.3f} arcmin")

# %%
bins = np.arange(10, 50+1, 1)


skysig_data = np.zeros_like(data)

for xx in range(nx):
	for yy in range(ny):

		x1 = int(xstep*(xx+1))
		x0 = int(xstep*xx)
		y1 = int(ystep*(yy+1))
		y0 = int(ystep*yy)

		_data = data[y0:y1,x0:x1]
		skysig_data[y0:y1,x0:x1] = np.std(_data[(_data<50) & (_data>10)])

		plt.hist(data[y0:y1,x0:x1].flatten(), bins=bins, histtype='step', lw=3, alpha=0.5)


# %%
# plt.hist(data[y0:y1,x0:x1].flatten(), bins=bins)


# %%
plt.close('all')
fig = plt.figure(figsize=(8, 4))
plt.title(os.path.basename(inim))
plt.imshow(skysig_data, cmap='bwr',)#, vmin=depth-0.2, vmax=depth+0.2)
cbar = plt.colorbar()
cbar.set_label('Sky Sigma')
plt.xlabel('X_IMAGE')
plt.ylabel('Y_IMAGE')
plt.tight_layout()
plt.show()

