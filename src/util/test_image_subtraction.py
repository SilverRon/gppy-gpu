# %%
# Image Subtraction Routine
#============================================================
#	Library
#------------------------------------------------------------
import os
import glob
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np
#============================================================
#	Function
#------------------------------------------------------------
#------------------------------------------------------------
def create_ds9_region_file(ra_array, dec_array, radius, filename="ds9_regions.reg"):
	"""
	RA, Dec 배열과 반경을 입력으로 받아 DS9 region 파일을 생성하는 함수.
	
	Parameters:
	- ra_array: RA 좌표 배열
	- dec_array: Dec 좌표 배열
	- radius: 원의 반경 (단위: arcsec)
	- filename: 생성될 DS9 region 파일의 이름
	"""
	# Region 파일 시작 부분에 필요한 헤더
	header = 'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5'
	
	# 파일 쓰기 시작
	with open(filename, "w") as file:
		file.write(header + "\n")
		
		# 각 좌표에 대한 원 형태의 region 추가
		for ra, dec in zip(ra_array, dec_array):
			region_line = f"circle({ra},{dec},{radius}\")\n"
			file.write(region_line)

	print(f"DS9 region file '{filename}' has been created.")

# def mask_vaild_area():
#============================================================
# %%
#	Path
#------------------------------------------------------------
path_data = '/large_data/processed_1x1_gain2750'
path_ref = '/large_data/factory/ref_frame/r'
#============================================================
#	Setting
#------------------------------------------------------------
obj = 'T09614'
filte = 'r'
tel = '7DT03'
path_sci = f'{path_data}/{obj}/{tel}/{filte}'
#
aperture_suffix = 'AUTO'
snr_lower = 10
classstar_lower = 0.5
flags_upper = 0
sep_upper = 1.
region_radius = 10
#============================================================
# %%
#	Data
#------------------------------------------------------------
#	Science
#------------------------------------------------------------
inim = f'{path_sci}/calib_7DT03_T09614_20240423_020757_r_360.com.fits'
inhdr = fits.getheader(inim)
infilter = inhdr['FILTER']
ingain = inhdr['EGAIN']
inskyval = inhdr['SKYVAL']
inskysig = inhdr['SKYSIG']
incat = inim.replace("fits", "phot.cat")
intbl = Table.read(incat, format='ascii')
#------------------------------------------------------------
#	Select substamp sources
#------------------------------------------------------------
indx_input_select = np.where(
	(intbl[f'SNR_{aperture_suffix}_{infilter}'] > snr_lower) &
	(intbl[f'CLASS_STAR'] > classstar_lower) &
	(intbl['FLAGS'] <= flags_upper)
)
selected_intbl = intbl[indx_input_select]
c_in = SkyCoord(selected_intbl['ALPHA_J2000'], selected_intbl['DELTA_J2000'], unit='deg')
print(f"{len(selected_intbl)} selected from {len(intbl)} ({len(selected_intbl)/len(intbl):.1%})")
#------------------------------------------------------------
# %%
#	Reference
#------------------------------------------------------------
refim = f'{path_ref}/ref_PS1_T09614_00000000_000000_r_0.fits'
refhdr = fits.getheader(refim)
# reffilter = refhdr['FILTER']
# refgain = refhdr['EGAIN']
refcat = refim.replace('fits', 'phot.cat')
reftbl = Table.read(refcat, format='ascii')
refskyval = np.median(reftbl['BACKGROUND'])
refskysig = np.std(reftbl['BACKGROUND'])
c_ref = SkyCoord(reftbl['ALPHA_J2000'], reftbl['DELTA_J2000'], unit='deg')
#------------------------------------------------------------
# %%
#============================================================
#	Mask Image
#------------------------------------------------------------
# Plot presetting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Jupyter Setting
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')
# %%
fig = plt.figure(figsize=(20, 4))
#	Science Image Mask
indata = fits.getdata(inim)
inmask = indata.copy()

indx_in_masked = (inmask == 0)
indx_in_no_masked = (inmask != 0)

inmask[indx_in_masked] = 1
inmask[indx_in_no_masked] = 0
inmask = inmask.astype(np.int8)
plt.subplot(131)
plt.imshow(inmask, vmin=0, vmax=1)
# plt.colorbar()
plt.title("SCI")

refdata = fits.getdata(refim)
refmask = refdata.copy()

refdx_ref_masked = (refmask == 0)
refdx_ref_no_masked = (refmask != 0)

refmask[refdx_ref_masked] = 1
refmask[refdx_ref_no_masked] = 0
refmask = refmask.astype(np.int8)
plt.subplot(132)
plt.imshow(refmask, vmin=0, vmax=1)
plt.yticks([])
plt.title("REF")
# mask = np.zeros_like(inmask)
plt.subplot(133)
mask = np.logical_or.reduce([inmask, refmask])
plt.imshow(mask, vmin=0, vmax=1)
plt.colorbar()
plt.title("MASK")
plt.yticks([])
plt.tight_layout()
#------------------------------------------------------------
#	File save
#------------------------------------------------------------
inmask_image = inim.replace("fits", "mask.fits")
fits.writeto(inmask_image, inmask, overwrite=True)
refmask_image = refim.replace("fits", "mask.fits")
fits.writeto(refmask_image, refmask, overwrite=True)

# %%
#	Select substamp sources
#------------------------------------------------------------
reftbl[f'SNR_{aperture_suffix}'] = reftbl[f'FLUX_{aperture_suffix}'] / reftbl[f'FLUXERR_{aperture_suffix}']

indx_ref_select = np.where(
	(reftbl[f'SNR_{aperture_suffix}'] > snr_lower) &
	(reftbl[f'CLASS_STAR'] > classstar_lower) &
	(reftbl['FLAGS'] <= flags_upper)
)
selected_reftbl = reftbl[indx_ref_select]
c_ref = SkyCoord(selected_reftbl['ALPHA_J2000'], selected_reftbl['DELTA_J2000'], unit='deg')
print(f"{len(selected_reftbl)} selected from {len(reftbl)} ({len(selected_reftbl)/len(reftbl):.1%})")

# %%
#	Matching
# indx_match, sep_match, _ = c_ref.match_to_catalog_sky(c_in)
indx_match, sep_match, _ = c_in.match_to_catalog_sky(c_ref)
matched_table = selected_intbl[sep_match.arcsec < sep_upper]
print(f"{len(matched_table)} sources matched")
ssf = f"{path_sci}/{os.path.basename(inim).replace('fits', 'ssf.txt')}"
f = open(ssf, "w")
for x, y in zip(matched_table['X_IMAGE'], matched_table['Y_IMAGE']):
	f.write(f"{x} {y}\n")
f.close()

# %%
#	Output
outim = f"{os.path.dirname(inim)}/hd{os.path.basename(inim)}"
convim = f"{os.path.dirname(inim)}/hc{os.path.basename(inim)}"
print(f"Output Image   : {outim}")
print(f"Convolved Image: {convim}")

# %%
#	Setting
n_sigma = 5
# il, iu = 0, 60000
# tl, tu = 0, 60000
# il, iu = inskyval - n_sigma * inskysig, inskyval + n_sigma * inskysig
# tl, tu = refskyval - n_sigma * refskysig, refskyval + n_sigma * refskysig
il, iu = inskyval - n_sigma * inskysig, 60000
##	Template
# tl, tu = refskyval - n_sigma * refskysig, 60000
# tl, tu = refskyval - n_sigma * refskysig, 60000000
tl, tu = -60000000, 60000000
##	Region Split (y, x = 6800, 10200)
nrx, nry = 3, 2
#	Run
com = (
	f"hotpants -c t -n i "
	f"-iu {iu} -il {il} -tu {tu} -tl {tl} "
	f"-inim {inim} -tmplim {refim} -outim {outim} -oci {convim} "
	f"-imi {inmask_image} -tmi {refmask_image} "
	f"-v 0 "
	f"-nrx {nrx} -nry {nry} "
	f"-ssf {ssf}"
)
print(com)
os.system(com)

# %%
ds9region_file = f"{path_sci}/{os.path.basename(inim).replace('fits', 'ssf.region.reg')}"
create_ds9_region_file(
	ra_array=matched_table['ALPHA_J2000'], 
	dec_array=matched_table['DELTA_J2000'], 
	radius=region_radius, 
	filename=f"{ds9region_file}"
	)

# ds9com = f"ds9 {inim} {convim} {outim} -tile column -frame lock wcs -region {ds9region_file}&"
ds9com = f"ds9 -tile column -frame lock wcs {inim} -region load {ds9region_file} {convim} -region load {ds9region_file} {outim} -region load {ds9region_file} &"

print(ds9com)

# %%
totmask_image = outim.replace("fits", "mask.fits")
mask = mask.astype(np.int8)
fits.writeto(totmask_image, mask, overwrite=True)

# %%
#	Subt
hddata, hdhdr = fits.getdata(outim, header=True)
new_hddata = hddata * (~mask + 2)
fits.writeto(outim, new_hddata, header=hdhdr, overwrite=True)
#	Conv
hcdata, hchdr = fits.getdata(convim, header=True)
new_hcdata = hcdata * (~mask + 2)
fits.writeto(convim, new_hcdata, header=hchdr, overwrite=True)
