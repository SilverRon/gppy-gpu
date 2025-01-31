# %%
# Image Subtraction Routine
import numpy as np
from astropy.io import fits

def create_mask_images(input_image, mask_suffix="mask.fits"):
    data = fits.getdata(input_image)
    mask = np.zeros_like(data, dtype=int)
    mask[data == 0] = 1
    mask[data != 0] = 0
    mask_filename = input_image.replace("fits", mask_suffix)
    fits.writeto(mask_filename, mask.astype(np.int8), overwrite=True)
    return mask_filename

# 사용 예
sci_mask = create_mask_images(inim)
ref_mask = create_mask_images(refim)
