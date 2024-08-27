from astropy.io import fits
from unidecode import unidecode
import os

inim = '/large_data/factory/scamp_test/single/calib_7DT01_LTT1020_20231011_060822_g_10.fits'
inhead = inim.replace('fits', 'head')
outcat = inim.replace('fits', 'cat')
conf_scamp = '/large_data/factory/scamp_test/single/7dt.scamp'

sexcom = f"source-extractor -c simple.sex {inim} -CATALOG_NAME {outcat}"
print(sexcom)
os.system(sexcom)

scampcom = f"scamp -c {conf_scamp} {outcat} -HEADER_NAME {inhead}"
print(scampcom)
os.system(scampcom)