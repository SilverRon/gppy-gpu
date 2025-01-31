# %%
from astropy.table import Table
from astropy.coordinates import SkyCoord
import sys
import tqdm
import os 
#	gpPy
sys.path.append('..')
sys.path.append('/home/gp/gppy')
from util import query

def query_reference_catalog(ra, dec, obj, path_calibration_field, path_refcat):
	ref_gaiaxp_synphot_cat = f'{path_refcat}/gaiaxp_dr3_synphot_{obj}.csv'

	if not os.path.exists(ref_gaiaxp_synphot_cat):
		reftbl = query.merge_catalogs(
			target_coord=SkyCoord(ra, dec, unit='deg'),
			path_calibration_field=path_calibration_field,
			matching_radius=1.5, 
			path_save=ref_gaiaxp_synphot_cat,
		)
		reftbl.write(ref_gaiaxp_synphot_cat, overwrite=True)
	else:
		try:
			reftbl = Table.read(ref_gaiaxp_synphot_cat)
		except Exception as e:
			print(f"{obj}: {e}")

			reftbl = query.merge_catalogs(
				target_coord=SkyCoord(ra, dec, unit='deg'),
				path_calibration_field=path_calibration_field,
				matching_radius=1.5, 
				path_save=ref_gaiaxp_synphot_cat,
			)
			reftbl.write(ref_gaiaxp_synphot_cat, overwrite=True)

	return reftbl

# %%
# Path
path_skygrid = "/large_data/factory/skygrid"
path_calibration_field = "/large_data/Calibration/7DT-Calibration/output/Calibration_Tile"
path_refcat = "/large_data/factory/ref_cat"

skygrid_table = Table.read(f"{path_skygrid}/displaycenter.txt", format='csv')
skygrid_table['name'] = [f"T{idname:0>5}" for idname in skygrid_table['#id']]
# %%
# Object
# ra = 123.456  # RA 값을 적절히 설정
# dec = -12.3456  # Dec 값을 적절히 설정
# obj = "ObjectName"  # Object 이름을 적절히 설정

for nn, (obj, ra, dec) in enumerate(zip(skygrid_table["name"], skygrid_table["ra"], skygrid_table["dec"])):
    print(f"[{nn}/{len(skygrid_table)}]: {obj} ({ra}, {dec})")
    reference_table = query_reference_catalog(ra, dec, obj, path_calibration_field, path_refcat)
