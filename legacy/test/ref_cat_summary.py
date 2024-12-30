from astropy.table import Table
from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

GAIA_ROOT_DIR = '/lyman/data1/factory/catalog/gaia_dr3_7DT'

tile_list = sorted(glob(f'{GAIA_ROOT_DIR}/T*fits'))

mag_list = [str(int(i)) for i in np.linspace(9, 25, 17)]
column_names = ['tile', 'total_num', 'checksum', 'nan_count'] + mag_list + ['wmean_mag']
summary_df = pd.DataFrame(columns=column_names)
summary_df['tile'] = tile_list
summary_df['checksum'] = 0
summary_df['wmean_mag'] = 0
for i, _tile in enumerate(tile_list):
    print(_tile, end='\r')
    f = Table.read(_tile).to_pandas()
    mag = f['phot_g_mean_mag'][f['phot_g_mean_mag'] != np.nan]
    del f
    summary_df.loc[i, 'total_num'] = len(mag)
    for j in mag_list:
        if j == '9':
            summary_df.loc[i, j] = len(mag[mag < int(j)])
            summary_df.loc[i, 'checksum'] = summary_df.loc[i, 'checksum'] + summary_df.loc[i, j]
        else:
            summary_df.loc[i, j] = len(mag[( (mag >= int(j)) & (mag < (int(j)+1)))])
            summary_df.loc[i, 'checksum'] = summary_df.loc[i, 'checksum'] + summary_df.loc[i, j]
        summary_df.loc[i, 'wmean_mag'] = summary_df.loc[i, 'wmean_mag'] + summary_df.loc[i, j] / (int(j)+1) ** 2
    del mag
    
summary_df['nan_count'] = summary_df['total_num'] - summary_df['checksum']

print(summary_df)
summary_df.to_csv('ref_cat_summary_test.csv')
# summary_tbl = Table.from_pandas(summary_df)
# Table.write(output='ref_cat_summary_test.fits', table=summary_tbl, overwrite=True)
