# %%
import os
from astropy.io import fits
from astropy.table import Table
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 500
plt.rc("font", family="serif")
# %%
def save_as_fits_ldac(outbl, tablename):
    """
    데이터를 FITS_LDAC 포맷으로 저장하는 함수.
    
    Parameters:
    outbl (Table): 저장할 데이터 테이블 (Astropy Table 형식)
    tablename (str): 저장할 파일 이름
    
    Returns:
    None
    """
    # FITS_LDAC 형식으로 저장하기 위해 BINTABLE 형식으로 변환
    hdu_list = fits.HDUList()

    # 기본 Primary HDU 생성 (데이터 없이 헤더만 포함)
    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # BINTABLE로 변환하여 'LDAC_OBJECTS' 이름을 부여
    bintable_hdu = fits.BinTableHDU(outbl.as_array(), name='LDAC_OBJECTS')
    hdu_list.append(bintable_hdu)

    # HDUList로 FITS_LDAC 포맷으로 저장
    hdu_list.writeto(tablename, overwrite=True)

# %%
# 탐색 반경 (deg 단위)
# 9576*0.5, 6388*0.5
# x1: (1.330, 0.887)
# x2: (2.660, 1.774)
# x3: (3.990 2.662)
# x4: (5.320 3.549)
# width = 2.5
# height = 1.5
# width = 4.0
# height = 2.7
# width = 1.4
# height = 0.8
width = 3.0
height = 2.0
radius = 1.0

columns_to_query = [
	# 'RAJ2000', 'DEJ2000',
	#
	#	Coordinate
	'RA_ICRS', 'DE_ICRS', 
	'e_RA_ICRS', 'e_DE_ICRS', 
	#	Proper Motion
	'pmRA', 'pmDE', 
	'e_pmRA', 'e_pmDE', 
	#	Magnitude
	'Gmag', 'e_Gmag',
]

obsdate = 2016.0

path_ref_scamp = '/large_data/factory/ref_scamp'
path_skygrid = '/large_data/factory/skygrid'

# 입력 데이터 파일에서 RA 및 Dec 값 읽기
skygrid_table = Table.read(f'{path_skygrid}/skygrid.fits')
radeg = skygrid_table['ra']
decdeg = skygrid_table['dec']

# 참조 카탈로그 파일 생성
ctr = 0
# %%
failist = []
skygrid_table['query_width'] = width
skygrid_table['query_height'] = height

skygrid_table['n_scamp_source'] = 0

for tt, (tile, ra, dec) in enumerate(zip(skygrid_table['tile'], skygrid_table['ra'], skygrid_table['dec'])):
	print(f"[{tt:0>5}/{len(skygrid_table):0>5}] {tile}")

	#	Output Name
	tablename = f"{path_ref_scamp}/{tile}.fits"

	if not os.path.exists(tablename):
		#	Coord. Object
		c_query = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
		
		#	Query Gaia DR3
		v = Vizier(columns=columns_to_query)
		v.ROW_LIMIT = -1  # 행 수 제한 없음
		v.TIMEOUT = 600  # 타임아웃 시간 설정

		result = v.query_region(
			c_query,
			width=width * u.deg,
			height=height * u.deg,
			# radius=radius * u.deg,
			#	Gaia DR3 only
			catalog=['I/355/gaiadr3'], cache=False
		)

		if len(result) > 0:
			#	Save as FITS_LDAC
			outbl = result[0]
			outbl['OBSDATE'] = obsdate
			save_as_fits_ldac(outbl, tablename)
			#	Check plot
			n_source = len(outbl)
			skygrid_table['n_scamp_source'][tt] = n_source

			plt.close('all')
			plt.plot(outbl['RA_ICRS'], outbl['DE_ICRS'], '.', alpha=0.1, c='silver', label=f'n={n_source:,}')
			#
			# margin = 0.1
			# indx_fit = np.where(
			# 	(outbl['RA_ICRS'] <  margin+np.max([skygrid_table[f'ra{ii}'] for ii in [1,2,3,4]])) &
			# 	(outbl['RA_ICRS'] > -margin+np.min([skygrid_table[f'ra{ii}'] for ii in [1,2,3,4]])) &
			# 	(outbl['DE_ICRS'] <  margin+np.max([skygrid_table[f'dec{ii}'] for ii in [1,2,3,4]])) &
			# 	(outbl['DE_ICRS'] > -margin+np.min([skygrid_table[f'dec{ii}'] for ii in [1,2,3,4]]))
			# )
			# select_outbl = outbl[indx_fit]
			# plt.plot(select_outbl['RA_ICRS'], select_outbl['DE_ICRS'], '.', alpha=0.1, c='dodgerblue', label=f'selected')
			#
			ra1, dec1 = skygrid_table['ra1'][tt], skygrid_table['dec1'][tt]
			ra2, dec2 = skygrid_table['ra2'][tt], skygrid_table['dec2'][tt]
			ra3, dec3 = skygrid_table['ra3'][tt], skygrid_table['dec3'][tt]
			ra4, dec4 = skygrid_table['ra4'][tt], skygrid_table['dec4'][tt]

			plt.plot(ra, dec, '+', zorder=999, ms=15, c='r', label=f'Center = ({ra:.1f}d,{dec:.1f}d)')
			plt.plot(ra1, dec1, 's', zorder=999, ms=10, c='tomato')
			plt.plot(ra2, dec2, 's', zorder=999, ms=10, c='tomato')
			plt.plot(ra3, dec3, 's', zorder=999, ms=10, c='tomato')
			plt.plot(ra4, dec4, 's', zorder=999, ms=10, c='tomato')

			plt.title(f"{tile} (w={width},h={height})")
			xl, xr = plt.xlim()
			plt.xlim(xr, xl)
			plt.xlabel('RA [deg]')
			plt.ylabel('Dec [deg]')
			plt.legend(loc='upper center', framealpha=1.0)
			plt.tight_layout()
			plt.savefig(f"{tablename.replace('fits', 'png')}", dpi=100)
		else:
			failist.append(tile)

skygrid_table.write(f"{path_ref_scamp}/tile_summary.csv", format='csv', overwrite=True)

print(f"Done")
print(f"Fail List {len(failist)}:")
print(f"{failist}")


#%%

