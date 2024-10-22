# %%
import os
from astropy.io import fits
from astropy.table import Table
from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
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
# radius = 1.4
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
skygrid_table = Table.read(f'{path_skygrid}/skygrid_tiles_7DS.csv')
radeg = skygrid_table['ra']
decdeg = skygrid_table['dec']

# 참조 카탈로그 파일 생성
ctr = 0
# %%
failist = []

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
			width=radius * u.deg,
			#	Gaia DR3 only
			catalog=['I/355/gaiadr3'], cache=False
		)

		if len(result) > 0:
			#	Save as FITS_LDAC
			outbl = result[0]
			outbl['OBSDATE'] = obsdate
			save_as_fits_ldac(outbl, tablename)
		else:
			failist.append(tile)

print(f"Done")
print(f"Fail List {len(failist)}:")
print(f"{failist}")