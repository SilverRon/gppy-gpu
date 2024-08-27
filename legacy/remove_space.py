import glob
from astropy.io import fits

imlist = sorted(glob.glob('*SDSS*.fits'))

for inim in imlist:
    # FITS 파일 열기
    with fits.open(inim, mode='update') as hdul:
        # 헤더에 접근
        hdr = hdul[0].header

        # 특정 헤더 값을 변경 (예: 'EXPTIME' 키의 값을 100.0으로 설정)
        # hdr['EXPTIME'] = 100.0
        _obj = hdr['OBJECT']
        obj = _obj.replace(' ', '')
        hdr['OBJECT'] = obj

        # 변경 사항 저장 및 파일 닫기
        hdul.flush()

        # 파일이 자동으로 닫히고, 변경 사항이 저장됩니다.

