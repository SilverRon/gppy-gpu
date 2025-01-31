import os, subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import datetime

# astropy
from astropy.table import Table, hstack
from astropy.table import MaskedColumn
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

# gppy moduels
from phot import gpphot
from util import query
from .utils import query_sextractor
from .logging import logger

class Photometry:
    def __init__(self, config):
        """
        Initialize the Photometry class.
        config: Config file
        """
        self.config = config
        self.path_sex = self.config.file.config_sex_photometry
        self.path_config = self.config.file.config_photometry
        
        
    def run(self):
        # logger.info("Photometry started.")
        # Get image list from config
        self.image = self.config.file.processed_files
        for im in self.image:
            self.define_info(im)
            reftbl = self.pre_sex(im)
            zptbl, setbl, indx_match, sep = self.do_photometry(im, reftbl)
            self.calculate_zp(zptbl, setbl)
            self.update_header(im, zptbl, reftbl, setbl, indx_match, sep)
            
    
    def define_info(self, inim):
        self.n_binning = self.config.obs.n_binning
        self.pixscale = self.config.obs.pixscale * self.n_binning
        
        #------------------------------------------------------------
        #	INFO. from file name
        #------------------------------------------------------------
        self.hdr = fits.getheader(inim)
        part = os.path.basename(inim).split('_')
        self.head = inim.replace('.fits', '')

        self.obs = part[1]
        self.obj = self.hdr['OBJECT']
        self.filte = self.hdr['FILTER']
        self.dateobs = self.hdr['DATE-OBS']
        self.refmagkey = f"mag_{self.filte}"
        self.refmagerkey = f"magerr_{self.filte}"
        timeobj = Time(self.dateobs, format='isot')
        self.jd = timeobj.jd
        self.mjd = timeobj.mjd
        
        self.gain = self.hdr['EGAIN']
        self.xcent, self.ycent= self.hdr['NAXIS1']/2., self.hdr['NAXIS2']/2.
        w = WCS(inim)
        self.racent, self.decent = w.all_pix2world(self.xcent, self.ycent, 1)
        self.racent, self.decent = self.racent.item(), self.decent.item()
        #------------------------------------------------------------
        self.cat = f"{self.head}.cat"
        #------------------------------------------------------------
        #	SExtractor Configuration
        #------------------------------------------------------------
        self.sex_conf = os.path.join(self.config.path.path_reference, 'sex')
        # SExtractor Configuration for pre SExtarctor
        self.conf_simple, self.param_simple, self.nnw_simple, self.conv_simple = query_sextractor(self.sex_conf, 'simple')
        
        # SExtractor Configuration for photometry
        self.param, self.conv, self.nnw, self.conf = query_sextractor(self.sex_conf, 'gregoryphot')

        self.phot_conf = self.config.photometry
        
        # logger.info('Photometry configuration is set.')
    
    
    def pre_sex(self, inim):
        #------------------------------------------------------------
        #	DATE-OBS, JD
        #------------------------------------------------------------
        ref_gaiaxp_synphot_cat = f'{self.config.path.path_refcat}/gaiaxp_dr3_synphot_{self.obj}.csv'
        if not os.path.exists(ref_gaiaxp_synphot_cat):
            reftbl = query.merge_catalogs(
                target_coord=SkyCoord(self.racent, self.decent, unit='deg'),
                path_calibration_field=self.config.path.path_calibration_field,
                matching_radius=1.5, path_save=ref_gaiaxp_synphot_cat,
                )
            reftbl.write(ref_gaiaxp_synphot_cat, overwrite=True)
        else:
            reftbl = Table.read(ref_gaiaxp_synphot_cat)
            
        precat = f"{self.head}.pre.cat"
        presexcom = f"source-extractor -c {self.conf_simple} {inim} -FILTER_NAME {self.conv_simple} -STARNNW_NAME {self.nnw_simple} -PARAMETERS_NAME {self.param_simple} -CATALOG_NAME {precat}"
        print(presexcom)
        # os.system(presexcom)  Commented out for logging
        os.system(phot_util.log2tmp(presexcom, "presex"))  # stderr is logged with stdout
        
        #------------------------------------------------------------
        #  Matching
        #------------------------------------------------------------
        pretbl = Table.read(precat, format='ascii.sextractor')
        pretbl['within_ellipse'] = phot_util.is_within_ellipse(
            pretbl['X_IMAGE'], pretbl['Y_IMAGE'], self.xcent, self.ycent, 
            self.phot_conf.frac*self.hdr['NAXIS1']/2, 
            self.phot_conf.frac*self.hdr['NAXIS2']/2
            )
        
        c_pre = SkyCoord(pretbl['ALPHA_J2000'], pretbl['DELTA_J2000'], unit='deg')
        c_ref = SkyCoord(reftbl['ra'], reftbl['dec'], unit='deg')
        
        indx_match, sep, _ = c_pre.match_to_catalog_sky(c_ref)
        _premtbl = hstack([pretbl, reftbl[indx_match]])
        _premtbl['sep'] = sep.arcsec
        matching_radius = 1.
        premtbl = _premtbl[_premtbl['sep']<matching_radius]
        premtbl['within_ellipse'] = phot_util.is_within_ellipse(
            premtbl['X_IMAGE'], premtbl['Y_IMAGE'], self.xcent, self.ycent, 
            self.phot_conf.frac*self.hdr['NAXIS1']/2, 
            self.phot_conf.frac*self.hdr['NAXIS2']/2
            )

        indx_star4seeing = np.where(
            #	Star-like Source
            # (premtbl['CLASS_STAR']>0.9) &
            (premtbl['FLAGS']==0) &
            #	Within Ellipse
            (premtbl['within_ellipse'] == True) &
            #
            # (premtbl['C_term']<2) &
            # (premtbl['ruwe']<1.4) &
            # (premtbl['phot_variable_flag']!='VARIABLE') &
            # (premtbl['ipd_frac_multi_peak']<7) &
            # (premtbl['ipd_frac_odd_win']<7) &
            #
            (premtbl[self.refmagkey]>11.75) &
            (premtbl[self.refmagkey]<18.0)
        )
        self.ellipticity = np.median(premtbl['ELLIPTICITY'][indx_star4seeing])
        self.elongation = np.median(premtbl['ELONGATION'][indx_star4seeing])
        self.seeing = np.median(premtbl['FWHM_WORLD'][indx_star4seeing]*3600)

        print(f"-"*60)
        print(f"{len(premtbl[indx_star4seeing])} Star-like Sources Found")
        print(f"-"*60)
        print(f"SEEING     : {self.seeing:.3f} arcsec")
        print(f"ELONGATION : {self.elongation:.3f}")
        print(f"ELLIPTICITY: {self.ellipticity:.3f}")

        return reftbl


    def do_photometry(self, inim, reftbl):
        #------------------------------------------------------------
        #	APERTURE SETTING
        #------------------------------------------------------------
        self.peeing = self.seeing/self.pixscale
        #	Aperture Dictionary
        self.aperture_dict = {
            'MAG_AUTO'  : (0., 'MAG_AUTO DIAMETER [pix]'),
            'MAG_APER'  : (2*0.6731*self.peeing, 'BEST GAUSSIAN APERTURE DIAMETER [pix]'),
            'MAG_APER_1': (2*self.peeing, '2*SEEING APERTURE DIAMETER [pix]'),
            'MAG_APER_2': (3*self.peeing, '3*SEEING APERTURE DIAMETER [pix]'),
            'MAG_APER_3': (3/self.pixscale, """FIXED 3" APERTURE DIAMETER [pix]"""),
            'MAG_APER_4': (5/self.pixscale, """FIXED 5" APERTURE DIAMETER [pix]"""),
            'MAG_APER_5': (10/self.pixscale, """FIXED 10" APERTURE DIAMETER [pix]"""),
        }

        self.add_aperture_dict = {}
        for key in list(self.aperture_dict.keys()):
            self.add_aperture_dict[key.replace('MAG_', '')] = (round(self.aperture_dict[key][0], 3), self.aperture_dict[key][1])
        #	MAG KEY
        self.inmagkeys = list(self.aperture_dict.keys())
        #	MAG ERROR KEY
        inmagerkeys = [key.replace('MAG_', 'MAGERR_') for key in self.inmagkeys]
        #	Aperture Sizes
        aperlist = [self.aperture_dict[key][0] for key in self.inmagkeys[1:]]

        PHOT_APERTURES = ','.join(map(str, aperlist))
        #------------------------------------------------------------
        #	SOURCE EXTRACTOR CONFIGURATION FOR PHOTOMETRY
        #------------------------------------------------------------
        
        param_insex = dict(	#------------------------------
                            #	CATALOG
                            #------------------------------
                            CATALOG_NAME = self.cat,
                            #------------------------------
                            #	CONFIG FILES
                            #------------------------------
                            CONF_NAME = self.conf,
                            PARAMETERS_NAME = self.param,
                            FILTER_NAME = self.conv,    
                            STARNNW_NAME = self.nnw,
                            #------------------------------
                            #	EXTRACTION
                            #------------------------------			
                            # PSF_NAME = psf,
                            DETECT_MINAREA = self.phot_conf.DETECT_MINAREA,
                            DETECT_THRESH = self.phot_conf.DETECT_THRESH,
                            DEBLEND_NTHRESH = self.phot_conf.DEBLEND_NTHRESH,
                            DEBLEND_MINCONT = self.phot_conf.DEBLEND_MINCONT,
                            #------------------------------
                            #	PHOTOMETRY
                            #------------------------------
                            #	DIAMETER
                            #	OPT.APER, (SEEING x2), x3, x4, x5
                            #	MAG_APER	OPT.APER
                            #	MAG_APER_1	OPT.GAUSSIAN.APER
                            #	MAG_APER_2	SEEINGx2
                            #	...
                            PHOT_APERTURES = PHOT_APERTURES,
                            SATUR_LEVEL  = '65000.0',
                            # GAIN = str(gain.value),
                            GAIN = str(self.gain),
                            PIXEL_SCALE = str(self.pixscale),
                            #------------------------------
                            #	STAR/GALAXY SEPARATION
                            #------------------------------
                            SEEING_FWHM = str(2.0),
                            #------------------------------
                            #	BACKGROUND
                            #------------------------------
                            BACK_SIZE = self.phot_conf.BACK_SIZE,
                            BACK_FILTERSIZE = self.phot_conf.BACK_FILTERSIZE,
                            BACKPHOTO_TYPE = self.phot_conf.BACKPHOTO_TYPE,
                            #------------------------------
                            #	CHECK IMAGE
                            #------------------------------
                            # CHECKIMAGE_TYPE = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND',
                            # CHECKIMAGE_NAME = '{},{},{},{}'.format(seg, aper, bkg, sub),
                            )
        #	Add Weight Map from SWarp
        weightim = inim.replace("com", "weight")
        if "com" in inim:
            if os.path.exists(weightim):
                param_insex['WEIGHT_TYPE'] = "MAP_WEIGHT"
                param_insex['WEIGHT_IMAGE'] = weightim
        #	Check Image
        if self.phot_conf.check == True:
            param_insex['CHECKIMAGE_TYPE'] = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND'
            param_insex['CHECKIMAGE_NAME'] = f'{self.head}.seg.fits,{self.head}.aper.fits,{self.head}.bkg.fits,{self.head}.sub.fits'
        else:
            pass

        
        #------------------------------------------------------------
        #	PHOTOMETRY
        #------------------------------------------------------------
        
        com = gpphot.sexcom(inim, param_insex)
        t0_sex = time.time()
        print(com)
        sexout = subprocess.getoutput(com)
        delt_sex = time.time() - t0_sex
        print(f"SourceEXtractor: {delt_sex:.3f} sec")
        line = [s for s in sexout.split('\n') if 'RMS' in s]
        self.skymed, self.skysig = float(line[0].split('Background:')[1].split('RMS:')[0]), float(line[0].split('RMS:')[1].split('/')[0])
        # os.system(f'rm {seg} {aper} {bkg} {sub}'.format(seg, aper, bkg, sub))

        setbl = Table.read(self.cat, format='ascii.sextractor')

        #------------------------------------------------------------
        #	Matching
        #------------------------------------------------------------

        # logger.info("Matching sources with reference catalog.")
        #	Proper Motion Correction
        # 관측 시점을 Astropy Time 객체로 변환
        obs_time = Time(self.dateobs, format='isot', scale='utc')

        # 기준 에포크 (Gaia DR3는 J2016.0)
        epoch_gaia = Time(2016.0, format='jyear')

        # reftbl의 ra, dec, pmra, pmdec, parallax를 가져옴
        ra = reftbl['ra']  # 단위: degrees
        dec = reftbl['dec']  # 단위: degrees
        pmra = reftbl['pmra']  # 단위: mas/yr
        pmdec = reftbl['pmdec']  # 단위: mas/yr
        parallax = reftbl['parallax']  # 단위: mas

        # NaN 또는 None인 값을 확인하여, 고유 운동 정보가 없는 경우 처리
        # pmra, pmdec, parallax가 NaN인 경우는 None으로 처리
        pmra = np.where(np.isnan(pmra), None, pmra)
        pmdec = np.where(np.isnan(pmdec), None, pmdec)
        parallax = np.where(np.isnan(parallax), None, parallax)

        # SkyCoord 객체 생성 (고유 운동이 없는 소스는 None으로 처리됨)
        c_ref = SkyCoord(ra=ra*u.deg,
                        dec=dec*u.deg,
                        pm_ra_cosdec=pmra*u.mas/u.yr if pmra is not None else None,
                        pm_dec=pmdec*u.mas/u.yr if pmdec is not None else None,
                        distance=(1/(parallax*u.mas)) if parallax is not None else None,
                        obstime=epoch_gaia)  # 기준 에포크를 J2016.0으로 지정

        # 관측 시점에 맞춰 고유 운동 보정
        c_ref_corrected = c_ref.apply_space_motion(new_obstime=obs_time)

        # 이제 이 좌표를 이용해 기존 좌표와 매칭
        c_sex = SkyCoord(setbl['ALPHA_J2000'], setbl['DELTA_J2000'], unit='deg')

        # 매칭 수행
        indx_match, sep, _ = c_sex.match_to_catalog_sky(c_ref_corrected)

        # SourceEXtractor Catalog + Reference Catalog
        _mtbl = hstack([setbl, reftbl[indx_match]])
        _mtbl['sep'] = sep.arcsec
        mtbl = _mtbl[_mtbl['sep']<self.phot_conf.matching_radius]
        mtbl['within_ellipse'] = phot_util.is_within_ellipse(mtbl['X_IMAGE'], mtbl['Y_IMAGE'], self.xcent, self.ycent, self.frac*self.hdr['NAXIS1']/2, self.frac*self.hdr['NAXIS2']/2)
        # logger.info(f"""Matched Sources: {len(mtbl):_} (r={matching_radius:.3f}")""")

        for _, inmagkey in enumerate(self.inmagkeys):
            suffix = inmagkey.replace("MAG_", "")
            mtbl[f"SNR_{suffix}"] = mtbl[f'FLUX_{suffix}'] / mtbl[f'FLUXERR_{suffix}']

        indx_star4zp = np.where(
            #	Star-like Source
            # (mtbl['CLASS_STAR']>0.9) &
            (mtbl['FLAGS']==0) &
            #	Within Ellipse
            (mtbl['within_ellipse'] == True) &
            #	SNR cut
            (mtbl['SNR_AUTO'] > 20) &
            #	Magnitude in Ref. Cat 
            # (mtbl[f'{refmagkey}']<refmagupper) &
            # (mtbl[f'{refmagkey}']>refmaglower) &
            # (mtbl[f'{refmagerkey}']<refmaglower)
            #
            # (mtbl[refmagkey]>11.75) &
            (mtbl[self.refmagkey] > self.refmaglower)# &
            # (mtbl[refmagkey]<18.0)
        )

        zptbl = mtbl[indx_star4zp]

        # logger.info(f"{len(zptbl)} sources to calibration ZP")
        return zptbl, setbl, indx_match, sep
    
    
    def calculate_zp(self, zptbl, setbl):
        #------------------------------------------------------------
        #	ZEROPOINT CALCULATION
        #------------------------------------------------------------
        # logger.info("Calculating zero points.")
        
        for _, inmagkey in enumerate(self.inmagkeys):
            inmagerrkey = inmagkey.replace("MAG", 'MAGERR')

            sigma=2.0

            zparr = zptbl[self.refmagkey]-zptbl[inmagkey]
            # zperrarr = tool.sqsum(zptbl[inmagerrkey], zptbl[refmagerkey])
            #	Temperary zeropoint error!!!!!!
            zperrarr = phot_util.sqsum(zptbl[inmagerrkey], np.zeros_like(len(zptbl)))

            zparr_clipped = sigma_clip(
                zparr,
                sigma=sigma,
                maxiters=None,
                cenfunc=np.median,
                copy=False
                )

            indx_alive = np.where( zparr_clipped.mask == False )
            indx_exile = np.where( zparr_clipped.mask == True )

            #	RE-DEF. ZP LIST AND INDEXING CLIPPED & NON-CLIPPED
            zptbl_alive = zptbl[indx_alive]
            zptbl_exile = zptbl[indx_exile]

            zp, zperr = phot_util.compute_median_mad(zparr[indx_alive])

            self.plot_zp(inmagkey, zp, zperr, zptbl, zparr, zperrarr, zptbl_alive, zptbl_exile)
            
            self.apply_zp(inmagkey, inmagerrkey, zp, zperr, setbl)
            
            # logger.info(f"{inmagkey} ZP: {zp:.3f}+/-{zperr:.3f}")
        
        
    def plot_zp(self, inmagkey, zp, zperr, zptbl, zparr, zperrarr, zptbl_alive, zptbl_exile):
        # plt.errorbar(zptbl[refmagkey], zparr, xerr=zptbl[refmagerkey], yerr=zperrarr, ls='none', c='grey', alpha=0.5)
        plt.errorbar(zptbl[self.refmagkey], zparr, xerr=0, yerr=zperrarr, ls='none', c='grey', alpha=0.5)
        plt.plot(zptbl_alive[self.refmagkey], zptbl_alive[self.refmagkey]-zptbl_alive[inmagkey], '.', c='dodgerblue', alpha=0.75, zorder=999, label=f'{len(zptbl_alive)}')
        plt.plot(zptbl_exile[self.refmagkey], zptbl_exile[self.refmagkey]-zptbl_exile[inmagkey], 'x', c='tomato', alpha=0.75, label=f'{len(zptbl_exile)}')
        plt.axhline(y=zp, ls='-', lw=1, c='grey', zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}")
        plt.axhspan(ymin=zp-zperr, ymax=zp+zperr, color='silver', alpha=0.5, zorder=0)
        plt.xlabel(self.refmagkey)
        # plt.xlim([8, 16])
        # plt.xlim([refmaglower-0.5, refmagupper+0.5])
        plt.axvspan(xmin=0, xmax=self.refmaglower, color='silver', alpha=0.25, zorder=0)
        plt.axvspan(xmin=self.refmagupper, xmax=25, color='silver', alpha=0.25, zorder=0)
        plt.xlim([10, 20])
        plt.ylim([zp-0.25, zp+0.25])
        plt.ylabel(f'ZP_{inmagkey}')
        plt.legend(loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"{self.head}.{inmagkey}.png", dpi=100)


    def apply_zp(self, inmagkey, inmagerrkey, zp, zperr, setbl):
        #	Apply ZP
        ##	MAG
        _calmagkey = f"{inmagkey}_{self.filte}"
        _calmagerrkey = f"{inmagerrkey}_{self.filte}"
        ##	FLUX
        _calfluxkey = _calmagkey.replace('MAG', 'FLUX')
        _calfluxerrkey = _calmagerrkey.replace('MAG', 'FLUX')
        ##  SNR
        _calsnrkey = _calmagkey.replace('MAG', 'SNR')

        setbl[_calmagkey] = setbl[inmagkey]+zp
        setbl[_calmagerrkey] = phot_util.sqsum(setbl[inmagerrkey], zperr)

        #	Flux [uJy]
        setbl[_calfluxkey] = (setbl[_calmagkey].data*u.ABmag).to(u.uJy).value
        # setbl[_calfluxerrkey] = setbl[_calfluxkey] * (10**(-0.4 * setbl[inmagerrkey]) - 1)
        # setbl[_calfluxerrkey] = compute_flux_density_error(magerr=setbl[_calmagerrkey], flux_density=setbl[_calfluxkey])
        setbl[_calfluxerrkey] = 0.4*np.log(10)*setbl[_calfluxkey]*setbl[_calmagerrkey]

        ## SNR
        setbl[_calsnrkey] = setbl[_calfluxkey]/setbl[_calfluxerrkey]


        #	Formatting
        setbl[_calmagkey].format = '.3f'
        setbl[_calmagerrkey].format = '.3f'
        setbl[_calfluxkey].format = '.3f'
        setbl[_calfluxerrkey].format = '.3f'

        #	Depth Calculation
        aperture_size = self.aperture_dict[inmagkey][0]
        if inmagkey == 'MAG_AUTO':
            ul_3sig = 0.0
            ul_5sig = 0.0
        else:
            ul_3sig = gpphot.limitmag(3, zp, aperture_size, self.skysig)
            ul_5sig = gpphot.limitmag(5, zp, aperture_size, self.skysig)


        #	Header keyword
        if inmagkey == 'MAG_AUTO':
            _zpkey = inmagkey.replace('MAG', 'ZP')
            _zperrkey = inmagerrkey.replace('MAGERR', 'EZP')
            _ul3key = inmagkey.replace('MAG', 'UL3')
            _ul5key = inmagkey.replace('MAG', 'UL5')
        elif inmagkey == 'MAG_APER':
            _zpkey = inmagkey.replace('MAG', 'ZP').replace('APER', '0')
            _zperrkey = inmagerrkey.replace('MAGERR', 'EZP').replace('APER', '0')
            _ul3key = inmagkey.replace('MAG', 'UL3').replace('APER', '0')
            _ul5key = inmagkey.replace('MAG', 'UL5').replace('APER', '0')
        else:
            _zpkey = inmagkey.replace('MAG', 'ZP').replace('APER_', '')
            _zperrkey = inmagerrkey.replace('MAGERR', 'EZP').replace('APER_', '')
            _ul3key = inmagkey.replace('MAG', 'UL3').replace('APER_', '')
            _ul5key = inmagkey.replace('MAG', 'UL5').replace('APER_', '')


        _zp_dict = {
            _zpkey: (round(zp, 3), f'ZERO POINT for {inmagkey}'),
            _zperrkey: (round(zperr, 3), f'ZERO POINT ERROR for {inmagkey}'),
            _ul3key: (round(ul_3sig, 3), f'3 SIGMA LIMITING MAG FOR {inmagkey}'),
            _ul5key: (round(ul_5sig, 3), f'5 SIGMA LIMITING MAG FOR {inmagkey}'),
        }

        # _zp_dict

        self.header_to_add.update(_zp_dict)
        return setbl

    def update_header(self, inim, zptbl, reftbl, setbl, indx_match, sep):
        #------------------------------------------------------------
        #	Header
        #------------------------------------------------------------
        self.header_to_add = {
            'AUTHOR': ('Gregory S.H. Paek', 'PHOTOMETRY AUTHOR'),
            'PHOTIME': (datetime.date.today().isoformat(), 'PHTOMETRY TIME [KR]'),
            #	Time
            'JD': (self.jd, 'Julian Date of the observation'),
            'MJD': (self.mjd, 'Modified Julian Date of the observation'),
            #	Image Definition
            'SEEING': (round(self.seeing, 3), 'SEEING [arcsec]'),
            'PEEING': (round(self.peeing, 3), 'SEEING [pixel]'),
            'ELLIP': (round(self.ellipticity, 3), 'ELLIPTICITY 1-B/A [0-1]'),
            'ELONG': (round(self.elongation, 3), 'ELONGATION A/B [1-]'),
            'SKYSIG': (round(self.skysig, 3), 'SKY SIGMA VALUE'),
            'SKYVAL': (round(self.skymed, 3), 'SKY MEDIAN VALUE'),
            #	Reference Source Conditions for ZP
            'REFCAT': (self.phot_conf.refcatname, 'REFERENCE CATALOG NAME'),
            'MAGLOW': (self.phot_conf.refmaglower, 'REF MAG RANGE, LOWER LIMIT'),
            'MAGUP': (self.phot_conf.refmagupper, 'REF MAG RANGE, UPPER LIMIT'),
            'STDNUMB': (len(zptbl), '# OF STD STARS TO CALIBRATE ZP'),
        }

        self.header_to_add.update(self.add_aperture_dict)

        #------------------------------------------------------------
        #	ADD HEADER INFO
        #------------------------------------------------------------
        with fits.open(inim, mode='update') as hdul:
            header = hdul[0].header
            for key, (value, comment) in self.header_to_add.items():
                header[key] = (value, comment)
            hdul.flush()

        #	Add Reference Catalog Information
        keys_from_refcat = ['source_id', 'bp_rp', 'phot_g_mean_mag', f'mag_{self.filte}']

        # 각 키에 대해 매칭된 값들을 처리
        for key in keys_from_refcat:
            valuearr = reftbl[key][indx_match].data  # 매칭된 값들 추출

            # MaskedColumn을 사용해 매칭 반경을 넘는 경우 마스킹 처리
            masked_valuearr = MaskedColumn(valuearr, mask=(sep.arcsec > self.phot_conf.matching_radius))
            
            # 결과를 setbl에 추가 (또는 업데이트)
            setbl[key] = masked_valuearr
        
        #	Meta data
        meta_dict = {
            'obs': self.obs,
            'object': self.obj,
            'filter': self.filte,
            'date-obs': self.hdr['date-obs'],
            'jd': self.jd,
            'mjd': self.mjd,
        }
        setbl.meta = meta_dict
        setbl.write(f'{self.head}.phot.cat', format='ascii.tab', overwrite=True)
        
        # logger.info("Photometry finished.")

    
class phot_util():
    def __init__(self):
        path_thisfile = Path(__file__).resolve()
        self.path_root = path_thisfile.parent.parent.parent  # Careful! not a str / PATH HAS TO BE REVISED


    def log2tmp(self, command, label):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_tmp = self.path_root / 'tmp'  # MODIFICATION REQUIRED
        if not path_tmp.exists():
            path_tmp.mkdir()
        sexlog = str(path_tmp / f"{label}_{timestamp}.log")
        # stderr is logged with stdout
        new_com = f"{command} > {sexlog} 2>&1"
        return new_com
    
    
    def file2dict(path_infile):
        out_dict = dict()
        f = open(path_infile)
        for line in f:
            key, val = line.split()
            out_dict[key] = val
        return out_dict
    
    
    def is_within_ellipse(x, y, center_x, center_y, a, b):
        term1 = ((x - center_x) ** 2) / (a ** 2)
        term2 = ((y - center_y) ** 2) / (b ** 2)
        return term1 + term2 <= 1
    
    
    def weighted_median(values, errors):
        # 오차를 역수로 사용하여 가중치 계산
        weights = 1.0 / np.array(errors)
        # 중앙값 계산
        median = np.median(values)
        # 중앙값과 각 데이터 포인트의 차이 계산
        deviations = np.abs(values - median)
        # 가중치를 곱하여 가중 중앙값 계산
        weighted_median = np.median(deviations * weights)
        return median, weighted_median
    
    
    def compute_median_mad(values):
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return median, mad


    def compute_flux_density_error(magerr, flux_density):
        flux_density_error = (2.5/np.log(10))*(flux_density)*magerr
        # flux_density_error = (1.086)*(flux_density)*magerr
        
        return flux_density_error
    
    
    def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
        """
        Calculate the corrected flux excess factor for the input Gaia EDR3 data.
        
        Parameters
        ----------
        
        bp_rp: float, numpy.ndarray
            The (BP-RP) colour listed in the Gaia EDR3 archive.
        phot_bp_rp_excess_factor: float, numpy.ndarray
            The flux excess factor listed in the Gaia EDR3 archive.
            
        Returns
        -------
        
        The corrected value for the flux excess factor, which is zero for "normal" stars.
        
        Example
        -------
        
        phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
        """
        
        if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
            bp_rp = np.float64(bp_rp)
            phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
        
        if bp_rp.shape != phot_bp_rp_excess_factor.shape:
            raise ValueError('Function parameters must be of the same shape!')
            
        do_not_correct = np.isnan(bp_rp)
        bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
        greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
        redrange = np.logical_not(do_not_correct) & (bp_rp >= 4.0)
        
        correction = np.zeros_like(bp_rp)
        correction[bluerange] = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
        correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) \
            - 0.005879*np.power(bp_rp[greenrange], 3)
        correction[redrange] = 1.057572 + 0.140537*bp_rp[redrange]
        
        return phot_bp_rp_excess_factor - correction

    def sqsum(a, b):
        '''
        SQUARE SUM
        USEFUL TO CALC. ERROR
        '''
        return np.sqrt(a ** 2. + b ** 2.)
