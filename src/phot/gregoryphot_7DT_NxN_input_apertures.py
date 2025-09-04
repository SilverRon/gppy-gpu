# PHOTOMETRY CODE FOR PYTHON 3.X
# CREATED 2020.12.10  Gregory S.H. Paek
# ------------------------------------------------------------
# Revised to support flexible aperture configuration via YAML (CLI > config > default)
# - Cleaned up comments (English)
# - Fixed several runtime issues and minor bugs
# ------------------------------------------------------------

import os, glob, sys, subprocess
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import date
import time
import warnings

# Matplotlib styles
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

# Astropy
from astropy.table import Table, vstack, hstack
from astropy.table import MaskedColumn
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

warnings.filterwarnings('ignore', message="Warning: 'partition' will ignore the 'mask' of the MaskedArray.")
warnings.filterwarnings('ignore', message="Warning: 'partition' will ignore the 'mask' of the MaskedColumn.")

# gpPy package
path_thisfile = Path(__file__).resolve()
path_root = path_thisfile.parent.parent.parent
sys.path.append(str(path_root / 'src'))
from phot import gpphot
from phot import gcurve
from preprocess import calib
from util import query
from util import tool
from util.path_manager import log2tmp

starttime = time.time()

# ============================================================
# Utility Functions
# ============================================================

def load_aperture_yaml(yaml_path, pixscale_arcsec_per_pix):
    """
    Load aperture definitions from YAML and return a dict compatible with the
    internal aperture_dict: { 'MAG_APER_3': (diameter_in_pixels, 'DESC'), ... }.

    Accepted YAML schemas:
    1) Verbose (pixels)
       MAG_APER_3:
         diameter_pix: 8.0
         description: 'FIXED 8 px APERTURE DIAMETER [pix]'

    2) Verbose (arcsec)
       MAG_APER_3:
         diameter_arcsec: 4.0
         description: 'FIXED 4" APERTURE DIAMETER [arcsec]'

    3) Short form (assumed pixels)
       MAG_APER_3: 8.0
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    out = {}
    for key, val in raw.items():
        if not str(key).startswith("MAG_"):
            raise ValueError(f"YAML key must start with 'MAG_': {key}")

        if isinstance(val, (int, float)):
            d_pix = float(val)
            desc = f"FIXED {d_pix:.2f} APERTURE DIAMETER [pix]"
        elif isinstance(val, dict):
            if "diameter_pix" in val:
                d_pix = float(val["diameter_pix"])
                desc = val.get("description", f"FIXED {d_pix:.2f} APERTURE DIAMETER [pix]")
            elif "diameter_arcsec" in val:
                d_arc = float(val["diameter_arcsec"])
                d_pix = d_arc / float(pixscale_arcsec_per_pix)
                desc = val.get("description", f'FIXED {d_arc:.2f}" APERTURE DIAMETER [arcsec]')
            else:
                raise ValueError(f"{key}: need diameter_pix or diameter_arcsec")
        else:
            raise ValueError(f"{key}: unsupported value type {type(val)}")

        # MAG_AUTO is always 0 pix (SExtractor ignores diameter input for AUTO)
        if key == "MAG_AUTO":
            d_pix, desc = 0.0, "MAG_AUTO DIAMETER [pix]"

        out[key] = (d_pix, desc)
    return out


def file2dict(path_infile):
    """Read a simple two-column key value config into a dict."""
    out_dict = {}
    with open(path_infile) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) == 1:
                key, val = parts[0], ''
            else:
                key, val = parts[0], parts[1]
            out_dict[key] = val
    return out_dict


def is_within_ellipse(x, y, center_x, center_y, a, b):
    term1 = ((x - center_x) ** 2) / (a ** 2)
    term2 = ((y - center_y) ** 2) / (b ** 2)
    return term1 + term2 <= 1


def weighted_median(values, errors):
    weights = 1.0 / np.array(errors)
    median = np.median(values)
    deviations = np.abs(values - median)
    wmed = np.median(deviations * weights)
    return median, wmed


def compute_median_mad(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, mad


def compute_flux_density_error(magerr, flux_density):
    # dF = 0.4 * ln(10) * F * dmag  (AB system)
    return 0.4 * np.log(10) * (flux_density) * magerr


def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """Gaia EDR3 flux-excess correction."""
    if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
        bp_rp = np.float64(bp_rp)
        phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)
    if bp_rp.shape != phot_bp_rp_excess_factor.shape:
        raise ValueError('Function parameters must be of the same shape!')

    do_not_correct = np.isnan(bp_rp)
    bluerange  = (~do_not_correct) & (bp_rp < 0.5)
    greenrange = (~do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange   = (~do_not_correct) & (bp_rp >= 4.0)

    correction = np.zeros_like(bp_rp)
    correction[bluerange]  = 1.154360 + 0.033772*bp_rp[bluerange] + 0.032277*np.power(bp_rp[bluerange], 2)
    correction[greenrange] = 1.162004 + 0.011464*bp_rp[greenrange] + 0.049255*np.power(bp_rp[greenrange], 2) - 0.005879*np.power(bp_rp[greenrange], 3)
    correction[redrange]   = 1.057572 + 0.140537*bp_rp[redrange]

    return phot_bp_rp_excess_factor - correction


# ============================================================
# Core Photometry Routine
# ============================================================

def phot_routine(inim):
    # --------------------------------------------------------
    # Basic info from filename / header
    # --------------------------------------------------------
    hdr = fits.getheader(inim)
    part = os.path.basename(inim).split('_')
    head = inim.replace('.fits', '')

    # Saturation level (stacked vs single)
    if 'com.fits' in inim:
        saturation_lv = hdr["SATURATE"] * (1 - saturation_margin)
    else:
        saturation_lv = 60.0 * 1e3

    obs = part[1]
    obj = hdr['OBJECT']
    filte = hdr['FILTER']

    refmagkey   = f"mag_{filte}"
    refmagerkey = f"magerr_{filte}"
    refsnrkey   = f"snr_{filte}"

    print(inim, obs, obj, refmagkey, refmagerkey)

    obsdict  = tool.getccdinfo(obs, path_obs)
    gain     = hdr.get('EGAIN', obsdict.get('gain', 1.0))  # fallback if header lacks EGAIN
    pixscale = obsdict['pixelscale'] * n_binning           # arcsec/pix (likely Quantity)
    fov      = obsdict['fov']

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------
    cat    = f"{head}.cat"
    seg    = f"{head}.seg.fits"
    bkg    = f"{head}.bkg.fits"
    sub    = f"{head}.sub.fits"
    psf    = f"{head}.psf"
    aper   = f"{head}.aper.fits"

    # Config for a pre-detection (simple SExtractor)
    conf_simple  = f"{path_sex_config}/simple_{n_binning}x{n_binning}.sex"
    param_simple = f"{path_sex_config}/simple.param"
    nnw_simple   = f"{path_sex_config}/simple.nnw"
    conv_simple  = f"{path_sex_config}/simple.conv"

    print('-'*60)
    print(inim)
    print(f'{obs}\t{obj} in {refmagkey}')
    print('-'*60)

    # --------------------------------------------------------
    # WCS center and timing
    # --------------------------------------------------------
    hdul = fits.open(inim)
    hdr  = hdul[0].header

    xcent, ycent = hdr['NAXIS1']/2., hdr['NAXIS2']/2.
    w = WCS(inim)
    racent, decent = w.all_pix2world(xcent, ycent, 1)
    racent, decent = racent.item(), decent.item()

    dateobs = hdr['DATE-OBS']
    timeobj = Time(dateobs, format='isot')
    jd  = timeobj.jd
    mjd = timeobj.mjd

    # --------------------------------------------------------
    # Reference catalog (prepared externally)
    # --------------------------------------------------------
    ref_gaiaxp_cat        = f'{path_refcat}/XP_CONTINUOUS_RAW_{obj}.csv'
    ref_gaiaxp_synphot_cat= f'{path_refcat}/gaiaxp_dr3_synphot_{obj}.csv'
    if not os.path.exists(ref_gaiaxp_synphot_cat):
        reftbl = query.merge_catalogs(
            target_coord=SkyCoord(racent, decent, unit='deg'),
            path_calibration_field=path_calibration_field,
            matching_radius=1.5, path_save=ref_gaiaxp_synphot_cat,
        )
        reftbl.write(ref_gaiaxp_synphot_cat, overwrite=True)
    else:
        reftbl = Table.read(ref_gaiaxp_synphot_cat)

    # --------------------------------------------------------
    # Pre SExtractor run (quick detection)
    # --------------------------------------------------------
    precat = f"{head}.pre.cat"
    presexcom_list = [
        "source-extractor",
        f"-c {conf_simple}",
        f"{inim}",
        f"-FILTER_NAME {conv_simple}",
        f"-STARNNW_NAME {nnw_simple}",
        f"-PARAMETERS_NAME {param_simple}",
        f"-CATALOG_NAME {precat}",
        f"-SATUR_LEVEL {saturation_lv}",
        f"-SEEING_FWHM {seeing_assume.value}",
    ]
    presexcom = " ".join(presexcom_list)
    print(presexcom)
    os.system(log2tmp(presexcom, "presex"))  # stderr logged with stdout
    pretbl = Table.read(precat, format='ascii.sextractor')
    pretbl['within_ellipse'] = is_within_ellipse(
        pretbl['X_IMAGE'], pretbl['Y_IMAGE'], xcent, ycent,
        frac*hdr['NAXIS1']/2, frac*hdr['NAXIS2']/2
    )

    # Star-like selection for seeing estimate (matched to reference)
    print('3. MATCHING (pre)')
    c_pre = SkyCoord(pretbl['ALPHA_J2000'], pretbl['DELTA_J2000'], unit='deg')
    c_ref = SkyCoord(reftbl['ra'], reftbl['dec'], unit='deg')
    indx_match, sep, _ = c_pre.match_to_catalog_sky(c_ref)
    _premtbl = hstack([pretbl, reftbl[indx_match]])
    _premtbl['sep'] = sep.arcsec
    matching_radius = 1.0
    premtbl = _premtbl[_premtbl['sep'] < matching_radius]
    premtbl['within_ellipse'] = is_within_ellipse(
        premtbl['X_IMAGE'], premtbl['Y_IMAGE'], xcent, ycent,
        frac*hdr['NAXIS1']/2, frac*hdr['NAXIS2']/2
    )

    idx_seeing = np.where(
        (premtbl['FLAGS'] == 0) &
        (premtbl['within_ellipse'] == True) &
        (premtbl[refmagkey] > 11.75) & (premtbl[refmagkey] < 18.0)
    )
    ellipticity = np.median(premtbl['ELLIPTICITY'][idx_seeing])
    elongation  = np.median(premtbl['ELONGATION'][idx_seeing])
    seeing      = np.median(premtbl['FWHM_WORLD'][idx_seeing] * 3600)

    print("-"*60)
    print(f"{len(premtbl[idx_seeing])} Star-like Sources Found")
    print("-"*60)
    print(f"SEEING     : {seeing:.3f} arcsec")
    print(f"ELONGATION : {elongation:.3f}")
    print(f"ELLIPTICITY: {ellipticity:.3f}")

    # --------------------------------------------------------
    # Aperture setting (default + optional YAML override)
    # --------------------------------------------------------
    peeing = seeing / pixscale.value  # [pix]

    # (A) Default, hard-coded
    aperture_dict = {
        'MAG_AUTO'  : (0., 'MAG_AUTO DIAMETER [pix]'),
        'MAG_APER'  : (30/pixscale.value,  'FIXED 30" APERTURE DIAMETER [pix]'),
        'MAG_APER_1': (15/pixscale.value,  'FIXED 15" APERTURE DIAMETER [pix]'),
        'MAG_APER_2': (20/pixscale.value,  'FIXED 20" APERTURE DIAMETER [pix]'),
        'MAG_APER_3': (3/pixscale.value,   'FIXED 3"  APERTURE DIAMETER [pix]'),
        'MAG_APER_4': (5/pixscale.value,   'FIXED 5"  APERTURE DIAMETER [pix]'),
        'MAG_APER_5': (10/pixscale.value,  'FIXED 10" APERTURE DIAMETER [pix]'),
    }

    # (B) YAML override (CLI > config). Uses global aperture_yaml_path_final
    if aperture_yaml_path_final and os.path.exists(aperture_yaml_path_final):
        try:
            yaml_ap = load_aperture_yaml(aperture_yaml_path_final, pixscale_arcsec_per_pix=pixscale.value)
            aperture_dict.update(yaml_ap)
            print(f"[Aperture] Loaded {len(yaml_ap)} entries from: {aperture_yaml_path_final}")
        except Exception as e:
            print(f"[Aperture][WARN] YAML load failed → fallback to defaults: {e}")
    else:
        if aperture_yaml_path_final:
            print(f"[Aperture][WARN] YAML not found: {aperture_yaml_path_final} → defaults")
        else:
            print("[Aperture] No YAML specified → defaults")

    # (C) Header entries derived from aperture_dict
    add_aperture_dict = {k.replace('MAG_', ''): (v[0], v[1]) for k, v in aperture_dict.items()}

    # (D) SExtractor PHOT_APERTURES string (exclude MAG_AUTO)
    inmagkeys   = list(aperture_dict.keys())
    inmagerkeys = [k.replace('MAG_', 'MAGERR_') for k in inmagkeys]
    aperlist    = [aperture_dict[k][0] for k in inmagkeys if k != 'MAG_AUTO']
    PHOT_APERTURES = ','.join(f"{x:.6f}" for x in aperlist)

    # --------------------------------------------------------
    # SExtractor configuration for photometry
    # --------------------------------------------------------
    # Defaults
    conf  = f"{path_sex_config}/gregoryphot_{n_binning}x{n_binning}.sex"
    param = f"{path_sex_config}/gregoryphot.param"
    conv  = f"{path_sex_config}/gregoryphot.conv"
    nnw   = f"{path_sex_config}/gregoryphot.nnw"
    # Optional overrides from gphot.config
    conf  = gphot_dict.get('sex_conf',  conf)
    param = gphot_dict.get('sex_param', param)
    conv  = gphot_dict.get('sex_conv',  conv)
    nnw   = gphot_dict.get('sex_nnw',   nnw)

    param_insex = dict(
        CATALOG_NAME  = cat,
        CONF_NAME     = conf,
        PARAMETERS_NAME = param,
        FILTER_NAME   = conv,
        STARNNW_NAME  = nnw,
        DETECT_MINAREA= DETECT_MINAREA,
        DETECT_THRESH = DETECT_THRESH,
        DEBLEND_NTHRESH= DEBLEND_NTHRESH,
        DEBLEND_MINCONT= DEBLEND_MINCONT,
        PHOT_APERTURES= PHOT_APERTURES,
        SATUR_LEVEL   = str(saturation_lv),
        GAIN          = str(gain),
        PIXEL_SCALE   = str(pixscale.value),
        SEEING_FWHM   = str(seeing),
        BACK_SIZE     = BACK_SIZE,
        BACK_FILTERSIZE= BACK_FILTERSIZE,
        BACKPHOTO_TYPE= BACKPHOTO_TYPE,
    )

    weightim = inim.replace("com", "weight")
    if ("com" in inim) and os.path.exists(weightim):
        param_insex['WEIGHT_TYPE']  = "MAP_WEIGHT"
        param_insex['WEIGHT_IMAGE'] = weightim

    if check:
        param_insex['CHECKIMAGE_TYPE'] = 'SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND'
        param_insex['CHECKIMAGE_NAME'] = f'{seg},{aper},{bkg},{sub}'

    print('2. SOURCE EXTRACTOR')
    com = gpphot.sexcom(inim, param_insex)
    t0_sex = time.time()
    print(com)
    sexout = subprocess.getoutput(com)
    delt_sex = time.time() - t0_sex
    print(f"SourceEXtractor: {delt_sex:.3f} sec")

    # Robust parse of background / RMS line
    lines = [s for s in sexout.split('\n') if 'RMS' in s and 'Background' in s]
    if not lines:
        raise RuntimeError("Failed to parse SExtractor output for background/RMS.")
    parts = lines[0].split('Background:')
    skymed = float(parts[1].split('RMS:')[0])
    skysig = float(parts[1].split('RMS:')[1].split('/')[0])

    setbl = Table.read(cat, format='ascii.sextractor')

    # --------------------------------------------------------
    # Match SExtractor catalog to reference (with space motion)
    # --------------------------------------------------------
    print('3. MATCHING')

    obs_time   = Time(dateobs, format='isot', scale='utc')
    epoch_gaia = Time(2016.0, format='jyear')  # Gaia DR3 epoch

    # Allow NaNs to pass through; Astropy handles them
    ra       = reftbl['ra']
    dec      = reftbl['dec']
    pmra     = reftbl['pmra']
    pmdec    = reftbl['pmdec']
    parallax = reftbl['parallax']

    c_ref = SkyCoord(
        ra=ra*u.deg, dec=dec*u.deg,
        pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmdec*u.mas/u.yr,
        distance=(1/(parallax*u.mas)),
        obstime=epoch_gaia
    )
    c_ref_corrected = c_ref.apply_space_motion(new_obstime=obs_time)

    c_sex = SkyCoord(setbl['ALPHA_J2000'], setbl['DELTA_J2000'], unit='deg')
    indx_match, sep, _ = c_sex.match_to_catalog_sky(c_ref_corrected)

    _mtbl = hstack([setbl, reftbl[indx_match]])
    _mtbl['sep'] = sep.arcsec
    matching_radius = 1.0
    mtbl = _mtbl[_mtbl['sep'] < matching_radius]
    mtbl['within_ellipse'] = is_within_ellipse(
        mtbl['X_IMAGE'], mtbl['Y_IMAGE'], xcent, ycent,
        frac*hdr['NAXIS1']/2, frac*hdr['NAXIS2']/2
    )
    print(f"Matched Sources: {len(mtbl):_} (r={matching_radius:.3f}")

    for nn, inmagkey in enumerate(inmagkeys):
        suffix = inmagkey.replace("MAG_", "")
        mtbl[f"SNR_{suffix}"] = mtbl[f'FLUX_{suffix}'] / mtbl[f'FLUXERR_{suffix}']

    idx_zp = np.where(
        (mtbl['FLAGS'] == 0) &
        (mtbl['within_ellipse'] == True) &
        (mtbl['SNR_AUTO'] > 20) &
        (mtbl[refmagkey] > refmaglower)
    )
    zptbl = mtbl[idx_zp]
    print(f"{len(zptbl)} sources to calibration ZP")

    # --------------------------------------------------------
    # Header preparation (image quality, sky, refcat info)
    # --------------------------------------------------------
    header_to_add = {
        'AUTHOR': ('Gregory S.H. Paek', 'PHOTOMETRY AUTHOR'),
        'PHOTIME': (date.today().isoformat(), 'PHOTOMETRY TIME [KR]'),
        'JD':   (jd,  'Julian Date of the observation'),
        'MJD':  (mjd, 'Modified Julian Date of the observation'),
        'SEEING': (seeing, 'SEEING [arcsec]'),
        'PEEING': (seeing / pixscale.value, 'SEEING [pixel]'),
        'ELLIP': (ellipticity, 'ELLIPTICITY 1-B/A [0-1]'),
        'ELONG': (elongation,  'ELONGATION A/B [1-]'),
        'SKYSIG': (skysig, 'SKY SIGMA VALUE'),
        'SKYVAL': (skymed, 'SKY MEDIAN VALUE'),
        'REFCAT': (refcatname, 'REFERENCE CATALOG NAME'),
        'MAGLOW': (refmaglower, 'REF MAG RANGE, LOWER LIMIT'),
        'MAGUP':  (refmagupper, 'REF MAG RANGE, UPPER LIMIT'),
        'STDNUMB': (len(zptbl), '# OF STD STARS TO CALIBRATE ZP'),
        'SATLV': (saturation_lv, 'APPLIED SATURATION LEVEL'),
    }
    header_to_add.update(add_aperture_dict)

    # --------------------------------------------------------
    # Zeropoint calculation & application
    # --------------------------------------------------------
    print('4. ZERO POINT CALCULATION')

    for nn, inmagkey in enumerate(inmagkeys):
        inmagerrkey = inmagkey.replace("MAG", 'MAGERR')
        sigma = 2.0

        zparr    = zptbl[refmagkey] - zptbl[inmagkey]
        zperrarr = tool.sqsum(zptbl[inmagerrkey], np.zeros(len(zptbl)))

        zparr_clipped = sigma_clip(
            zparr, sigma=sigma, maxiters=None, cenfunc=np.median, copy=False
        )
        idx_alive = np.where(zparr_clipped.mask == False)
        idx_exile = np.where(zparr_clipped.mask == True)

        zptbl_alive = zptbl[idx_alive]
        zptbl_exile = zptbl[idx_exile]

        zp, zperr = compute_median_mad(zparr[idx_alive])
        print(f"{inmagkey} ZP: {zp:.3f}+/-{zperr:.3f}")

        # Plot (lightweight)
        plt.close()
        plt.errorbar(zptbl[refmagkey], zparr, xerr=0, yerr=zperrarr, ls='none', c='grey', alpha=0.5)
        plt.plot(zptbl_alive[refmagkey], zptbl_alive[refmagkey]-zptbl_alive[inmagkey], '.', c='dodgerblue', alpha=0.75, zorder=999, label=f'{len(zptbl_alive)}')
        plt.plot(zptbl_exile[refmagkey], zptbl_exile[refmagkey]-zptbl_exile[inmagkey], 'x', c='tomato', alpha=0.75, label=f'{len(zptbl_exile)}')
        plt.axhline(y=zp, ls='-', lw=1, c='grey', zorder=1, label=f"ZP: {zp:.3f}+/-{zperr:.3f}")
        plt.axhspan(ymin=zp-zperr, ymax=zp+zperr, color='silver', alpha=0.5, zorder=0)
        plt.xlabel(refmagkey)
        plt.axvspan(xmin=0, xmax=refmaglower, color='silver', alpha=0.25, zorder=0)
        plt.axvspan(xmin=refmagupper, xmax=25, color='silver', alpha=0.25, zorder=0)
        plt.xlim([10, 20])
        plt.ylim([zp-0.25, zp+0.25])
        plt.ylabel(f'ZP_{inmagkey}')
        plt.legend(loc='upper center', ncol=3)
        plt.tight_layout()
        plt.savefig(f"{head}.{inmagkey}.png", dpi=100)

        # Apply ZP to measured magnitudes and convert to flux
        _calmagkey    = f"{inmagkey}_{filte}"
        _calmagerrkey = f"{inmagerrkey}_{filte}"
        _calfluxkey   = _calmagkey.replace('MAG', 'FLUX')
        _calfluxerrkey= _calmagerrkey.replace('MAG', 'FLUX')
        _calsnrkey    = _calmagkey.replace('MAG', 'SNR')

        setbl[_calmagkey]    = setbl[inmagkey] + zp
        setbl[_calmagerrkey] = tool.sqsum(setbl[inmagerrkey], zperr)

        setbl[_calfluxkey]   = (setbl[_calmagkey].data*u.ABmag).to(u.uJy).value
        setbl[_calfluxerrkey]= 0.4*np.log(10)*setbl[_calfluxkey]*setbl[_calmagerrkey]
        setbl[_calsnrkey]    = setbl[_calfluxkey]/setbl[_calfluxerrkey]

        # Limiting magnitude (3σ, 5σ). For AUTO we set zeros.
        aperture_size = aperture_dict[inmagkey][0]
        if inmagkey == 'MAG_AUTO':
            ul_3sig, ul_5sig = 0.0, 0.0
        else:
            ul_3sig = gpphot.limitmag(3, zp, aperture_size, skysig)
            ul_5sig = gpphot.limitmag(5, zp, aperture_size, skysig)

        # Header keywords
        if inmagkey == 'MAG_AUTO':
            _zpkey   = inmagkey.replace('MAG', 'ZP')
            _zperrkey= inmagerrkey.replace('MAGERR', 'EZP')
            _ul3key  = inmagkey.replace('MAG', 'UL3')
            _ul5key  = inmagkey.replace('MAG', 'UL5')
        elif inmagkey == 'MAG_APER':
            _zpkey   = inmagkey.replace('MAG', 'ZP').replace('APER', '0')
            _zperrkey= inmagerrkey.replace('MAGERR', 'EZP').replace('APER', '0')
            _ul3key  = inmagkey.replace('MAG', 'UL3').replace('APER', '0')
            _ul5key  = inmagkey.replace('MAG', 'UL5').replace('APER', '0')
        else:
            _zpkey   = inmagkey.replace('MAG', 'ZP').replace('APER_', '')
            _zperrkey= inmagerrkey.replace('MAGERR', 'EZP').replace('APER_', '')
            _ul3key  = inmagkey.replace('MAG', 'UL3').replace('APER_', '')
            _ul5key  = inmagkey.replace('MAG', 'UL5').replace('APER_', '')

        header_to_add.update({
            _zpkey:   (zp,   f'ZERO POINT for {inmagkey}'),
            _zperrkey:(zperr,f'ZERO POINT ERROR for {inmagkey}'),
            _ul3key:  (ul_3sig, f'3 SIGMA LIMITING MAG FOR {inmagkey}'),
            _ul5key:  (ul_5sig, f'5 SIGMA LIMITING MAG FOR {inmagkey}'),
        })

    # --------------------------------------------------------
    # Write header, attach reference info, and save catalog
    # --------------------------------------------------------
    with fits.open(inim, mode='update') as hdul2:
        header = hdul2[0].header
        for key, (value, comment) in header_to_add.items():
            header[key] = (value, comment)
        hdul2.flush()

    # Add selected reference columns to photometry table
    keys_from_refcat = [
        'source_id','ra','dec','parallax','pmra','pmdec',
        'phot_g_mean_mag','bp_rp', f'mag_{filte}'
    ]
    broad_filters = [f"mag_{b}" for b in ['u','g','r','i','z']]
    for k in broad_filters:
        if k not in keys_from_refcat:
            keys_from_refcat.append(k)

    for key in keys_from_refcat:
        valuearr = reftbl[key][indx_match].data
        setbl[key] = MaskedColumn(valuearr, mask=(sep.arcsec > matching_radius))

    setbl.meta = {
        'obs': obs,
        'object': obj,
        'filter': filte,
        'date-obs': hdr['DATE-OBS'],
        'jd': jd,
        'mjd': mjd,
    }
    setbl.write(f'{head}.phot.cat', format='ascii.tab', overwrite=True)


# ============================================================
# USER SETTINGS / CLI HANDLING
# ============================================================

# Base path (first CLI arg) – kept for backward compatibility
# try:
#     path_base = sys.argv[1]
# except Exception:
#     path_base = '.'

# Paths
path_refcat  = f'/lyman/data1/factory/ref_cat'
path_config  = str(path_root / 'config')
path_to_filterset = f"{path_config}/filterset"
path_obs     = f'{path_config}/obs.dat'
path_gphot   = f'{path_config}/gphot.config'
path_default_gphot = f'{path_config}/gphot.config'
path_calibration_field = \
    "/lyman/data1/Calibration/7DT-Calibration/output/Calibration_Tile"
path_sex_config = "/lyman/data1/3I_ATLAS/config"


print(path_gphot)
if os.path.exists(path_gphot):
    gphot_dict = file2dict(path_gphot)
else:
    gphot_dict = file2dict(path_default_gphot)
    print('There is no gregoryphot configuration. Use default.')

# Read config values
# imkey          = gphot_dict['imkey']
imkey          = sys.argv[1]
refqueryradius = float(gphot_dict['refqueryradius'])
frac           = float(gphot_dict['photfraction'])
refcatname     = gphot_dict['refcatname']
refmaglower    = float(gphot_dict['refmaglower'])
refmagupper    = float(gphot_dict['refmagupper'])
refmagerupper  = float(gphot_dict['refmagerupper'])
inmagerupper   = float(gphot_dict['inmagerupper'])
flagcut        = int(gphot_dict['flagcut'])
check          = (gphot_dict['check'] == 'True')

# Saturation margin (optional)
saturation_margin = float(gphot_dict.get('saturation_margin', 0.08))

# SExtractor background options
DETECT_MINAREA   = gphot_dict['DETECT_MINAREA']
DETECT_THRESH    = gphot_dict['DETECT_THRESH']
DEBLEND_NTHRESH  = gphot_dict['DEBLEND_NTHRESH']
DEBLEND_MINCONT  = gphot_dict['DEBLEND_MINCONT']
BACK_SIZE        = gphot_dict['BACK_SIZE']
BACK_FILTERSIZE  = gphot_dict['BACK_FILTERSIZE']
BACKPHOTO_TYPE   = gphot_dict['BACKPHOTO_TYPE']

seeing_assume = 2.0 * u.arcsecond

# ncore (deprecated)
# try:
#     ncore = int(sys.argv[2])
# except Exception:
#     ncore = 1

try:
    n_binning = int(sys.argv[2])
except Exception:
    n_binning = 1

# Determine image list
if ("@" in imkey):
    print("Input: Use image list.")
    image_list_file = imkey[1:]
    with open(image_list_file, 'r') as f:
        imlist = [line.strip() for line in f if line.strip()]
else:
    print("Input: Wild Card.")
    imlist = sorted(glob.glob(imkey))

print(f'#\t{len(imlist)} images to do photometry')
print('='*60)
for i, img in enumerate(imlist):
    print(f'{i:0>4} {img}')
print('='*60)

# Aperture YAML path decision (CLI > config > None)
aperture_yaml_cli = None
if len(sys.argv) > 3 and sys.argv[3] and sys.argv[3] != "-":
    aperture_yaml_cli = sys.argv[3]

aperture_yaml_cfg = gphot_dict.get('aperture_yaml', None)
aperture_yaml_path_final = aperture_yaml_cli or aperture_yaml_cfg
if aperture_yaml_cli:
    print(f"[Aperture] CLI yaml: {aperture_yaml_cli}")
elif aperture_yaml_cfg:
    print(f"[Aperture] config yaml: {aperture_yaml_cfg}")
else:
    print(f"[Aperture] yaml not specified → using defaults")

# Run photometry
fail_image_list = []
for ii, inim in enumerate(imlist):
    try:
        phot_routine(inim)
    except Exception as e:
        print(f"\nPhotometry for {os.path.basename(inim)} failed!\n")
        print(f"Error:\n{e}")
        fail_image_list.append(inim)

# Log failed images
if len(fail_image_list) > 0:
    outdir = os.path.dirname(fail_image_list[0]) or '.'
    with open(f"{outdir}/phot.fail.list", 'w') as f:
        for finim in fail_image_list:
            f.write(f"{os.path.basename(finim)}\n")

# Print time summary
elt = time.time() - starttime
unit = 'seconds'
if elt > 60.:
    elt /= 60.
    unit = 'mins'
if elt > 60.:
    elt /= 60.
    unit = 'hours'

for fail_image in fail_image_list:
    print(fail_image)
print(f"{len(fail_image_list):,} Failed.")
print(f'PHOTOMETRY IS DONE.\t({elt:.3f} {unit})')
