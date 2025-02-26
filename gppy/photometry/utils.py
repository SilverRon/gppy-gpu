import numpy as np
from numba import njit
from astropy.table import Table, hstack, vstack, unique
from astropy.coordinates import SkyCoord
from typing import Any, Tuple, Optional, Dict, Union


@njit
def rss(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate Root Sum Square of two arrays.

    Args:
        a: First input array
        b: Second input array

    Returns:
        Root sum square of inputs: sqrt(a^2 + b^2)
    """
    return np.sqrt(np.nan_to_num(a) ** 2.0 + np.nan_to_num(b) ** 2.0)


@njit
def is_within_ellipse(
    x: np.ndarray, y: np.ndarray, center_x: float, center_y: float, a: float, b: float
) -> np.ndarray:
    """
    Check if points lie within an ellipse.

    Args:
        x, y: Arrays of point coordinates
        center_x, center_y: Ellipse center coordinates
        a, b: Semi-major and semi-minor axes

    Returns:
        Boolean array indicating points inside ellipse
    """
    term1 = ((x - center_x) ** 2) / (a**2)
    term2 = ((y - center_y) ** 2) / (b**2)
    return term1 + term2 <= 1


@njit
def compute_median_mad(values: np.ndarray) -> tuple:
    """
    Compute median and Median Absolute Deviation (MAD).

    Args:
        values: Input array

    Returns:
        Tuple of (median, MAD)
    """
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, mad


@njit
def limitmag(N: np.ndarray, zp: float, aper: float, skysigma: float) -> np.ndarray:
    """
    Calculate limiting magnitude.

    Args:
        N: Signal-to-noise ratio array
        zp: Zero point
        aper: Aperture diameter
        skysigma: Sky background sigma

    Returns:
        Array of limiting magnitudes
    """
    R = aper / 2.0  # Convert to radius
    braket = N * skysigma * np.sqrt(np.pi * R**2)
    upperlimit = zp - 2.5 * np.log10(braket)
    return np.round(upperlimit, 3)


@njit
def zp_correction(
    mag: np.ndarray, mag_err: np.ndarray, zp: float, zperr: float
) -> tuple:
    """
    Apply zero point correction to magnitudes.

    Args:
        mag: Magnitude array
        mag_err: Magnitude error array
        zp: Zero point value
        zperr: Zero point error

    Returns:
        Tuple of (corrected_mag, corrected_err, flux, flux_err, SNR)
    """
    mag = mag + zp
    mag_err = mag_err + zperr
    flux = 10 ** ((23.9 - mag) / 2.5)
    flux_err = 0.4 * np.log(10) * flux * mag_err
    snr = flux / flux_err
    return mag, mag_err, flux, flux_err, snr


def parse_gaia_catalogs(target_coord, path_calibration_field, matching_radius=1.0):
    """
    Merge Gaia DR3 catalog sources near the specified coordinates.

    Parameters:
        target_coord (astropy.coordinates.SkyCoord): Target coordinates
        path_calibration_field (str): Directory path containing catalog files
        matching_radius (float): Matching radius in degrees
        path_save (str): Path to save the results. If None, saves in current directory

    Returns:
        astropy.table.Table: Combined reference catalog table
    """

    grid_table = Table.read(f"{path_calibration_field}/grid.csv")
    c_grid = SkyCoord(grid_table["center_ra"], grid_table["center_dec"], unit="deg")

    sep_arr = target_coord.separation(c_grid).deg
    indx_match = np.where(sep_arr < matching_radius)
    matched_grid_table = grid_table[indx_match]

    all_filters = [
        "u", "g", "r", "i", "z",
        "m375w", "m400", "m412", "m425", "m425w", "m437", "m450",
        "m462", "m475", "m487", "m500", "m512", "m525", "m537",
        "m550", "m562", "m575", "m587", "m600", "m612", "m625",
        "m637", "m650", "m662", "m675", "m687", "m700", "m712",
        "m725", "m737", "m750", "m762", "m775", "m787", "m800",
        "m812", "m825", "m837", "m850", "m862", "m875", "m887",
    ]  # fmt:skip

    gaia_general_keys = [
        "source_id",
        "ra",
        "dec",
        "parallax",
        # 'parallax_over_error', # TBD
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        # 'phot_bp_mean_mag', # TBD
        # 'phot_rp_mean_mag', # TBD
        "bp_rp",
    ]

    all_tablelist = []
    for prefix in matched_grid_table["prefix"]:
        tablelist = []
        for ff, filte in enumerate(all_filters):
            _tablename = f"{path_calibration_field}/{prefix}/{filte}.fits"
            _reftbl = Table.read(_tablename)
            if ff == 0:
                _table = Table()
                for gaia_key in gaia_general_keys:
                    _table[gaia_key] = _reftbl[gaia_key]
            # 	Mag & SNR Keys
            filter_magkey = f"mag_{filte}"
            filter_snrkey = f"snr_{filte}"
            _table[filter_magkey] = _reftbl[filter_magkey]
        tablelist.append(_table)
        all_tablelist.append(hstack(tablelist))

    all_reftbl = vstack(all_tablelist)
    all_reftbl = unique(all_reftbl, keys="source_id")

    # if not os.path.exists(path_save):
    # 	all_reftbl.write(path_save, overwrite=True)

    return all_reftbl


def filter_table(table: Table, key: str, value: Any, method: str = "equal") -> Table:
    """
    Filter table based on column values.

    Args:
        table: Input table
        key: Column name to filter on
        value: Value to compare against
        method: Comparison method ('equal', 'lower', or 'upper')

    Returns:
        Filtered table
    """
    if method == "equal":
        return table[table[key] == value]
    elif method == "lower":
        return table[table[key] > value]
    elif method == "upper":
        return table[table[key] < value]
    else:
        raise ValueError("method must be 'equal', 'lower', or 'upper'")


def keyset(mag_key: str, filter: str) -> list:
    """
    Generate photometry key names for a given filter.

    Args:
        mag_key: Base magnitude key
        filter: Filter name

    Returns:
        List of keys for magnitude, error, flux, flux error, and SNR
    """
    _magkey = f"{mag_key}_{filter}"
    _magerrkey = _magkey.replace("MAG", "MAGERR")
    _fluxkey = _magkey.replace("MAG", "FLUX")
    _fluxerrkey = _magerrkey.replace("MAG", "FLUX")
    _snrkey = _magkey.replace("MAG", "SNR")
    return [_magkey, _magerrkey, _fluxkey, _fluxerrkey, _snrkey]


def get_aperture_dict(peeing: float, pixscale: float) -> dict:
    """
    Generate dictionary of aperture configurations.

    Args:
        peeing: Seeing in pixels
        pixscale: Pixel scale in arcsec/pixel

    Returns:
        Dictionary of aperture configurations
    """
    aperture_dict = {
        "MAG_AUTO": (0.0, "MAG_AUTO DIAMETER [pix]"),
        "MAG_APER": (2 * 0.6731 * peeing, "BEST GAUSSIAN APERTURE DIAMETER [pix]"),
        "MAG_APER_1": (2 * peeing, "2*SEEING APERTURE DIAMETER [pix]"),
        "MAG_APER_2": (3 * peeing, "3*SEEING APERTURE DIAMETER [pix]"),
        "MAG_APER_3": (3 / pixscale, """FIXED 3" APERTURE DIAMETER [pix]"""),
        "MAG_APER_4": (5 / pixscale, """FIXED 5" APERTURE DIAMETER [pix]"""),
        "MAG_APER_5": (10 / pixscale, """FIXED 10" APERTURE DIAMETER [pix]"""),
    }
    return aperture_dict


def get_sex_args(
    image: str, phot_conf: Any, gain: float, peeing: float, pixscale: float
) -> list:
    """
    Generate SExtractor configuration arguments.

    Args:
        image: Path to image file
        phot_conf: Photometry configuration object
        gain: CCD gain value
        peeing: Seeing in pixels
        pixscale: Pixel scale in arcsec/pixel

    Returns:
        List of SExtractor command line arguments
    """
    aperture_dict = get_aperture_dict(peeing, pixscale)

    magkeys = list(aperture_dict.keys())
    aperlist = [aperture_dict[key][0] for key in magkeys[1:]]

    PHOT_APERTURES = ",".join(map(str, aperlist))

    sex_config = {}
    sex_config["PHOT_APERTURES"] = PHOT_APERTURES
    sex_config["SATUR_LEVEL"] = "65000.0"
    sex_config["GAIN"] = str(gain)
    sex_config["PIXEL_SCALE"] = str(pixscale)
    sex_config["SEEING_FWHM"] = "2.0"

    for key in phot_conf.sex_vars.keys():
        sex_config[key] = phot_conf.sex_vars[key]

    # 	Add Weight Map from SWarp
    weightim = image.replace("com", "weight")
    if "com" in image and os.path.exists(weightim):
        sex_config["WEIGHT_TYPE"] = "MAP_WEIGHT"
        sex_config["WEIGHT_IMAGE"] = weightim
    # 	Check Image
    if phot_conf.check == True:
        sex_config["CHECKIMAGE_TYPE"] = "SEGMENTATION,APERTURES,BACKGROUND,-BACKGROUND"
        sex_config["CHECKIMAGE_NAME"] = (
            f"{self.head}.seg.fits,{self.head}.aper.fits,{self.head}.bkg.fits,{self.head}.sub.fits"
        )
    else:
        pass

    sex_args = [s for key, val in sex_config.items() for s in (f"-{key}", f"{val}")]

    return sex_args
