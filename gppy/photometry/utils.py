from pathlib import Path
import datetime
import yaml
import numpy as np
from astropy.table import Table, hstack, vstack, unique
from astropy.coordinates import SkyCoord
from ..const import REF_DIR, FACTORY_DIR


def log2tmp(command, label):
    # path_thisfile = Path(__file__).resolve()
    # path_root = (
    #     path_thisfile.parent.parent.parent
    # )  # Careful! not a str / PATH HAS TO BE REVISED
    path_root = Path(FACTORY_DIR).resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_tmp = path_root / "_tmp"  # MODIFICATION REQUIRED
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
    term1 = ((x - center_x) ** 2) / (a**2)
    term2 = ((y - center_y) ** 2) / (b**2)
    return term1 + term2 <= 1


def weighted_median(values, errors):
    #   Calculate weights using the inverse of the errors
    weights = 1.0 / np.array(errors)
    #   Calculate the median
    median = np.median(values)
    #   Calculate the deviations from the median
    deviations = np.abs(values - median)
    #   Calculate the weighted median
    weighted_median = np.median(deviations * weights)
    return median, weighted_median


def compute_median_mad(values):
    if isinstance(values, np.ma.MaskedArray):
        values = values.data
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median, mad


def compute_flux_density_error(magerr, flux_density):
    flux_density_error = (2.5 / np.log(10)) * (flux_density) * magerr
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
        raise ValueError("Function parameters must be of the same shape!")

    do_not_correct = np.isnan(bp_rp)
    bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
    greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange = np.logical_not(do_not_correct) & (bp_rp >= 4.0)

    correction = np.zeros_like(bp_rp)
    correction[bluerange] = (
        1.154360
        + 0.033772 * bp_rp[bluerange]
        + 0.032277 * np.power(bp_rp[bluerange], 2)
    )
    correction[greenrange] = (
        1.162004
        + 0.011464 * bp_rp[greenrange]
        + 0.049255 * np.power(bp_rp[greenrange], 2)
        - 0.005879 * np.power(bp_rp[greenrange], 3)
    )
    correction[redrange] = 1.057572 + 0.140537 * bp_rp[redrange]

    return phot_bp_rp_excess_factor - correction


def sqsum(a, b):
    """
    SQUARE SUM
    USEFUL TO CALC. ERROR
    """
    return np.sqrt(a**2.0 + b**2.0)


def merge_catalogs(
    target_coord, path_calibration_field, matching_radius=1.0, path_save="./ref.cat"
):
    """
    지정된 좌표 근처의 Gaia DR3 카탈로그 소스를 가져와서 합치는 함수입니다.

    Parameters:
    - target_coord: astropy.coordinates.SkyCoord, 대상 좌표
    - field_name: str, 필드 이름
    - path_calibration_field: str, 카탈로그 파일이 있는 디렉토리 경로
    - matching_radius: float, 매칭 반경(단위: degree)
    - filters: list, 사용할 필터 목록. None일 경우 모든 필터 사용
    - path_save: str, 결과를 저장할 경로. None일 경우 현재 경로에 저장

    Returns:
    - None
    """
    # Grid table 읽기
    grid_table = Table.read(f"{path_calibration_field}/grid.csv")
    c_grid = SkyCoord(grid_table["center_ra"], grid_table["center_dec"], unit="deg")

    # 매칭 반경 내에서 grid 찾기
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


def sexcom(inim, param_insex, dualmode=False):
    """ """

    with open(f"{REF_DIR}/srcExt/default_sex.yml", "r") as f:
        param_sex = yaml.safe_load(f)

    for key in param_insex.keys():
        param_sex[key] = param_insex[key]

    sexcom_normal = "source-extractor -c {} {} ".format(param_sex["CONF_NAME"], inim)
    sexcom_dual = "source-extractor -c {} {} ".format(param_sex["CONF_NAME"], inim)
    for key in param_sex.keys():
        if key != "CONF_NAME":
            sexcom_normal += "-{} {} ".format(key, param_sex[key])

    return sexcom_normal


# -------------------------------------------------------------------------#
def limitmag(N, zp, aper, skysigma):  # 3? 5?, zp, diameter [pixel], skysigma
    import numpy as np

    R = float(aper) / 2.0  # to radius
    braket = N * skysigma * np.sqrt(np.pi * (R**2))
    upperlimit = float(zp) - 2.5 * np.log10(braket)
    return round(upperlimit, 3)
