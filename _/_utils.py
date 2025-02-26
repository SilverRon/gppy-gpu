
@njit
def compute_flux_density_error(magerr, flux_density):
    flux_density_error = (2.5 / np.log(10)) * (flux_density) * magerr
    return flux_density_error


@njit
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


@njit
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



def log2tmp(command, label):
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