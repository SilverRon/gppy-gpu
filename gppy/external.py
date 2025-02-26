import os
import subprocess
from astropy.io import fits
from .const import FACTORY_DIR, REF_DIR


def solve_field(
    inim, outim=None, dump_dir=None, get_command=False, pixscale=0.505, radius=1.0
):
    """
    Runs Astrometry.net's `solve-field` to compute the World Coordinate System (WCS) for an input FITS image.

    This function creates a temporary working directory, generates a symbolic link to the input FITS file,
    and runs `solve-field` to solve the astrometry of the image. It supports real-time output streaming
    and optional command retrieval.

    Args:
        inim (str):
            Path to the input FITS image.
        outim (str):
            Path to the output FITS image.
        dump_dir (str, optional):
            Directory where intermediate results will be stored. If None, a temporary directory is created
            inside the input image's directory.
        get_command (bool, optional):
            If True, returns the command as a string instead of executing it. Defaults to False.
        pixscale (float, optional):
            Approximate pixel scale of the image in arcseconds per pixel. Defaults to 0.505 arcsec/pixel.
        radius (float, optional):
            Search radius for the solution in degrees. Defaults to 1.0 degree.

    Returns:
        str:
            If `get_command` is False, returns the path to the solved FITS file with the WCS solution.
            If `get_command` is True, returns the command string that would be executed.

    Raises:
        OSError: If the FITS file cannot be read.
        KeyError: If RA and DEC cannot be retrieved from the FITS header.

    Example:
        Solve an image normally:
        ```python
        solved_file = solve_field("image.fits")
        print(f"Solved FITS file: {solved_file}")
        ```

        Get the command without executing it:
        ```python
        command = solve_field("image.fits", get_command=True)
        print(f"Command: {command}")
        ```
    """

    inim = os.path.abspath(inim)
    img_dir = os.path.dirname(inim)
    working_dir = dump_dir or os.path.join(img_dir, "tmp_solvefield")
    working_dir = os.path.abspath(working_dir)
    os.makedirs(working_dir, exist_ok=True)

    # soft link inside working_dir
    fname = os.path.basename(inim)
    soft_link = os.path.join(working_dir, fname)
    if not os.path.exists(soft_link):
        os.symlink(inim, soft_link)

    # outname = os.path.join(working_dir, f"{Path(inim).stem}_solved.fits")
    outname = outim or os.path.join(os.path.splitext(soft_link)[0] + "_solved.fits")

    # Solve-field using the soft link
    # e.g., solve-field calib_7DT11_T00139_20250102_014643_m425_100s.fits --crpix-center --scale-unit arcsecperpix --scale-low '0.4949' --scale-high '0.5151' --no-plots --new-fits solved.fits --overwrite --use-source-extractor --cpulimit 4
    solvecom = [
        "solve-field", f"{soft_link}",  # this file is not changed by solve-field
        "--new-fits", outname,  # you can give 'none'
        # "--config", f"{path_cfg}",
        # "--source-extractor-config", f"{path_sex_cfg}",
        # "--no-fits2fits",  # Do not create output FITS file
        "--overwrite",
        "--crpix-center",
        "--scale-unit", "arcsecperpix",
        "--scale-low", f"{pixscale*0.98}",
        "--scale-high", f"{pixscale*1.02}",
        "--use-source-extractor",  # Crucial speed boost. 30 s -> 5 s
        "--cpulimit", f"{4}",  # 8 cores were 0.1 sec slower
        "--no-plots",  # MASSIVE speed boost. 2 min -> 5 sec
        # "--no-tweak",  # Skip SIP distortion correction. 0.3 seconds boost.
        # "--downsample", "4",  # not much difference
    ]  # fmt: skip

    try:
        header = fits.getheader(inim)
        ra = header["ra"]
        dec = header["dec"]
        solvecom = solvecom + [
            "--ra", f"{ra:.4f}",
            "--dec", f"{dec:.4f}",
            "--radius", f"{radius:.1f}",
        ]  # fmt: skip
    except:
        print("Couldn't get RA Dec from header. Solving blindly")

    if get_command:
        return " ".join(solvecom)

    # # Use Popen for real-time output
    # process = subprocess.Popen(
    #     solvecom,
    #     cwd=working_dir,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    #     text=True,
    # )

    # # Also print messages to shell: should be captured by logger
    # for line in process.stdout:
    #     print(line, end="")

    # process.wait()  # Ensure the process completes
    log_file = os.path.join(working_dir, "solvefield.log")
    solvecom = f"{' '.join(solvecom)} > {log_file} 2>&1"
    # print(f"solve-field command {solvecom}")
    subprocess.run(solvecom, cwd=working_dir, shell=True)

    return outname


def scamp(input, ahead=None, local_astref=None, get_command=False):
    """
    Input is a fits-ldac catalog or a text file.
    Supply a text file of filenames to run multiple files jointly.
    """
    scampconfig = os.path.join(REF_DIR, "7dt.scamp")
    # "/data/pipeline_reform/dhhyun_lab/scamptest/7dt.scamp"

    path_ref_scamp = os.path.join(FACTORY_DIR, "ref_scamp")
    # "/data/pipeline_reform/dhhyun_lab/scamptest"
    log_file = os.path.splitext(input)[0] + "_scamp.log"

    # assumes joint run if input is not fits
    if os.path.splitext(input)[1] != ".fits":
        input = f"@{input}"

    # scampcom = f’scamp {catname} -c {os.path.join(path_cfg, “kmtnet.scamp”)} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {gaialdac} -POSITION_MAXERR 20.0 -CROSSID_RADIUS 5.0 -DISTORT_DEGREES 3 -PROJECTION_TYPE TPV -AHEADER_GLOBAL {ahead} -STABILITY_TYPE INSTRUMENT’
    scampcom = f"scamp -c {scampconfig} {input}"

    if local_astref:
        scampcom = f"{scampcom} -ASTREF_CATALOG FILE -ASTREFCAT_NAME {local_astref}"

    else:
        scampcom = f"{scampcom} -REFOUT_CATPATH {path_ref_scamp}"

    if ahead:
        # scampcom = f"{scampcom} -AHEADER_NAME {ahead}"
        scampcom = f"{scampcom} -AHEADER_GLOBAL {ahead}"
    scampcom = f"{scampcom} > {log_file} 2>&1"
    # print(scampcom)

    if get_command:
        return scampcom
    os.system(scampcom)
    # scampcom = f"scamp -c {scampconfig} {outcat} -REFOUT_CATPATH {path_ref_scamp} -AHEADER_NAME {ahead_file}"
    # subprocess.run(f"{scampcom} > {log_file} 2>&1", shell=True, text=True)

    # astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
    # scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
    # scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"
    # try:
    #     result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     print(result.stdout.decode())  # 명령어 실행 결과 출력
    # except subprocess.CalledProcessError as e:
    #     print(f"Command failed with error code {e.returncode}")
    #     print(f"stderr output: {e.stderr.decode()}")
    return os.path.splitext(input)[0] + ".head"


def missfits(inim):
    """
    Input images gets wcs updated, .back is made as a copy or the original
    Searches .head file in the same directory and with the same stem as inim and applies it to inim
    """

    missfitsconf = f"{REF_DIR}/7dt.missfits"
    missfitscom = f"missfits -c {missfitsconf} {inim}"
    # missfitscom = f"missfits -c {path_config}/7dt.missfits @{path_image_missfits_list}"

    os.system(missfitscom)
    # working_dir = "/data/pipeline_reform/dhhyun_lab/scamptest/solvefield"
    # subprocess.run(missfitscom, shell=True, cwd=working_dir)


def sextractor(
    inim: str,
    outcat: str = None,
    prefix="prep",
    log_file: str = None,
    sex_args: list = None,
    config=None,  # supply config.config
    logger=None,
    return_output=False,
):
    """
    e.g., override default by supplying sex_args like ["-PIXEL_SCALE", f"{pixscale}"]
    Sextractor log file is created in the same directory as outcat.
    No support for dual mode yet.
    """

    def get_sex_config(prefix, ref_path=None):
        from .const import REF_DIR

        # "/data/pipeline_reform/gppy-gpu/gppy/ref/srcExt"
        ref_path = ref_path or os.path.join(REF_DIR, "srcExt")
        postfix = ["sex", "param", "conv", "nnw"]
        return [os.path.join(ref_path, f"{prefix}.{pf}") for pf in postfix]

    def chatter(message):
        if logger:
            logger.debug(message)
        else:
            print(message)

    if config:
        chatter("Using Configuration Class")
        sex = config.sex.sex
        param = config.sex.param
        nnw = config.sex.nnw
        conv = config.sex.conv
    else:
        sex, param, conv, nnw = get_sex_config(prefix)

    outcat = outcat or os.path.splitext(inim)[0] + f".{prefix}.cat"
    log_file = log_file or os.path.splitext(outcat)[0] + f"_sextractor.log"

    sexcom = [
        "source-extractor", f"{inim}",
        "-c", f"{sex}",
        "-CATALOG_NAME", f"{outcat}",
        # "-catalog_type", "fits_ldac",  # this is for scamp presex
        "-PARAMETERS_NAME", f"{param}",
        "-FILTER_NAME", f"{conv}",
        "-STARNNW_NAME", f"{nnw}",
    ]  # fmt: skip

    # add additional arguments when given
    if sex_args:
        sexcom = sexcom + sex_args

    chatter(f"Sextractor output catalog: {outcat}")
    chatter(f"Sextractor Log: {log_file}")

    sexcom = " ".join(sexcom)
    chatter(f"Sextractor Command: {sexcom}")

    sexout = subprocess.getoutput(sexcom)
    # result = subprocess.run(sexcom, shell=True, capture_output=True, text=True, stderr=subprocess.STDOUT)
    # sexout = result.stdout + result.stderr
    with open(log_file, "w") as f:
        f.write(sexcom)
        f.write(sexout)
        # f.write(sexerr)

    # Run the command and capture output
    # with open(log_file, "w") as f:
    #     process = subprocess.Popen(
    #         sexcom, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    #     )
    #     f.write(f"{sexcom}")

    #     output_lines = []
    #     for line in process.stdout:
    #         # print(line, end="")  # Print to console (optional)
    #         output_lines.append(line)
    #     process.wait()  # Ensure process completes
    #     sexout = "".join(output_lines)
    #     f.write(sexout)

    # This redirects all output to log_file
    # sexcom = f"{' '.join(sexcom)} > {log_file} 2>&1"
    # subprocess.run(sexcom, shell=True, text=True)

    # alternative; not working
    # with open(log_file, "w") as f:
    #     DEVNUL tricks SEx to think its running non-interactively
    #     subprocess.run(sexcom, stdout=f, stderr=f, text=True, stdin=subprocess.DEVNULL)

    chatter(f"Sextractor completed")

    if return_output:
        return outcat, sexout
    else:
        return outcat
