from .logging import logger


class Astrometry:

    def __init__(self, config):
        """Initialize the astrometry module."""
        self.config = config

    # fmt: off
    def run(self):
        self.config.config.file.processed_files

        precatlist = [f"{path_data}/{_fname.replace('fits', 'pre.cat')}" for _fname in ic_fdzobj.summary['file']]  # fmt:skip

        # pass as arguments
        conf_simple = f"{path_config}/simple.sex"
        param_simple = f"{path_config}/simple.param"
        nnw_simple = f"{path_config}/simple.nnw"
        conv_simple = f"{path_config}/simple.conv"

#

        cpcom_default_cfg = f"cp {path_config}/default.sex {path_data}"
        print(cpcom_default_cfg)
        os.system(cpcom_default_cfg)

        #	Astrometry
        while psutil.virtual_memory().percent > memory_threshold:
            print(f"Memory Usage is above {memory_threshold}% ({psutil.virtual_memory().percent}%) - Start the Astrometry!!!")
            time.sleep(10)

        print(f"Memory Usage is below {memory_threshold}% - Start the Astrometry!!!")
        with ProcessPoolExecutor(max_workers=ncore) as executor:
            # results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, cfglist, _))
            results = list(executor.map(calib.astrometry, fnamelist, objectlist, ralist, declist, fovlist, cpulimitlist, _, _))

# path_processed.
# path_factory



#	Catalog list for Scamp
    path_cat_scamp_list = f"{path_data}/cat.scamp.list"
    print(f"Generate Catalog List for SCAMP: {path_cat_scamp_list}")
    s = open(path_cat_scamp_list, 'w')
    for incat in outcatlist:
        s.write(f"{incat}\n")
    s.close()

    #	Head list for MissFits
    path_head_missfits_list = f"{path_data}/head.missfits.list"
    print(f"Generate Head List for MissFits: {path_head_missfits_list}")
    m = open(path_head_missfits_list, 'w')
    for inhead in outheadlist:
        m.write(f"{inhead}\n")
    m.close()

    #	Image list for MissFits
    path_image_missfits_list = f"{path_data}/image.missfits.list"
    print(f"Generate Image List for MissFits: {path_image_missfits_list}")
    i = open(path_image_missfits_list, 'w')
    # for inim in calimlist:
    for inim in fdzimlist:
        i.write(f"{inim}\n")
    i.close()


        #	SCAMP (input CATALOG)
    print(f"= = = = = = = = = = = = Astrometric Correction = = = = = = = = = = = =")
    for oo, obj in enumerate(objarr):
        print(f"[{oo+1}/{len(objarr)}] {obj}")

        path_cat_scamp_list = f"{path_data}/{obj}.cat.scamp.list"
        s = open(path_cat_scamp_list, 'w')
        obj_outcatlist = [incat for incat in outcatlist if obj in os.path.basename(incat)]
        for incat in obj_outcatlist:
            s.write(f"{incat}\n")
        s.close()

        if local_astref and (re.match(tile_name_pattern, obj)) and (obj not in ['T04231', 'T04409', 'T04590']):
            astrefcat = f"{path_ref_scamp}/{obj}.fits" if 'path_astrefcat' not in upaths or upaths['path_astrefcat'] == '' else upaths['path_astrefcat']
            if debug:
                print('='*79)
                print('astrefcat', astrefcat)
                print('='*79)
            scamp_addcom = f"-ASTREF_CATALOG FILE -ASTREFCAT_NAME {astrefcat}"
        else:
            scamp_addcom = f"-REFOUT_CATPATH {path_ref_scamp}"

        #	Run
        # scampcom = f"scamp -c {path_config}/7dt.scamp @{path_cat_scamp_list} {scamp_addcom}"
        # print(scampcom)
        # os.system(scampcom)

        # Run with subprocess
        path_scampconfig = f"{path_config}/7dt.scamp" if 'path_scampconfig' not in upaths or upaths['path_scampconfig'] == '' else upaths['path_scampconfig']
        scampcom = ["scamp", "-c", path_scampconfig, f"@{path_cat_scamp_list}"]
        scampcom += scamp_addcom.split()
        print(" ".join(scampcom))  # Join the command list for printing

        try:
            result = subprocess.run(scampcom, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # 명령어 실행 결과 출력
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error code {e.returncode}")
            print(f"stderr output: {e.stderr.decode()}")


    missfitscom = f"missfits -c {path_config}/7dt.missfits @{path_image_missfits_list}"
    ##	Multi-Threads MissFits Compile
    # missfitscom = f"missfits @{path_image_missfits_list} @{path_head_missfits_list} -NTHREADS {ncore}"
    print(missfitscom)
    os.system(missfitscom)

    # TAN-TPV

    # polygon - field rotation

    # pixscale to header


#


# fmt: off
def astrometry(inim, pixscale, ra=None, dec=None, fov=1, cpulimit=60, path_sex_cfg=None, path_cfg=None):
	'''
	ra : hh:mm:ss
	dec : dd:mm:ss
	fov [deg]
	'''
	import os
	upscl = str(pixscale + pixscale*0.02)
	loscl = str(pixscale - pixscale*0.02)
	outname = os.path.dirname(inim)+'/a'+os.path.basename(inim)
	# com     ='/usr/local/astrometry/bin/solve-field/solve-field '+inim \

	# com     ='/usr/local/astrometry/bin/solve-field '+inim \
	com     ='solve-field '+inim \
			+' --crpix-center --scale-unit arcsecperpix --scale-low '+loscl+' --scale-high '+upscl \
			+' --no-plots --new-fits '+outname+' --overwrite --use-source-extractor --cpulimit {}'.format(cpulimit)
			# +' --no-plots --new-fits '+outname+' --overwrite --cpulimit {}'.format(cpulimit)
	#	RA, Dec
	if ((ra != None) & (dec != None) & (fov != None)):
		com	= com + f' --ra {ra:.3f} --dec {dec:.3f} --radius {fov:.3f}'
	else:
		pass
	#	SouceEXtractor Configuration
	if path_sex_cfg != None:
		com = com + f" --source-extractor-config {path_sex_cfg}"
	#	
	if path_cfg != None:
		com = com + f' --config {path_cfg}'
	print(com); os.system(com)


def fnamechange(inim, obs):

	ext = inim.split('.')[-1]

	#	7DT Format
	keywords = ['OBJECT', 'FILTER', 'DATE-OBS', 'EXPTIME',]

	#	Open Image Header
	with fits.open(inim) as hdulist:
		# Get the primary header
		header = hdulist[0].header

		# Extract the header
		ehdr = {key: header.get(key, None) for key in keywords}
	datestr, timestr = extract_date_and_time(ehdr['DATE-OBS'], round_seconds=True)

	exptime = tool.convert_number(ehdr['EXPTIME'])

	#	7DT Formated name
	newname = f"{os.path.dirname(inim)}/calib_{obs}_{ehdr['OBJECT']}_{datestr}_{timestr}_{ehdr['FILTER']}_{exptime}.{ext}"
	#	Rename
	os.rename(inim, newname)
