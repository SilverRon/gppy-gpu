#------------------------------------------------------------
#	Flat
#------------------------------------------------------------
print("""#------------------------------------------------------------
#	Flat
#------------------------------------------------------------
""")

t0_flat = time.time()
#	Check Flat Frames 
try:
	filterlist = list(np.unique(ic1.filter(imagetyp='FLAT').summary['filter']))
	print(f"{len(filterlist)} filters found")
	print(f"Filters: {filterlist}")
	flatnumb = len(filterlist)
except:
	print(f"There is no flat frames")
	flatnumb = 0
#
flatdict = dict()
if flatnumb > 0:

	#	master flat dictionary
	for filte in filterlist:
		print(f'- {filte}-band')
		fimlist = []

		flat_raw_imlist = list(ic1.filter(imagetyp='FLAT', filter=filte).summary['file'])
		flat_raw_exptarr = cp.array(ic1.filter(imagetyp='FLAT', filter=filte).summary['exptime'].data.data)[:, None, None]
		_ffc = FitsContainer(flat_raw_imlist)

		_exptarr = np.array([int(expt) for expt in list(darkdict.keys())])

		closest_dark_exptime = np.min(_exptarr)
		exptime_scale_arr = flat_raw_exptarr / closest_dark_exptime

		#	Bias Correction
		_ffc.data -= mbias
		#	Dark Correction
		_ffc.data -= darkdict[str(int(closest_dark_exptime))] * exptime_scale_arr

		#	Normalization
		_ffc.data /= cp.median(_ffc.data, axis=(1, 2), keepdims=True)
	
		if verbose_gpu:
			print(f"Flat fits container GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		#	Generate Master Flat
		foutput = f'{path_data}/n{filte}.fits'
		mflat = imcombine(
			_ffc.data, name=foutput, list=flat_raw_imlist, overwrite=True,
			combine='median' # specify the co-adding method
			#width=3.0 # specify the clipping width
			#iters=5 # specify the number of iterations
		)
		#--------------------------------------------------------

		dateobs_mflat = tool.calculate_average_date_obs(ic1.filter(imagetyp='FLAT', filter=filte).summary['date-obs'])
		date = dateobs_mflat[:10].replace('-', '')
		flatim = f'{path_mframe}/{obs}/flat/{date}-n{filte}.fits'


		#	Save to the database
		if not os.path.exists(os.path.dirname(flatim)):
			os.makedirs(os.path.dirname(flatim))

		cpcom = f'cp {path_data}/n{filte}.fits {flatim}'
		print(cpcom)
		os.system(cpcom)

		#	Save to the dictionary 
		flatdict[filte] = mflat

		if verbose_gpu:
			print(f"flat combine GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

		#	Clear the momory pool
		mempool.free_all_blocks()
		del _ffc
		del mflat

		if verbose_gpu:
			print(f"Check the cleared GPU Memory Usage : {mempool.used_bytes()*1e-6:1.1f} Mbytes")

	timetbl['status'][timetbl['process']=='master_frame_flat'] = True

else:
	pass

delt_flat = time.time() - t0_flat
print(f"Flat Master Frame: {delt_flat:.3f} sec")
timetbl['time'][timetbl['process']=='master_frame_flat'] = delt_flat