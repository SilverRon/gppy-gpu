import glob
from ccdproc import ImageFileCollection

path_proc = '/lyman/data1/processed_1x1_gain2750'

# obj = 'UDS'
obj = input(f"Type Field Name (UDS):")
if obj == 'RIS':
	objkey = '*'
else:
	objkey = obj


telesopes = sorted(glob.glob(f'{path_proc}/{objkey}/7DT??'))
for tel in telesopes: print(tel)

for tel in telesopes:
	_filters = sorted(glob.glob(f"{tel}/*"))
	# print(f"{tel}: {_filters}")
	# for filte in ['/large_data/processed/UDS/7DT06/m600']:
	for filte in _filters:
		filenames = sorted(glob.glob(f'{filte}/calib*0.fits'))
		if len(filenames) > 0:
			print(f"{filte}/calib*0.fits: {len(filenames)}")
			ic = ImageFileCollection(filenames=filenames)
			table = ic.summary
			table.write(f'{filte}/summary.csv', format='csv', overwrite=True)
