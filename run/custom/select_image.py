# import modules
import os
import glob
import argparse
import numpy as np
from astropy.table import Table
from ccdproc import ImageFileCollection
from tqdm import tqdm

#import matplotlib & visualization setting
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as img

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams["savefig.dpi"] = 500
mpl.rcParams["xtick.labelsize"] = 14  # X축 틱의 글자 크기 설정
mpl.rcParams["ytick.labelsize"] = 14  # Y축 틱의 글자 크기 설정
plt.rc("font", family="serif")

class color:
	PURPLE = '\033[95m'
	CYAN = '\033[96m'
	DARKCYAN = '\033[36m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	YELLOW = '\033[93m'
	RED = '\033[91m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	END = '\033[0m'

# functions
def makeSpecColors(n, palette='Spectral'):
	#	Color palette
	import seaborn as sns
	palette = sns.color_palette(palette, as_cmap=True,)
	palette.reversed

	clist_ = [palette(i) for i in range(palette.N)]
	cstep = int(len(clist_)/n)
	clist = [clist_[i*cstep] for i in range(n)]
	return clist

def find_rec(N):
    # Start from the square root of N and work downwards
    num_found = False
    while ~num_found:
        for k in range(int(N ** 0.5), 0, -1):
            if N % k == 0:  # k must divide N
                l = N // k  # Calculate l
                # Check the condition that neither exceeds twice the other
                if k <= 2 * l and l <= 2 * k:
                    num_found = True
                    return k, l
        N = N + 1
    return None, None  # Return None if no valid pair is found

# general filter set
wavelengths= np.arange(4000, 8875+125, 125)

wfilters = ['m375w', 'm425w']
mfilters = [f"m{str(center_lam)[0:3]}" for center_lam in wavelengths]
mcolors = np.array(makeSpecColors(len(mfilters)))[::-1]
mlamarr = np.array([float(filte[1:]) for filte in mfilters])
bfilters = ['u', 'g', 'r', 'i', 'z']
filter_color_palette_dict = {
	'u': 'blue',
	'g': 'green',
	'r': 'tomato',
	'i': 'crimson',
	'z': 'purple',
	'm375w': 'cyan',
	'm425w': 'dodgerblue',
}

filters = mfilters+bfilters+wfilters

for filte, c in zip(mfilters, mcolors):  # filrer colours
	filter_color_palette_dict[filte] = c

# path configuration
path_data = '/lyman/data1/processed'
path_save = '/lyman/data1/factory_whlee/selection'


def get_seeing_depth_stat(table, verbose=False):

	# NaN 값을 무시하고 계산하도록 수정
	seeing_median = np.nanmedian(table['seeing'])
	seeing_std = np.nanstd(table['seeing'])
	seeing_min = np.nanmin(table['seeing'])
	seeing_max = np.nanmax(table['seeing'])

	depth_median = np.nanmedian(table['ul5_1'])
	depth_std = np.nanstd(table['ul5_1'])
	depth_min = np.nanmin(table['ul5_1'])
	depth_max = np.nanmax(table['ul5_1'])

	if verbose:
		print(f"Seeing Median Value: {seeing_median:.3f} +/- {seeing_std:.3f}")
		print(f"Depth  Median Value: {depth_median:.3f} +/- {depth_std:.3f}")
	
	return (seeing_median, seeing_std, seeing_min, seeing_max, depth_median, depth_std, depth_min, depth_max)


def plot_seeing_depth_histogram(table, seeing_median, depth_median, n_sigma, 
                                n_median_select, n_nsigma_select, obj, filte, 
                                path_output, color='k', for_total=False):
    mosaic = """
    SSSx
    MMMD
    MMMD
    MMMD
    """
    
    fig, ax = plt.subplot_mosaic(
        mosaic, figsize=(9, 9),
        empty_sentinel="x",
        gridspec_kw=dict(
            wspace=0.0, hspace=0.0,
        )
    )
    
    # Main Axe
    ax["M"].plot(table['seeing'], table['ul5_1'], 's', c=color, mec='k', alpha=0.75, label=f'All: {len(table)}')
    
    xl, xr = ax["M"].set_xlim()
    yl, yu = ax["M"].set_ylim()
    
    ax["M"].plot([seeing_median, seeing_median, xl], [yu, depth_median, depth_median], color='tomato', zorder=0, lw=2, label=f"Median: {n_median_select}")
    ax["M"].plot([seeing_median+seeing_std, seeing_median+seeing_std, xl], [yu, depth_median-depth_std, depth_median-depth_std], color='tomato', zorder=0, lw=2, ls='--', label=f"{n_sigma} sigma: {n_nsigma_select}")
    
    ax["M"].set_xlim([xr, xl])
    ax["M"].set_xlabel("""Seeing ["]""")
    ax["M"].set_ylabel(r'5$\sigma$ Depth [AB mag]')
    ax["M"].grid('both', ls='--', alpha=0.5, color='silver', zorder=0)
    ax["M"].legend(loc='upper left', fontsize=14, framealpha=1.0)
    
    # Seeing Axe
    ax["S"].hist(table['seeing'], color='k', histtype='step', lw=3, alpha=0.75)
    ax["S"].axvline(seeing_median, color='tomato', zorder=1, lw=2)
    ax["S"].axvline(seeing_median+seeing_std, color='tomato', zorder=0, lw=2, ls='--')
    ax["S"].set_xlim([xr, xl])
    ax["S"].set_yticks([])
    
    # Depth Axe
    ax["D"].hist(table['ul5_1'], color='k', histtype='step', lw=3, alpha=0.75, orientation='horizontal')
    ax["D"].axhline(depth_median, color='tomato', zorder=1, lw=2)
    ax["D"].axhline(depth_median-depth_std, color='tomato', zorder=0, lw=2, ls='--')
    ax["D"].set_ylim([yl, yu])
    ax["D"].set_xticks([])
    
    # Remove tick labels for cleaner look
    ax["S"].tick_params(labelbottom=False)
    ax["D"].tick_params(labelleft=False)
    
    # Add a title
    plt.suptitle(f"{obj} ({filte})", fontsize=20)
    fig.tight_layout()

    fig_location = f"{path_output}/select.png"
    fig.savefig(fig_location)
    plt.close()
    # Save the figure
    if for_total:
        return fig_location


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--nsigma", "-s", type=float, nargs='?', help="Standard deviation cut of images")
	parser.add_argument("--object", "-o", type=str, nargs='?', help="Target object name")
	parser.add_argument("--verbose", "-v", type=bool, nargs='?', help="Print outputs")

	verbose = False

	args = parser.parse_args()
	if args.object:
		obj = args.object
	else: 
		print(color.RED+' Need object name, Terminate the program'+color.END)
		exit()
	if args.nsigma:
		n_sigma = args.nsigma
		print(' Sigma-Clipping value:'+color.YELLOW+f'{n_sigma:2f}'+color.END)
	else:
		print(' Sigma-Clipping value is set to default:'+color.YELLOW+' n_sigma = 1'+color.END)
		n_sigma = 1  # default
	if args.verbose:
		verbose = args.verbose

	columns_to_pick = [
		# "file", "naxis", "naxis1", "naxis2", "mjd-obs", "ctype1", "cunit1", "crval1", 
		# "file", "naxis", "naxis1", "naxis2", "ctype1", "cunit1", "crval1", 
		"file",
		# "crpix1", "cd1_1", "cd1_2", "ctype2", "cunit2", "crval2", "crpix2", "cd2_1", 
		# "crpix1", "cd1_1", "cd1_2", 
		# "cd1_1", "cd1_2", 
		"cd1_2", 
		# "cd2_2", "exptime", "gain", "saturate", "date", "object", "egain", "filter", 
		"cd2_2", "exptime", "gain", "date", "object", "egain", "filter", 
		"date-obs", "date-loc", "exposure", "centalt", "centaz", "airmass", "mjd", 
		"jd", "seeing", "peeing", "ellip", "elong", "skysig", "skyval", "refcat", 
		"maglow", "magup", "stdnumb", "auto", "aper", "aper_1", "aper_2", "aper_3", 
		"aper_4", "aper_5", "zp_auto", "ezp_auto", "ul3_auto", "ul5_auto", "zp_0", 
		"ezp_0", "ul3_0", "ul5_0", "zp_1", "ezp_1", "ul3_1", "ul5_1", "zp_2", "ezp_2", 
		"ul3_2", "ul5_2", "zp_3", "ezp_3", "ul3_3", "ul5_3", "zp_4", "ezp_4", "ul3_4", 
		"ul5_4", "zp_5", "ezp_5", "ul3_5", "ul5_5"
	]
	if verbose: print(f"{len(columns_to_pick)} columns selected")

	keys_to_check = ['seeing', 'ul5_1', 'ellip', 'skyval', 'skysig', 'airmass']

	units = [os.path.basename(folder) for folder in sorted(glob.glob(f"{path_data}/{obj}/7DT*"))]
	print(f"{len(units)} Units: {units}")

	# filters = list(np.unique([os.path.basename(folder) for folder in sorted(glob.glob(f"{path_data}/{obj}/*/*"))]))
	filters = list(np.unique([os.path.basename(folder) for folder in sorted(glob.glob(f"{path_data}/{obj}/7DT??/*"))]))
	print(f"{len(filters)} Filters: {filters}")
    
    
    # Define Output Table
	result_table = Table()
	result_table['filter'] = filters
	#	Number of Images
	result_table['n_all'] = 0
	result_table['n_bad'] = 0
	result_table['n_image'] = 0
	result_table['n_select_median'] = 0
	result_table['n_select_nsigma'] = 0
	#	Stats
	result_table['seeing_median'] = 0.
	result_table['seeing_std'] = 0.
	result_table['seeing_min'] = 0.
	result_table['seeing_max'] = 0.
	result_table['depth_median'] = 0.
	result_table['depth_std'] = 0.
	result_table['depth_min'] = 0.
	result_table['depth_max'] = 0.
	# Verbose: Do in-line outputs

	fig_list = []
	for ff, filte in tqdm(enumerate(result_table['filter'])):
		color = filter_color_palette_dict[filte]

		#	Path
		path_output = f"{path_save}/{obj}/{filte}"
		if not os.path.exists(path_output):
			os.makedirs(path_output)

		#	Images
		images = sorted(glob.glob(f"{path_data}/{obj}/*/{filte}/calib*0.fits"))
		if verbose: print(f"{len(images)} images found")

		#	Image Collection & Selection
		try:
			ic = ImageFileCollection(filenames=images)
			raw_table = ic.summary[columns_to_pick]

			bad_image_mask = ~raw_table['cd1_2'].mask
			table = raw_table[~raw_table['cd1_2'].mask]

			if verbose:
				print(f"All  images: {len(raw_table)}")
				print(f"Bad  images: {len(raw_table[~bad_image_mask])}")
				print(f"Good images: {len(table)}")

			for key in keys_to_check:
				table[key] = table[key].filled(np.nan).astype(float)

			#	Stat
			(seeing_median, seeing_std, seeing_min, seeing_max, depth_median, depth_std, depth_min, depth_max) = get_seeing_depth_stat(table=table, verbose=verbose)

			#	Index
			indx_median = (table['seeing'] < seeing_median) & (table['ul5_1'] > depth_median)
			indx_nsigma = (table['seeing'] < seeing_median + seeing_std) & (table['ul5_1'] > depth_median - depth_std)

			n_median_select = len(table[indx_median])
			n_nsigma_select = len(table[indx_nsigma])

			if verbose:
				print(f"Median Select : {n_median_select}")
				print(f"{n_sigma} Sigma Select: {n_nsigma_select}")

			#	Plot
			fig_list.append(plot_seeing_depth_histogram(table, seeing_median, depth_median, n_sigma, n_median_select, n_nsigma_select, obj, filte, path_output, color=color, for_total=True))


			#	Result Tables
			#	median
			f = open(f"{path_output}/select_median.txt", "w")
			f.write("file\n")
			for inim in table['file'][indx_median].data:
				f.write(f"{inim}\n")
			f.close()
			#	N sigma
			g = open(f"{path_output}/select_{n_sigma}sigma.txt", "w")
			g.write("file\n")
			for inim in table['file'][indx_nsigma].data:
				g.write(f"{inim}\n")
			g.close()

			#	Result Table
			
			#	Number of Images
			result_table['n_all'][ff] = len(raw_table)
			result_table['n_bad'][ff] = len(raw_table[~bad_image_mask])
			result_table['n_image'][ff] = len(table)
			result_table['n_select_median'][ff] = len(table[indx_median])
			result_table['n_select_nsigma'][ff] = len(table[indx_nsigma])
			#	Stats
			result_table['seeing_median'][ff] = seeing_median
			result_table['seeing_std'][ff] = seeing_std
			result_table['seeing_min'][ff] = seeing_min
			result_table['seeing_max'][ff] = seeing_max
			result_table['depth_median'][ff] = depth_median
			result_table['depth_std'][ff] = depth_std
			result_table['depth_min'][ff] = depth_min
			result_table['depth_max'][ff] = depth_max
		except KeyError:
			print(f"[{filte}]: Key Error!")

	k, l = find_rec(len(filters))
	fig, axes = plt.subplots(ncols=k, nrows=l, figsize=(k*2, l*2), dpi=300)
	for i, ax in enumerate(axes.flatten()):
		try:
			impng = img.imread(fig_list[i])
			ax.imshow(impng)
			ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
		except IndexError:
			ax.remove()

	fig.tight_layout()
	fig.savefig(f"{path_save}/{obj}/selection_all.png")

	for key in result_table.keys()[6:]: result_table[key].format = '1.3f'
	result_table.write(f"{path_save}/{obj}/summary.csv", overwrite=True)
	result_table