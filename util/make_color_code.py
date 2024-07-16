import numpy as np
import seaborn as sns
import json 

def makeSpecColors(n, palette='Spectral'):
	#	Color palette
	palette = sns.color_palette(palette, as_cmap=True,)
	palette.reversed

	clist_ = [palette(i) for i in range(palette.N)]
	cstep = int(len(clist_)/n)
	clist = [clist_[i*cstep] for i in range(n)]
	return clist

wavelengths= np.arange(4000, 8875+125, 125)

mfilters = [f"m{str(center_lam)[0:3]}" for center_lam in wavelengths]
mcolors = np.array(makeSpecColors(len(mfilters)))[::-1]
mlamarr = np.array([float(filte[1:]) for filte in mfilters])
bfilters = ['u', 'g', 'r', 'i', 'z']
filter_color_palette_dict = {
	'u': 'blue',
	'g': 'green',
	'r': 'tomato',
	'i': 'crimson',
	'z': 'purple'
}

filters = mfilters+bfilters

for filte, c in zip(mfilters, mcolors):
	filter_color_palette_dict[filte] = c

# NumPy 배열을 리스트로 변환하는 과정
for key, value in filter_color_palette_dict.items():
    if isinstance(value, np.ndarray):
        filter_color_palette_dict[key] = value.tolist()

# JSON 파일로 저장
with open('filter_color_palette.json', 'w', encoding='utf-8') as f:
    json.dump(filter_color_palette_dict, f, ensure_ascii=False, indent=4)
