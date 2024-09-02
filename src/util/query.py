#============================================================
#	https://ps1images.stsci.edu/ps1image.html
#	By Sophia Kim 2019.01.22. based on code PS1 suggests on the link above
#	Pan-STARRS DR1 data query
#	from https://michaelmommert.wordpress.com/2017/02/13/accessing-the-gaia-and-pan-starrs-catalogs-using-python/
#	By CS Choi 
#	REVISED AND ORGANIZED BY GREGORY S.H. PAEK
#	UPDATE : 20.01.03
#============================================================
# from __future__ import print_function

import os
import sys
import glob
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from astropy.table import Table
from astropy.table import hstack
from astropy.table import vstack
from astropy.table import unique

path_thisfile = Path(__file__).resolve()
path_root = path_thisfile.parent.parent.parent
sys.path.append(str(path_root / 'src'))
# sys.path.append('..')
from util import tool
from phot import gpphot
# from imsng import tool
# from imsng import phot_tbd
#============================================================
def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False):
	"""
	Get URL for images in the table
	
	ra, dec = position in degrees
	size = extracted image size in pixels (0.25 arcsec/pixel)
	output_size = output (display) image size in pixels (default = size).
				  output_size has no effect for fits format images.
	filters = string with filters to include
	format = data format (options are "jpg", "png" or "fits")
	color = if True, creates a color image (only for jpg or png format).
			Default is return a list of URLs for single-filter grayscale images.
	Returns a string with the URL
	"""	
	#------------------------------------------------------------
	if color and format == "fits":
		raise ValueError("color images are available only for jpg or png formats")
	if format not in ("jpg","png","fits"):
		raise ValueError("format must be one of jpg, png, fits")

	table	= getimages(ra, dec, size=size, filters=filters)
	url		= (	"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
				"ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
	if output_size:
		url = url + "&output_size={}".format(output_size)
	# sort filters from red to blue
	flist = ["yzirg".find(x) for x in table['filter']]
	table = table[np.argsort(flist)]
	if color:
		if len(table) > 3:
			# pick 3 filters
			table = table[[0,len(table)//2,len(table)-1]]
		for i, param in enumerate(["red","green","blue"]):
			url = url + "&{}={}".format(param,table['filename'][i])
	else:
		urlbase = url + "&red="
		url = []
		for filename in table['filename']:
			url.append(urlbase+filename)
	return url
#------------------------------------------------------------
def getimages(ra,dec,size=240,filters="grizy"):
	"""
	Query ps1filenames.py service to get a list of images
	ra, dec = position in degrees
	size = image size in pixels (0.25 arcsec/pixel)
	filters = string with filters to include
	Returns a table with the results
	"""	
	service	= "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
	url		= ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
		   "&filters={filters}").format(**locals())
	table	= Table.read(url, format='ascii.ecsv')
	return table
#------------------------------------------------------------
def downimage_routine(outim, name, ra, dec, size, output_size, filters, format, save_dir='.'):
	filt = filters
	param_geturl= dict(	ra			= ra,
						dec			= dec,
						size		= size,
						output_size	= None,
						filters		= filt,
						format		= "fits")
	url			= geturl(**param_geturl)
	fh		= fits.open(url[0])
	# newname	= 'Ref'+'-PS1-'+name+'-'+filt+'.fits'
	newname = outim
	fh.writeto(save_dir+'/'+newname, overwrite=True)
	pan, panhd	= fits.getdata(save_dir+'/'+newname, header=True)
	pan0		= np.nan_to_num(pan)
	fits.writeto(save_dir+'/'+newname, pan0, panhd, overwrite=True)
#------------------------------------------------------------
#	SAMPLE LINES FOR DOWNLOADING SINGLE IMAGE FROM PS1
'''
param_down	= dict(  name = 'AT2019dko',
					ra = 181.4641667,
					dec = 67.2569444,
					size = 5000,
					output_size = None,
					filters = 'r',
					format = 'fits',
					save_dir='.')
try:
	query.downimage_routine(**param_down)
except:
	try:
		query.downimage_routine(**param_down)
	except:
		try:
			query.downimage_routine(**param_down)		
		except:
			query.downimage_routine(**param_down)

#------------------------------------------------------------
#	SAMPLE LINES FOR DOWNLOADING MULTIPLE IMAGES FROM PS1
#	intbl : 'ascii' TABLE
for i in range(len(intbl)):
	print('['+str(i+1)+'/'+str(len(intbl))+']')
	param_down= dict(	name		= intbl['name'][i],
						ra			= intbl['ra'][i],
						dec			= intbl['dec'][i],
						size		= 5000,
						output_size	= None,
						filters		= 'r',
						format		= "fits")
	try:
		downimage_routine(**param_down)
	except:
		try:
			downimage_routine(**param_down)
		except:
			downimage_routine(**param_down)
'''
#------------------------------------------------------------
def querybox(refcatname, obj, racent, decent, path_refcat, radius=0.5, refmagkey=''):
	'''
	reftbl = querybox(**param_query)
	'''
	#------------------------------------------------------------
	#	REF. CATALOG QUERY
	#------------------------------------------------------------
	refcatlist	= glob.glob(path_refcat+'/*.cat')
	#------------------------------------------------------------
	if refcatname	== 'PS1':
		if path_refcat+'/ps1-'+obj+'.cat' not in refcatlist:
			querytbl = ps1_query(obj, racent, decent, path_refcat, radius=radius)
		else:
			querytbl = ascii.read(path_refcat+'/ps1-'+obj+'.cat')
		reftbl, refcat = ps1_Tonry(querytbl, obj)
	#------------------------------------------------------------
	elif refcatname	== 'SDSS':
		if path_refcat+'/sdss-'+obj+'.cat' not in refcatlist:
			querytbl = sdss_query(obj, racent, decent, path_refcat)
		else:
			querytbl = ascii.read(path_refcat+'/sdss-'+obj+'.cat')
		reftbl, refcat = sdss_Blaton(querytbl, obj)
	#------------------------------------------------------------
	elif refcatname	== 'GALEX':
		if path_refcat+'/galex-'+obj+'.cat' not in refcatlist:
			querytbl = galex_query(obj, racent, decent, path_refcat)
		else:
			querytbl = ascii.read(path_refcat+'/galex-'+obj+'.cat')
		reftbl, refcat = querytbl, f'galex-{obj}.cat'
	#------------------------------------------------------------
	elif (refcatname == 'APASS'):
		if 'm' in refmagkey:
			incat = f'{path_refcat}/apass-{obj}.cat'
			outcat = f'{path_refcat}/apass-{obj}.med.cat'
			#	No med catalog
			if outcat not in refcatlist:
				#	No APASS catalog
				if incat not in refcatlist:
					querytbl_ = apass_query(obj, racent, decent, path_refcat, radius=radius)
				else:
					querytbl_ = ascii.read(incat)
				#	APASS catalog --> med catalog
				reftbl = gpphot.apass2med(incat, outcat)
				refcat = outcat
				reftbl.rename_column('RA_ICRS', 'ra')
				reftbl.rename_column('DE_ICRS', 'dec')
			else:
				reftbl = ascii.read(outcat)
				reftbl.rename_column('RA_ICRS', 'ra')
				reftbl.rename_column('DE_ICRS', 'dec')
				for key in reftbl.keys():
					if 'mag' in key:
						nkey = key.replace('mag', '')
						if 'e_' in key:
							nkey = nkey.replace('e_', '')
							nkey = nkey+'err'
						# print(key, nkey)
						reftbl.rename_column(key, nkey)
		elif 'n' in refmagkey:
			incat = f'{path_refcat}/apass-{obj}.cat'
			outcat = f'{path_refcat}/apass-{obj}.narrow.cat'
			#	No narrow catalog
			if outcat not in refcatlist:
				#	No APASS catalog
				if incat not in refcatlist:
					querytbl_ = apass_query(obj, racent, decent, path_refcat, radius=radius)
				else:
					querytbl_ = ascii.read(incat)
				#	APASS catalog --> med catalog
				reftbl = gpphot.apass2med(incat, outcat)
				refcat = outcat
			else:
				reftbl = ascii.read(outcat)
				reftbl.rename_column('RA_ICRS', 'ra')
				reftbl.rename_column('DE_ICRS', 'dec')
				for key in reftbl.keys():
					if 'mag' in key:
						nkey = key.replace('mag', '')
						if 'e_' in key:
							nkey = nkey.replace('e_', '')
							nkey = nkey+'err'
						# print(key, nkey)
						reftbl.rename_column(key, nkey)
		else:
			if path_refcat+'/apass-'+obj+'.cat' not in refcatlist:
				querytbl = apass_query(obj, racent, decent, path_refcat, radius=radius)
			else:
				querytbl = ascii.read(path_refcat+'/apass-'+obj+'.cat')
			reftbl, refcat = apass_Blaton(querytbl, obj)
	#------------------------------------------------------------
	elif refcatname	== '2MASS':
		if path_refcat+'/2mass-'+obj+'.cat' not in refcatlist:
			querytbl        = twomass_query(obj, racent, decent, path_refcat, band=refmagkey, radius=radius)
		else:
			querytbl = ascii.read(path_refcat+'/2mass-'+obj+'.cat')
		reftbl, refcat = querytbl, '2mass-'+obj+'.cat'
	elif refcatname == 'GAIA':
		refcatfile = f"{path_refcat}/gaia-{obj}.cat"
		if refcatfile not in refcatlist:
			print(f'There is no gaia reference catalog for {obj}!')
		else:
			querytbl = Table.read(refcatfile, format='csv')
		reftbl, refcat = querytbl, os.path.basename(refcatfile)
	return reftbl

#-------------------------------------------------------------------------#
def galex_query(name, radeg, dedeg, path, radius=0.6):
	"""
	https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=II/335/galex_ais
	"""
	from astroquery.vizier import Vizier 
	from astropy.coordinates import Angle
	import astropy.units as u
	import astropy.coordinates as coord
	import numpy as np
	'''
	name = 'test'
	radeg, dedeg = 359.4956397, -6.486178333
	path = '.'
	radius=0.6
	'''
	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING GALEX Catalog ...'+'\n'
	print(comment)
	outname	= f'galex-{name}.cat'
	#	QUERY PART
	Vizier.ROW_LIMIT    = -1

	from astropy.coordinates import Angle

	v = Vizier(
		columns=["**",],
		catalog="II/335/galex_ais",
		row_limit=-1,
		)
	result = v.query_region(
			coord.SkyCoord(ra=radeg, dec=dedeg, unit=(u.deg, u.deg)),
			radius=f"{radius*60}m"
			)
	qrtbl = result[0]

	indx = np.where(
		#	largeobjsize==0
		(qrtbl['Size'].mask==True) &
		#	artifact == 0
		#	NUV & FUV
		(qrtbl['Fafl']==0) &
		(qrtbl['Nafl']==0) &
		(qrtbl['nS_G']>0.9) &
		(qrtbl['fS_G']>0.9)
	)
	querytbl = qrtbl[indx]
	#	Table
	querytbl.rename_column('RAJ2000', 'ra')
	querytbl.rename_column('DEJ2000', 'dec')
	querytbl.rename_column('NUVmag', 'NUV')
	querytbl.rename_column('e_NUVmag', 'NUVerr')
	querytbl.rename_column('FUVmag', 'FUV')
	querytbl.rename_column('e_FUVmag', 'FUVerr')

	querytbl.pprint()
	querytbl.write(f"{path}/{outname}", format='ascii.ecsv', overwrite=True)
	return querytbl
#-------------------------------------------------------------------------#
def sdss_query(name, radeg, dedeg, path, radius=1.0):
	"""
	SDSS QUERY
	INPUT   :   NAME, RA [deg], Dec [deg], radius
	OUTPUT  :   QUERY TABLE
				sdss-[NAME].cat
	"""
	from astroquery.vizier import Vizier 

	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING SDSS Catalog ...'+'\n'
	print(comment)
	outname = 'sdss-'+name+'.cat'
	Vizier.ROW_LIMIT    = -1
	query   = Vizier.query_region(coord.SkyCoord(ra=radeg, dec=dedeg, \
								unit=(u.deg, u.deg), frame='icrs'), \
								width=str(radius*60)+'m', catalog=["SDSS12"])
	querycat= query[query.keys()[0]]
	querycat.write(path+'/'+outname, format='ascii.ecsv', overwrite=True)
	return querycat
#-------------------------------------------------------------------------#
def apass_query(name, radeg, dedeg, path, radius=1.0):
	"""
	APASS QUERY
	INPUT   :   NAME, RA [deg], Dec [deg], radius
	OUTPUT  :   QUERY TABLE
				apass-[NAME].cat
	"""
	from astroquery.vizier import Vizier 
	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING APASS Catalog ...'+'\n'
	print(comment)
	outname = 'apass-'+name+'.cat'
	Vizier.ROW_LIMIT    = -1
	query   = Vizier.query_region(coord.SkyCoord(ra=radeg, dec=dedeg, \
								unit=(u.deg, u.deg), frame='icrs'), \
								width=str(radius*60)+'m', catalog=["APASS9"])
								# width=str(radius*60)+'m', catalog=["APASS10"])
	dum     = query[0]
	colnames= dum.colnames
	for col in colnames:
		indx    = np.where( dum[col].mask == False )
		dum     = dum[indx]
	#   Vega    : B, V
	#   AB      : g, r, i
	#   Vega - AB Magnitude Conversion (Blanton+07)
	#   U       : m_AB - m_Vega = 0.79
	#   B       : m_AB - m_Vega =-0.09
	#   V       : m_AB - m_Vega = 0.02
	#   R       : m_AB - m_Vega = 0.21
	#   I       : m_AB - m_Vega = 0.45
	#   J       : m_AB - m_Vega = 0.91
	#   H       : m_AB - m_Vega = 1.39
	#   K       : m_AB - m_Vega = 1.85
	querycat			= Table()
	querycat['NUMBER']  = dum['recno']
	querycat['RA_ICRS'] = dum['RAJ2000']
	querycat['DE_ICRS'] = dum['DEJ2000']
	querycat['Numb_obs']= dum['nobs']
	querycat['Numb_img']= dum['mobs']
	querycat['B-V']     = dum['B-V']    + (-0.09 - 0.02)
	querycat['e_B-V']   = dum['e_B-V']
	querycat['Bmag']    = dum['Bmag']   - 0.09  # [Vega] to [AB]
	querycat['e_Bmag']  = dum['e_Bmag']
	querycat['Vmag']    = dum['Vmag']   + 0.02  # [Vega] to [AB]
	querycat['e_Vmag']  = dum['e_Vmag']
	querycat['gmag']    = dum['g_mag']
	querycat['e_gmag']  = dum['e_g_mag']
	querycat['rmag']    = dum['r_mag']
	querycat['e_rmag']  = dum['e_r_mag']
	querycat['imag']    = dum['i_mag']
	querycat['e_imag']  = dum['e_i_mag']
	
	querycat.write(path+'/'+outname, format='ascii.ecsv', overwrite=True)
	return querycat
#-------------------------------------------------------------------------#
def ps1_query(name, radeg, dedeg, path, radius=1.0):
	"""
	#	SELECT STARS FROM STARS & GALAXIES (iPSF - iKron <= 0.05)
	https://outerspace.stsci.edu/display/PANSTARRS/How+to+separate+stars+and+galaxies#
	"""
	from astroquery.vizier import Vizier 
	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING PS1 Catalog ...'+'\n'
	print(comment)
	outname	= 'ps1-'+name+'.cat'
	#	QUERY PART
	Vizier.ROW_LIMIT    = -1
	query   = Vizier.query_region(coord.SkyCoord(ra=radeg, dec=dedeg, \
								unit=(u.deg, u.deg), frame='icrs'), \
								width=str(radius*60)+'m', catalog=["II/349/ps1"])
	dum0    = query[0]
	colnames= dum0.colnames
	#	REMOVE MASKED VALUE ROW
	for col in colnames:
		indx    = np.where( dum0[col].mask == False )
		dum1    = dum0[indx]
	f_objID_bin		= []
	for i in dum1['f_objID']:
		f_objID_bin.append(bin(i)[2:])
	f_objID_bin		= np.array( f_objID_bin )
	#	SELECT POINT SOURCE & NON-VARIABLE & GOOD QUALITY STARS
	indx_sel		= []
	for j in range(len(f_objID_bin)):
		i	= f_objID_bin[j]
		#	REJECT EXTENDED SOURCES THAT CONFIMED BY PS1 & 2MASS (23, 24)
		#	REJECT QSO, RR Lyra, VARIABLE, TRANSIENT (2, 3, 4, 5, 6, 7, 8)
		#	REJECT POOR-QUALITY STACK OBJECT (30 = 0) -> not applied yet
		try:
			if (i[-23] != '1') and (i[-24] != '1') and (i[-2] != '1') and (i[-3] != '1') and (i[-4] != '1') and (i[-5] != '1') and (i[-6] != '1') and (i[-7] != '1') and (i[-8] != '1'):# and (i[0] != '1'):
				indx_sel.append(j)
		except:
			pass
	dum2	= dum1[indx_sel]
	#	SELECT STARS FROM STARS & GALAXIES (iPSF - iKron <= 0.05)
	indx_stars		= np.where( (dum2['imag'] - dum2['iKmag']) <= 0.05 )
	dum		= dum2[indx_stars]
	#	CHANGE TO GENERTAL COL. NAMES
	querytbl			= Table()
	querytbl['NUMBER']  = dum['objID']
	querytbl['RA_ICRS'] = dum['RAJ2000']
	querytbl['DE_ICRS'] = dum['DEJ2000']
	querytbl['Q']		= dum['Qual']
	querytbl['Numb_obs']= dum['Nd']
	querytbl['Numb_img']= dum['Ns']
	querytbl['gmag']    = dum['gmag']
	querytbl['e_gmag']  = dum['e_gmag']
	querytbl['rmag']    = dum['rmag']
	querytbl['e_rmag']  = dum['e_rmag']
	querytbl['imag']    = dum['imag']
	querytbl['e_imag']  = dum['e_imag']
	querytbl['zmag']    = dum['zmag']
	querytbl['e_zmag']  = dum['e_zmag']
	querytbl['ymag']    = dum['ymag']
	querytbl['e_ymag']  = dum['e_ymag']

	querytbl.write(path+'/'+outname, format='ascii.ecsv', overwrite=True)
	return querytbl
#-------------------------------------------------------------------------#
def twomass_query(name, radeg, dedeg, path, band=None, radius=1.0):
	"""
	QUERY Point Source Catalog(PSC) PROVIDED BY 2MASS
	REMOVE COMTAMINATED SOURCE BY EXTENDED SOURCE AND MINOR PLANET
	IF GIVE BAND INPUT, 

	INPUT	:	NAME, RA [deg], DEC [deg], BAND, RADIUS
	OUTPUT	:	TABLE, MAGNITUDE [AB]
	
	"""
	from astroquery.vizier import Vizier 
	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING 2MASS Catalog ...'+'\n'
	print(comment)
	outname	= '2mass-'+name+'.cat'
	#	QUERY PART
	Vizier.ROW_LIMIT    = -1
	query   = Vizier.query_region(coord.SkyCoord(ra=radeg, dec=dedeg, \
								unit=(u.deg, u.deg), frame='icrs'), \
								width=str(radius*60)+'m', catalog=["II/246"])
	dum0    = query[0]
	colnames= dum0.colnames
	['RAJ2000', 'DEJ2000', '_2MASS',
	'Jmag', 'e_Jmag', 'Hmag', 'e_Hmag', 'Kmag', 'e_Kmag',
	'Qflg', 'Rflg', 'Bflg', 'Cflg', 'Xflg', 'Aflg']
	#	REMOVE MASKED VALUE ROW
	for col in colnames:
		indx    = np.where( dum0[col].mask == False )
		dum1    = dum0[indx]
	#	SELECT POINT SOURCE & NON-VARIABLE & GOOD QUALITY STARS
	if	band	== None:
		indx_flg		= np.where(	(dum1['Aflg'] == 0) &
									(dum1['Xflg'] == 0)	)
	else:
		if band	== 'J':
			order	= 0
		if band	== 'H':
			order	= 1
		if band	== 'K':
			order	= 2
		indx_flg		= []
		for i in range(len(dum1)):
			Qflg	= dum1['Qflg'][i]
			Rflg	= dum1['Rflg'][i]
			Bflg	= dum1['Bflg'][i]
			Cflg	= dum1['Cflg'][i]
			Xflg	= dum1['Xflg'][i]
			Aflg	= dum1['Aflg'][i]
			if	(	(Qflg[order]	== 'A')		|
					(Qflg[order]	== 'B')		|
					(Qflg[order]	== 'C'))	& \
				(	(Bflg[order]	!= '0'))	& \
				(	(Cflg[order]	== '0'))	& \
				(	(Xflg			== 0))		& \
				(	(Aflg			== 0)	):
				indx_flg.append(i)
		indx_flg	= np.array( list(set(indx_flg)) )
	dum				= dum1[indx_flg]
	#	CHANGE TO GENERTAL COL. NAMES
	querytbl			= Table()
	querytbl['name']  	= dum['_2MASS']
	querytbl['ra'] 		= dum['RAJ2000']
	querytbl['dec'] 	= dum['DEJ2000']
	#	AB OFFSET
	querytbl['J']    	= dum['Jmag']	+ 0.91
	querytbl['Jerr']  	= dum['e_Jmag']
	querytbl['H']    	= dum['Hmag']	+ 1.39
	querytbl['Herr']  	= dum['e_Hmag']
	querytbl['K']    	= dum['Kmag']	+ 1.85
	querytbl['Kerr']  	= dum['e_Kmag']
	querytbl['Qflg']	= dum['Qflg']
	querytbl['Rflg']	= dum['Rflg']
	querytbl['Bflg']	= dum['Bflg']
	querytbl['Cflg']	= dum['Cflg']

	querytbl.write(path+'/'+outname, format='ascii.ecsv', overwrite=True)
	return querytbl
#-------------------------------------------------------------------------#
def nomad_query(name, radeg, dedeg, path, radius=1.0):
	from astroquery.vizier import Vizier 
	comment = 'NAME'+'\t'+': '+name+'\n' \
			+ 'RA'+'\t'+': '+str(round(radeg, 3))+'\n' \
			+ 'Dec'+'\t'+': '+str(round(dedeg, 3))+'\n' \
			+ 'Radius'+'\t'+': '+str(radius)+' deg'+'\n'*2 \
			+ 'LOADING NOMAD Catalog ...'+'\n'
	print(comment)
	outname = 'nomad-'+name+'.cat'
	Vizier.ROW_LIMIT    = -1
	query   = Vizier.query_region(coord.SkyCoord(ra=radeg, dec=dedeg, \
								unit=(u.deg, u.deg), frame='icrs'), \
								width=str(radius*60)+'m', catalog=["I/297/out"])
	querycat= query[query.keys()[0]]
	querycat.write(path+'/'+outname, format='ascii.ecsv', overwrite=True)
	return querycat
#-------------------------------------------------------------------------#
def sdss_Blaton(intbl, name):
	"""
	=====================================================================
	CONVERSION SDSS FILTER SYSTEM TO JOHNSON FILTER SYSTEM [AB]
	INPUT   :   QUERIED SDSS CATALOG
	OUTPUT  :   1.  ONLY STAR CLASS (=6)
				2.  (2)ACCEPTABLE & (3)GOOD QUALITY
				3.  NAME, RA, Dec, ugriz, BVRI [AB] (Blaton+07), clean
				sdss-conv.cat
	---------------------------------------------------------------------
	Blaton+07
	CONVERSION TABLE to AB
	---------------------------------------------------------------------
	Equation                                        Color Dispersion
	---------------------------------------------------------------------
	U   = u - 0.0682 - 0.0140[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	B   = u - 1.0286 - 0.7981[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	*B   = g + 0.2354 + 0.3915[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	V   = g - 0.3516 - 0.7585[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	R   = r - 0.0576 - 0.3718[ (r - i) - 0.2589 ]   sigma[r - i] = 0.10
	I   = i - 0.0647 - 0.7177[ (i - z) - 0.2083 ]   sigma[i - z] = 0.10
	=====================================================================
	"""

	outfile = 'sdss-Blaton-'+name+'.cat'
	
	clas, Q = intbl['class'],       intbl['Q']
	indx    = np.where( (Q != 1) & (clas == 6) )
	reftbl  = intbl[indx]
	
	name    = reftbl['SDSS12']
	ra, de  = reftbl['RA_ICRS'],    reftbl['DE_ICRS']
	clean   = reftbl['q_mode']

	u, uer  = reftbl['umag'],       reftbl['e_umag']
	g, ger  = reftbl['gmag'],       reftbl['e_gmag']
	r, rer  = reftbl['rmag'],       reftbl['e_rmag']
	i, ier  = reftbl['imag'],       reftbl['e_imag']
	z, zer  = reftbl['zmag'],       reftbl['e_zmag']

	uger, grer, rier, izer	= 0.26, 0.15, 0.10, 0.10
	ug, gr, ri, iz			= u-g, g-r, r-i, i-z
	'''
	U		= u - 0.0682 - 0.0140*(ug-1.2638)
	Uer		= np.sqrt( ((uer)**2.) + ((-0.0140*uger)**2.) )
	'''
	B		= g + 0.2354 + 0.3915*(gr-0.6102)
	Ber		= np.sqrt( ((ger)**2.) + ((0.3915*grer)**2.) )
	V		= g - 0.3516 - 0.7585*(gr-0.6102)
	Ver		= np.sqrt( ((ger)**2.) + ((-0.7585*grer)**2.) )
	R		= r - 0.0576 - 0.3718*(ri-0.2589)
	Rer		= np.sqrt( ((rer)**2.) + ((-0.3718*rier)**2.) )
	I		= i - 0.0647 - 0.7177*(iz-0.2083)
	Ier		= np.sqrt( ((ier)**2.) + ((-0.7177*izer)**2.) )

	outbl	= Table([name, ra, de, u, uer, g, ger, r, rer, i, ier, z, zer, B, Ber, V, Ver, R, Rer, I, Ier, clean], names=['name', 'ra', 'dec', 'u', 'uerr', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'z', 'zerr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr', 'clean'])
	
	outtbl0	= Table([name, ra, de, u, uer, g, ger, r, rer, i, ier, z, zer, B, Ber, V, Ver, R, Rer, I, Ier, clean], names=['#name', 'ra', 'dec', 'u', 'uerr', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'z', 'zerr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr', 'clean'])
	# outtbl0.write(outfile, format='ascii.ecsv', overwrite=True)
	return outbl, outfile
#-------------------------------------------------------------------------#
"""
def apass_Blaton(intbl, name):
	'''
	=====================================================================
	CONVERSION SDSS FILTER SYSTEM TO JOHNSON FILTER SYSTEM [AB]
	INPUT   :   QUERIED SDSS CATALOG
	OUTPUT  :   1.  ONLY STAR CLASS (=6)
				2.  (2)ACCEPTABLE & (3)GOOD QUALITY
				3.  NAME, RA, Dec, ugriz, BVRI [AB] (Blaton+07), clean
				sdss-conv.cat
	---------------------------------------------------------------------
	Blaton+07
	CONVERSION TABLE to AB
	---------------------------------------------------------------------
	Equation                                        Color Dispersion
	---------------------------------------------------------------------
	U   = u - 0.0682 - 0.0140[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	B   = u - 1.0286 - 0.7981[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	*B   = g + 0.2354 + 0.3915[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	V   = g - 0.3516 - 0.7585[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	R   = r - 0.0576 - 0.3718[ (r - i) - 0.2589 ]   sigma[r - i] = 0.10
	I   = i - 0.0647 - 0.7177[ (i - z) - 0.2083 ]   sigma[i - z] = 0.10
	=====================================================================
	'''

	
	outfile = 'apass-Blaton-'+name+'.cat'
	
	reftbl	= intbl
	
	name    = reftbl['NUMBER']
	ra, de  = reftbl['RA_ICRS'],    reftbl['DE_ICRS']
	Numb_obs= reftbl['Numb_obs']
	Numb_img= reftbl['Numb_img']
	B		= reftbl['Bmag']
	Ber		= reftbl['e_Bmag']
	V		= reftbl['Vmag']
	Ver		= reftbl['e_Vmag']	
	BV		= reftbl['B-V']
	e_BV	= reftbl['e_B-V']

	g, ger  = reftbl['gmag'],       reftbl['e_gmag']
	r, rer  = reftbl['rmag'],       reftbl['e_rmag']
	i, ier  = reftbl['imag'],       reftbl['e_imag']

	grer, rier		= 0.15, 0.10
	gr, ri			= g-r, r-i

	R		= r - 0.0576 - 0.3718*(ri-0.2589)
	Rer		= np.sqrt( ((rer)**2.) + ((-0.3718*rier)**2.) )
	'''
	I		= i - 0.0647 - 0.7177*(iz-0.2083)
	Ier		= np.sqrt( ((ier)**2.) + ((-0.7177*izer)**2.) )
	'''
	outbl	= Table([name, ra, de, Numb_obs, Numb_img, g, ger, r, rer, i, ier, B, Ber, V, Ver, R, Rer], names=['name', 'ra', 'dec', 'numb_obs', 'numb_img', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr'])
	
	outtbl0 = Table([name, ra, de, Numb_obs, Numb_img, g, ger, r, rer, i, ier, B, Ber, V, Ver, R, Rer], names=['#name', 'ra', 'dec', 'numb_obs', 'numb_img', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr'])
	# outtbl0.write(outfile, format='ascii.ecsv', overwrite=True)
	return outbl, outfile
"""
#-------------------------------------------------------------------------#
def apass_Blaton(intbl, name):
	"""
	=====================================================================
	CONVERSION SDSS FILTER SYSTEM TO JOHNSON FILTER SYSTEM [AB]
	INPUT   :   QUERIED SDSS CATALOG
	OUTPUT  :   1.  ONLY STAR CLASS (=6)
				2.  (2)ACCEPTABLE & (3)GOOD QUALITY
				3.  NAME, RA, Dec, ugriz, BVRI [AB] (Blaton+07), clean
				sdss-conv.cat
	---------------------------------------------------------------------
	Blaton+07
	CONVERSION TABLE to AB
	---------------------------------------------------------------------
	Equation                                        Color Dispersion
	---------------------------------------------------------------------
	U   = u - 0.0682 - 0.0140[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	B   = u - 1.0286 - 0.7981[ (u - g) - 1.2638 ]   sigma[u - g] = 0.26
	*B   = g + 0.2354 + 0.3915[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	V   = g - 0.3516 - 0.7585[ (g - r) - 0.6102 ]   sigma[g - r] = 0.15
	R   = r - 0.0576 - 0.3718[ (r - i) - 0.2589 ]   sigma[r - i] = 0.10
	I   = i - 0.0647 - 0.7177[ (i - z) - 0.2083 ]   sigma[i - z] = 0.10
	=====================================================================
	"""

	
	outfile = 'apass-Blaton-'+name+'.cat'
	
	reftbl	= intbl
	
	name    = reftbl['NUMBER']
	ra, de  = reftbl['RA_ICRS'],    reftbl['DE_ICRS']
	Numb_obs= reftbl['Numb_obs']
	Numb_img= reftbl['Numb_img']
	B		= reftbl['Bmag']
	Ber		= reftbl['e_Bmag']
	V		= reftbl['Vmag']
	Ver		= reftbl['e_Vmag']	
	BV		= reftbl['B-V']
	e_BV	= reftbl['e_B-V']

	g, ger  = reftbl['gmag'],       reftbl['e_gmag']
	r, rer  = reftbl['rmag'],       reftbl['e_rmag']
	i, ier  = reftbl['imag'],       reftbl['e_imag']

	grer, rier		= 0.15, 0.10
	gr, ri			= g-r, r-i

	R		= r - 0.0576 - 0.3718*(ri-0.2589)
	Rer		= np.sqrt( ((rer)**2.) + ((-0.3718*rier)**2.) )
	'''
	#	Blaton
	I		= i - 0.0647 - 0.7177*(iz-0.2083)
	Ier		= np.sqrt( ((ier)**2.) + ((-0.7177*izer)**2.) )
	'''

	#	Lupton
	Isig  = 0.0078

	I       = r - 1.2444*(r - i) - 0.3820
	Ier1    = ((1.-1.2444)**2)*(ger**2)+((+1.2444)**2)*(rer**2)
	Ier     = np.sqrt(Ier1**2 + Isig**2)
	#	Vega to AB
	I = I + 0.45

	outtbl	= Table([name, ra, de, Numb_obs, Numb_img, g, ger, r, rer, i, ier, B, Ber, V, Ver, R, Rer, I, Ier], names=['name', 'ra', 'dec', 'numb_obs', 'numb_img', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr'])
	
	outtbl0 = Table([name, ra, de, Numb_obs, Numb_img, g, ger, r, rer, i, ier, B, Ber, V, Ver, R, Rer, I, Ier], names=['#name', 'ra', 'dec', 'numb_obs', 'numb_img', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr'])
	# ascii.write(outtbl0, outfile)#, format='fixed_width', delimiter=' ')
	return outtbl, outfile
#-------------------------------------------------------------------------#
def ps1_Tonry(intbl, name):
	'''
	PS1 -> Johnson/COusins [Vega] -> [AB]	(Tonry+12)
	#   Vega - AB Magnitude Conversion (Blanton+07)
	U       : m_AB - m_Vega = 0.79
	B       : m_AB - m_Vega =-0.09
	V       : m_AB - m_Vega = 0.02
	R       : m_AB - m_Vega = 0.21
	I       : m_AB - m_Vega = 0.45
	J       : m_AB - m_Vega = 0.91
	H       : m_AB - m_Vega = 1.39
	K       : m_AB - m_Vega = 1.85
	'''

	#	REJECT BAD QUALITY
	intbl	= intbl[	(intbl['gmag']>12)	&
						(intbl['rmag']>12)	&
						(intbl['imag']>12)	&
						(intbl['zmag']>12)	&
						(intbl['ymag']>12)]


	outfile	= 'ps1-Tonry-'+name+'.cat'
	Q		= intbl['Q']
	indx    = np.where(Q < 128)

	intbl	= intbl[indx]
	Q		= intbl['Q']
	
	name    = intbl['NUMBER']
	ra, de  = intbl['RA_ICRS'],    intbl['DE_ICRS']

	g, ger  = intbl['gmag'],	intbl['e_gmag']
	r, rer  = intbl['rmag'],	intbl['e_rmag']
	i, ier  = intbl['imag'],	intbl['e_imag']
	z, zer  = intbl['zmag'],	intbl['e_zmag']
	y, yer  = intbl['ymag'],	intbl['e_ymag']
	#	TRANSF. ERROR FOR B CONST. TERMS
	Bsig, Vsig, Rsig, Isig	= 0.034, 0.012, 0.01, 0.016
	#	COLOR TERM
	gr		= intbl['gmag']-intbl['rmag']
	grer	= tool.sqsum(intbl['e_gmag'], intbl['e_rmag'])
	#	CONVERT TO B
	B0		= 0.213
	B1		= 0.587
	B		= B0 + B1*gr + intbl['gmag'] - 0.09
	Ber		= tool.sqsum( Bsig, tool.sqsum(B1*grer, intbl['e_gmag']) )
	#	CONVERT TO V
	B0		= 0.006
	B1		= 0.474
	V		= B0 + B1*gr + intbl['rmag'] + 0.02
	Ver	= tool.sqsum( Bsig, tool.sqsum(B1*grer, intbl['e_rmag']) )
	#	CONVERT TO R
	B0		=-0.138
	B1		=-0.131
	R		= B0 + B1*gr + intbl['rmag'] + 0.21
	Rer		= tool.sqsum( Rsig, tool.sqsum(B1*grer, intbl['e_rmag']) )
	#	CONVERT TO I
	B0		=-0.367
	B1		=-0.149
	I		= B0 + B1*gr + intbl['imag'] + 0.45
	Ier		= tool.sqsum( Isig, tool.sqsum(B1*grer, intbl['e_imag']) )
	outbl	= Table([name, ra, de, g, ger, r, rer, i, ier, z, zer, y, yer, B, Ber, V, Ver, R, Rer, I, Ier, Q], names=['name', 'ra', 'dec', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'z', 'zerr', 'y', 'yerr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr', 'Q'])
	outtbl0	= Table([name, ra, de, g, ger, r, rer, i, ier, z, zer, y, yer, B, Ber, V, Ver, R, Rer, I, Ier, Q], names=['#name', 'ra', 'dec', 'g', 'gerr', 'r', 'rerr', 'i', 'ierr', 'z', 'zerr', 'y', 'yerr', 'B', 'Berr', 'V', 'Verr', 'R', 'Rerr', 'I', 'Ierr', 'Q'])
	# outtbl0.write(outfile, format='ascii.ecsv', overwrite=True)
	return outbl, outfile
#-------------------------------------------------------------------------#
#
#	7DT
#
#-------------------------------------------------------------------------#
# def querybox(refcatname, obj, racent, decent, path_refcat, radius=0.5, refmagkey=''):
#-------------------------------------------------------------------------#
def calculate_snr(flux, flux_error):
    snr = flux / flux_error
    return snr
#-------------------------------------------------------------------------#
def calculate_magnitude_error(flux, flux_error):
    snr = calculate_snr(flux, flux_error)
    magnitude_error = 2.5*np.log10(1+(1/snr))    
    return magnitude_error
#-------------------------------------------------------------------------#
from gaiaxpy import generate, PhotometricSystem
from gaiaxpy import PhotometricSystem, load_additional_systems
from astroquery.gaia import GaiaClass
import pandas as pd
# path_to_filterset = '../config/filterset'
# PhotometricSystem = load_additional_systems(path_to_filterset)
# PhotometricSystem.get_available_systems().split(', ')[-2:]
#
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

#-------------------------------------------------------------------------#
#	Keys to Query
#-------------------------------------------------------------------------#
columns_to_query = [
	'gaia_source.solution_id',
	'gaia_source.source_id',
	'gaia_source.ra',
	'gaia_source.dec',
	'gaia_source.parallax',
	'gaia_source.l',
	'gaia_source.b',

	'gaia_source.phot_g_mean_mag',
	'gaia_source.bp_rp',
	'gaia_source.bp_g',
	'gaia_source.g_rp',

	'gaia_source.phot_variable_flag',
	'gaia_source.in_galaxy_candidates',
	'gaia_source.non_single_star',
	'gaia_source.has_xp_continuous',

	'gaia_source.has_rvs',
	# 'gaia_source.has_epoch_photometry',
	'gaia_source.ebpminrp_gspphot',
	'gaia_source.phot_bp_rp_excess_factor',
	'gaia_source.ruwe',
	'gaia_source.ipd_frac_multi_peak',
	'gaia_source.ipd_frac_odd_win',
	]

str_to_query = ', '.join(columns_to_query)
# print(f"{len(columns_to_query)} columns to query:")
# print(str_to_query)
#-------------------------------------------------------------------------#
def query_gaiaxp_continuous_raw(radeg, decdeg, query_radius, fname, mode='default'):
	#	Login to Gaia Server
	user = 'gpaek'
	password = 'Silenthill0!'
	# Connect to Gaia archive
	gaia = GaiaClass(
		gaia_tap_server='https://gea.esac.esa.int/',
		gaia_data_server='https://gea.esac.esa.int/',
		)
	gaia.login(
		user=user,
		password=password
	)

	# example_query = f"""SELECT {str_to_query}, DISTANCE({radeg},{decdeg}, ra, dec) AS ang_sep
	# FROM gaiadr3.gaia_source
	# WHERE DISTANCE({radeg},{decdeg}, ra, dec) < {query_radius}
	# AND has_epoch_photometry = 'True'
	# AND has_xp_continuous = 'True'
	# AND non_single_star = 0"""

	# example_query = f"""SELECT {str_to_query}, DISTANCE({radeg},{decdeg}, ra, dec) AS ang_sep
	# FROM gaiadr3.gaia_source
	# WHERE DISTANCE({radeg},{decdeg}, ra, dec) < {query_radius}
	# AND has_xp_continuous = 'True'
	# AND in_galaxy_candidates = 'False'
	# AND non_single_star = 0"""

	if mode == 'default':
		example_query = f"""SELECT {str_to_query}, DISTANCE({radeg},{decdeg}, ra, dec) AS ang_sep
		FROM gaiadr3.gaia_source
		WHERE DISTANCE({radeg},{decdeg}, ra, dec) < {query_radius}
		AND has_xp_continuous = 'True'
		AND ruwe < 1.4
		AND phot_variable_flag != 'VARIABLE'
		AND ipd_frac_multi_peak < 7
		AND ipd_frac_odd_win < 7
		"""
	elif mode == 'crowd':
		example_query = f"""SELECT {str_to_query}, DISTANCE({radeg},{decdeg}, ra, dec) AS ang_sep
		FROM gaiadr3.gaia_source
		WHERE DISTANCE({radeg},{decdeg}, ra, dec) < {query_radius}
		AND has_xp_continuous = 'True'
		AND ruwe < 1.4
		AND phot_variable_flag != 'VARIABLE'
		AND ipd_frac_multi_peak < 7
		AND ipd_frac_odd_win < 7
		AND phot_g_mean_mag < 15
		"""
	elif mode == 'too_crowd':
		example_query = f"""SELECT {str_to_query}, DISTANCE({radeg},{decdeg}, ra, dec) AS ang_sep
		FROM gaiadr3.gaia_source
		WHERE DISTANCE({radeg},{decdeg}, ra, dec) < {query_radius}
		AND has_xp_continuous = 'True'
		AND ruwe < 1.4
		AND phot_variable_flag != 'VARIABLE'
		AND ipd_frac_multi_peak < 7
		AND ipd_frac_odd_win < 7
		AND phot_g_mean_mag < 14
		"""


	#	Most Sources are variable... 
	# AND phot_variable_flag = 'CONSTANT'"""

	job = gaia.launch_job_async(example_query, dump_to_file=False)
	ids = job.get_results()

	# Now retrieve the BP/RP mean spectra in the continuous representation
	result = gaia.load_data(
		ids=ids['source_id'], 
		format='csv', 
		data_release='Gaia DR3', 
		data_structure='raw', 
		retrieval_type='XP_CONTINUOUS', 
		avoid_datatype_check=True
		)

	# Result will be a dictionary, so you can check the available keys by running result.keys() # In this example we are looking in particular for the XP_CONTINUOUS_RAW key
	continuous_key = [key for key in result.keys() if 'continuous' in key.lower()][0]

	# The first element is the result we want as an Astropy table
	data = result[continuous_key][0]

	# Astropy has a 'write' method for tables
	# Write the table to CSV
	data.write(fname, format='csv', overwrite=True)

	return ids.to_pandas()
#-------------------------------------------------------------------------#
# path_to_filterset = '/home/gp/gppy/config/filterset'
path_to_filterset = str(path_root / 'config/filterset')
PhotometricSystem = load_additional_systems(path_to_filterset)
#-------------------------------------------------------------------------#
def convert_gaiaxp_to_7DT_synphot(fname, metadf,):
	#
	#============================================================
	#	Synthetic Photometry
	#------------------------------------------------------------
	#	SDSS
	#------------------------------------------------------------
	phot_system_sdss = PhotometricSystem.SDSS
	synthetic_photometry_sdss = generate(fname, photometric_system=phot_system_sdss, save_file=False)
	synthetic_photometry_sdss
	#------------------------------------------------------------
	#	7DT
	#------------------------------------------------------------
	phot_system_7dt = PhotometricSystem.USER_7DT_Edmund
	synthetic_photometry_7dt = generate(fname, photometric_system=phot_system_7dt, save_file=False)
	synthetic_photometry_7dt
	#============================================================
	#	Merge the Table
	#------------------------------------------------------------
	#	Meta Table + SDSS + 7DT
	#------------------------------------------------------------
	synthetic_photometry = pd.merge(synthetic_photometry_sdss, synthetic_photometry_7dt, on='source_id')
	merged_df = pd.merge(synthetic_photometry, metadf, on='source_id')
	#------------------------------------------------------------
	#	Filter List
	#------------------------------------------------------------
	filterlist_sdss = ['u', 'g', 'r', 'i', 'z',]
	filterlist_7dt = []
	for key in merged_df.keys():
		if 'USER_7DT_Edmund_mag_' in key:
			filte = key.replace('USER_7DT_Edmund_mag_', '')
			filterlist_7dt.append(filte)
	#------------------------------------------------------------
	#	Rename configurations
	#------------------------------------------------------------
	#	SDSS
	prefix_mag_sdss = 'Sdss_mag_'
	prefix_flux_sdss = 'Sdss_flux_'
	prefix_fluxerr_sdss = 'Sdss_flux_error_'
	#	7DT
	prefix_mag_7dt = 'USER_7DT_Edmund_mag_'
	prefix_flux_7dt = 'USER_7DT_Edmund_flux_'
	prefix_fluxerr_7dt = 'USER_7DT_Edmund_flux_error_'
	#------------------------------------------------------------
	#	Rename the columns
	#------------------------------------------------------------
	key_to_rename_dict = {}
	#	SDSS
	for filte in filterlist_sdss:
		#	New Values
		# _mag = merged_df[f"{prefix_mag_sdss}{filte}"]
		_flux = merged_df[f"{prefix_flux_sdss}{filte}"]
		_fluxerr = merged_df[f"{prefix_fluxerr_sdss}{filte}"]
		_snr = calculate_snr(_flux, _fluxerr)
		_magerr = calculate_magnitude_error(_flux, _fluxerr)

		#	Former Keywords
		magkey = f"{prefix_mag_sdss}{filte}"
		fluxkey = f"{prefix_flux_sdss}{filte}"
		fluxerrkey = f"{prefix_fluxerr_sdss}{filte}"

		#	New Keywords
		newmagkey = f"{filte}_mag"
		newmagerrkey = f"{filte}_magerr"
		newfluxkey = f"{filte}_flux"
		newfluxerrkey = f"{filte}_fluxerr"
		newsnrkey = f"{filte}_snr"

		#	Keywords to Rename
		key_to_rename_dict[magkey] = newmagkey
		key_to_rename_dict[fluxkey] = newfluxkey
		key_to_rename_dict[fluxerrkey] = newfluxerrkey

		merged_df[newmagerrkey] = _magerr
		merged_df[newsnrkey] = _snr

	#	7DT
	# for filte in filterlist_7dt:
	# 	#	New Values
	# 	# _mag = merged_df[f"{prefix_mag_7dt}{filte}"]
	# 	_flux = merged_df[f"{prefix_flux_7dt}{filte}"]
	# 	_fluxerr = merged_df[f"{prefix_fluxerr_7dt}{filte}"]
	# 	_snr = calculate_snr(_flux, _fluxerr)
	# 	_magerr = calculate_magnitude_error(_flux, _fluxerr)

	# 	#	Former Keywords
	# 	magkey = f"{prefix_mag_7dt}{filte}"
	# 	fluxkey = f"{prefix_flux_7dt}{filte}"
	# 	fluxerrkey = f"{prefix_fluxerr_7dt}{filte}"

	# 	#	New Keywords
	# 	if '_50' in filte:
	# 		replace_key = '_50'
	# 		suffix = 'w'
	# 	elif '_25' in filte:
	# 		replace_key = '_25'
	# 		suffix = ''
	# 	else:
	# 		replace_key = ''
	# 		suffix = ''

	# 	newmagkey = f"m{filte.replace(replace_key, suffix)}_mag"
	# 	newmagerrkey = f"m{filte.replace(replace_key, suffix)}_magerr"
	# 	newfluxkey = f"m{filte.replace(replace_key, suffix)}_flux"
	# 	newfluxerrkey = f"m{filte.replace(replace_key, suffix)}_fluxerr"
	# 	newsnrkey = f"m{filte.replace(replace_key, suffix)}_snr"

	# 	#	Keywords to Rename
	# 	key_to_rename_dict[magkey] = newmagkey
	# 	key_to_rename_dict[fluxkey] = newfluxkey
	# 	key_to_rename_dict[fluxerrkey] = newfluxerrkey

	# 	merged_df[newmagerrkey] = _magerr
	# 	merged_df[newsnrkey] = _snr

	new_columns = {}

	# 7DT
	for filte in filterlist_7dt:
		# New Values
		_flux = merged_df[f"{prefix_flux_7dt}{filte}"]
		_fluxerr = merged_df[f"{prefix_fluxerr_7dt}{filte}"]
		_snr = calculate_snr(_flux, _fluxerr)
		_magerr = calculate_magnitude_error(_flux, _fluxerr)

		# Former Keywords
		magkey = f"{prefix_mag_7dt}{filte}"
		fluxkey = f"{prefix_flux_7dt}{filte}"
		fluxerrkey = f"{prefix_fluxerr_7dt}{filte}"

		# New Keywords
		if '_50' in filte:
			replace_key = '_50'
			suffix = 'w'
		elif '_25' in filte:
			replace_key = '_25'
			suffix = ''
		else:
			replace_key = ''
			suffix = ''

		newmagkey = f"m{filte.replace(replace_key, suffix)}_mag"
		newmagerrkey = f"m{filte.replace(replace_key, suffix)}_magerr"
		newfluxkey = f"m{filte.replace(replace_key, suffix)}_flux"
		newfluxerrkey = f"m{filte.replace(replace_key, suffix)}_fluxerr"
		newsnrkey = f"m{filte.replace(replace_key, suffix)}_snr"

		# Keywords to Rename
		key_to_rename_dict[magkey] = newmagkey
		key_to_rename_dict[fluxkey] = newfluxkey
		key_to_rename_dict[fluxerrkey] = newfluxerrkey

		new_columns[newmagerrkey] = _magerr
		new_columns[newsnrkey] = _snr

	# 새로운 열을 한 번에 추가
	merged_df = pd.concat([merged_df, pd.DataFrame(new_columns)], axis=1)

	#------------------------------------------------------------
	#	Final Dataframe
	#------------------------------------------------------------
	merged_df.rename(columns=key_to_rename_dict, inplace=True)
	return merged_df
#-------------------------------------------------------------------------#
def querybox_7dt(radeg, decdeg, query_radius, fname, verbose=True, mode='default'):
	if verbose:
		print(f"="*60)
		print(f"Query Gaia DR3")
		print(f"-"*60)
		print(f"Output Name: {fname}")
		print(f"RA, Dec    : {radeg:.3f} deg, {decdeg:.3f} deg (r={query_radius:.3f} deg)")
		print(f"-"*60)
	metadf = query_gaiaxp_continuous_raw(radeg, decdeg, query_radius, fname, mode)
	print(f"\n{len(metadf)} sources are queried\n")
	print(f"Convert Gaia XP Spectra to Synthetic Photometries")
	_refdf = convert_gaiaxp_to_7DT_synphot(fname, metadf,)
	reftbl = Table.from_pandas(_refdf)
	print(f"\nDone\n")
	return reftbl
#-------------------------------------------------------------------------#


def merge_catalogs(target_coord, path_calibration_field, matching_radius=1.0, path_save='./ref.cat'):
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
	c_grid = SkyCoord(grid_table['center_ra'], grid_table['center_dec'], unit='deg')

	# 매칭 반경 내에서 grid 찾기
	sep_arr = target_coord.separation(c_grid).deg
	indx_match = np.where(sep_arr < matching_radius)
	matched_grid_table = grid_table[indx_match]

	all_filters = [
		'u',
		'g',
		'r',
		'i',
		'z',
		'm375w',
		'm400',
		'm412',
		'm425',
		'm425w',
		'm437',
		'm450',
		'm462',
		'm475',
		'm487',
		'm500',
		'm512',
		'm525',
		'm537',
		'm550',
		'm562',
		'm575',
		'm587',
		'm600',
		'm612',
		'm625',
		'm637',
		'm650',
		'm662',
		'm675',
		'm687',
		'm700',
		'm712',
		'm725',
		'm737',
		'm750',
		'm762',
		'm775',
		'm787',
		'm800',
		'm812',
		'm825',
		'm837',
		'm850',
		'm862',
		'm875',
		'm887',
		]

	gaia_general_keys = [
		'source_id',
		'ra',
		'dec',
		'parallax',
		# 'parallax_over_error', # TBD
		'pmra',
		'pmdec',
		'phot_g_mean_mag',
		# 'phot_bp_mean_mag', # TBD
		# 'phot_rp_mean_mag', # TBD
		'bp_rp',
	]

	all_tablelist = []
	for prefix in matched_grid_table['prefix']:
		tablelist = []
		for ff, filte in enumerate(all_filters):
			_tablename = f"{path_calibration_field}/{prefix}/{filte}.fits"
			_reftbl = Table.read(_tablename)
			if ff == 0:
				_table = Table()
				for gaia_key in gaia_general_keys:
					_table[gaia_key] = _reftbl[gaia_key]
			#	Mag & SNR Keys
			filter_magkey = f"mag_{filte}"
			filter_snrkey = f"snr_{filte}"
			_table[filter_magkey] = _reftbl[filter_magkey]
		tablelist.append(_table)
		all_tablelist.append(hstack(tablelist))

	all_reftbl = vstack(all_tablelist)
	all_reftbl = unique(all_reftbl, keys='source_id')

	# if not os.path.exists(path_save):
	# 	all_reftbl.write(path_save, overwrite=True)

	return all_reftbl