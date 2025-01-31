def phot(path_data, path_default_gphot, path_phot_mp, ncore, n_binning, timetbl):
    import os
    import time
    
    #------------------------------------------------------------
    t0_photometry = time.time()
    print('#\tPhotometry')
    path_infile = f'{path_data}/{os.path.basename(path_default_gphot)}'
    path_new_gphot = f'{os.path.dirname(path_infile)}/gphot.config'

    #------------------------------------------------------------
    #	Photometry
    #------------------------------------------------------------

    #	Copy default photometry configuration
    cpcom = f'cp {path_default_gphot} {path_new_gphot}'
    print(cpcom)
    os.system(cpcom)

    #	Read default photometry configuration
    f = open(path_default_gphot, 'r')
    lines = f.read().splitlines()
    f.close()

    #	Write photometry configuration for a single exposure frame
    g = open(path_new_gphot, 'w')
    for line in lines:
        if 'imkey' in line:
            line = f'imkey\t{path_data}/calib*0.fits'
        else:
            pass
        g.write(line+'\n')
    g.close()

    path_phot = path_phot_mp
    #	Execute
    #	(e.g. com = f'python {path_phot} {path_data} 1')
    com = f'python {path_phot} {path_data} {ncore} {n_binning}'
    print(com)
    os.system(com)

    delt_photometry = time.time() - t0_photometry
    timetbl['status'][timetbl['process']=='photometry'] = True
    timetbl['time'][timetbl['process']=='photometry'] = delt_photometry

    return path_new_gphot, timetbl, lines


def phot_com(path_new_gphot, lines, path_data, path_phot_mp, ncore, n_binning, timetbl):
    # ## Photometry for combined images
    import os
    import time

    t0_photometry_com = time.time()
    #	Write photometry configuration
    h = open(path_new_gphot, 'w')
    for line in lines:
        if 'imkey' in line:
            line = f'imkey\t{path_data}/c*com.fits'
        else:
            pass
        h.write(line+'\n')
    h.close()

    #	Execute
    path_phot = path_phot_mp
    com = f'python {path_phot} {path_data} {ncore} {n_binning}'
    print(com)
    os.system(com)

    delt_photometry_com = time.time() - t0_photometry_com
    timetbl['status'][timetbl['process']=='photometry_com'] = True
    timetbl['time'][timetbl['process']=='photometry_com'] = delt_photometry_com

    return timetbl
