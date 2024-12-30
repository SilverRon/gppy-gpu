#%%
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from astropy.table import Table
from spherical_geometry.polygon import SphericalPolygon
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from scipy.optimize import minimize
import os
from astropy.io import fits
from astropy.io import ascii
from glob import glob
# import sds as dhutil
import dhutil as dh
import healpy as hp
from time import time
from tqdm import tqdm

#%%


def tile_vertex(tile_num):
    tile_vertice = dh.get_tiles(tile_num)
    tile_vertice = pd.DataFrame([tile_vertice], columns=tile_vertice.columns)
    indice = sorted(set([i[-1] for i in tile_vertice.columns]))

    for i in indice:
        sel_df = tile_vertice.filter(regex=f'{i}$', axis=1)
        sel_df = sel_df.rename(columns={key:key.replace(i, '') for key in list(sel_df.columns)})
        sel_df.index = [i]
        if i == indice[0]:
            idx_df = sel_df
        else:
            idx_df = pd.concat([idx_df, sel_df])
    
    return idx_df


def is_point_in_tile(point, tile_vertices):
    """
    Check if a point (SkyCoord) is inside a tile defined by its vertices.

    :param point: SkyCoord, the point to check.
    :param tile_vertices: List of SkyCoord, vertices of the tile in order.
    :return: Boolean, True if the point is inside the tile.
    """
    # Convert vertices to SphericalPolygon
    polygon = SphericalPolygon.from_lonlat(
        [v.ra.deg for v in tile_vertices],
        [v.dec.deg for v in tile_vertices]
    )

    # Check if the point is inside the polygon
    return polygon.contains_lonlat(point.ra.deg, point.dec.deg)


def load_tiles():
    table = dh.get_tiles()

    _tiles = []
    for i, row in enumerate(table):
        # Create SkyCoord objects for the four vertices
        vertices = [
            SkyCoord(ra=row[f"ra{j}"], dec=row[f"dec{j}"], unit='deg')
            for j in range(1, 5)
        ]
        # Append the tile dictionary to the list
        _tiles.append({
            'tile_id': i,  # Tile ID starting from 1
            'vertices': vertices
        })
    return _tiles


def find_tile_for_point(point, tiles):
    """
    Find which tile a point belongs to.

    :param point: SkyCoord, the point to check.
    :param tiles: List of dictionaries, where each dictionary contains:
                  - 'tile_id': int, identifier for the tile
                  - 'vertices': List of SkyCoord, vertices of the tile
    :return: The tile_id of the tile containing the point, or None if not found.
    """
    
    for tile in tiles:
        tile_id = tile['tile_id']
        tile_vertices = tile['vertices']

        if is_point_in_tile(point, tile_vertices):
            return tile_id
    return None


def is_points_in_ellipse_skycoord(points, ra_center, dec_center, a, b, theta):
    # Convert center and points to SkyCoord
    center = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg, frame='icrs')
    points = SkyCoord(ra=points[:, 0]*u.deg, dec=points[:, 1]*u.deg, frame='icrs')
    
    # Compute offsets in a rotated frame
    separation = points.separation(center).radian
    position_angle = points.position_angle(center).radian - np.radians(theta)
    
    # Convert to ellipse frame
    x = separation * np.cos(position_angle)
    y = separation * np.sin(position_angle)
    
    # Check ellipse condition
    return (x / a)**2 + (y / b)**2 <= 1


# minimum volume enclosing ellipse
def mvee(points, tol=1e-3):
    """
    Find the minimum volume enclosing ellipse (MVEE) for a set of 2D points.
    
    Parameters:
    - points: (n, 2) array of 2D points.
    - tol: Tolerance for stopping criterion.
    
    Returns:
    - Center of the ellipse.
    - Radii (semi-axes lengths) of the ellipse.
    - Rotation matrix of the ellipse.
    """
    points = np.array(points)
    n, d = points.shape
    Q = np.column_stack((points, np.ones(n)))
    
    # Initializations
    err = tol + 1.0
    u = np.ones(n) / n
    
    # Khachiyan Algorithm
    while err > tol:
        X = Q.T @ np.diag(u) @ Q
        M = np.diag(Q @ np.linalg.inv(X) @ Q.T)
        j = np.argmax(M)
        step_size = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # Center of the ellipse
    center = u @ points
    
    # Ellipse parameters
    A = np.linalg.inv(points.T @ np.diag(u) @ points - np.outer(center, center)) / d
    U, s, rotation = np.linalg.svd(A)
    radii = 1.0 / np.sqrt(s)
    
    return center, radii, rotation


def read_healpix_gaia(fpath, extract=True):
    # load full column names
    colname_path = '/lyman/data1/factory/catalog/gaia_source_dr3/column_names.txt'
    with open(colname_path, 'r') as f:
        colnames = [line.strip() for line in f]

    df = pd.read_csv(fpath, header=None, dtype={151: 'str'}, names=colnames, low_memory=False)
    if not extract:
        return df
    
    # extract useful columns
    columns_to_extract = [
        "ra",
        "dec",
        "ra_error",
        "dec_error",
        "pmra",
        "pmdec",
        "pmra_error",
        "pmdec_error",
        "phot_g_mean_flux",
        "phot_g_mean_flux_error",
        "phot_g_mean_mag",  # CAUTION: its error should be derived later
        "ref_epoch"
    ]

    df = df[columns_to_extract]

    # cal mag err
    df['phot_g_mean_mag_error'] = 2.5 / np.log(10) * df['phot_g_mean_flux_error'] / df['phot_g_mean_flux']
    # drop flux cols
    df = df.drop(columns=['phot_g_mean_flux_error', 'phot_g_mean_flux'])
    # reorder columns
    cols = [col for col in df.columns if col != 'ref_epoch'] + ['ref_epoch']
    df = df[cols]
    return df


def find_healpix_tiles(tile_idx, matching_r):

    GAIA_ROOT_DIR = '/lyman/data1/factory/catalog/gaia_source_dr3/healpix_nside64'
    
    # load healpix ids
    files = sorted(glob(f"{GAIA_ROOT_DIR}/tile_*.csv"))
    # ipix_list = [s.split('_')[-1].replace('.csv', '') for s in files]
    ipix_list = [int(Path(s).stem.split('_')[1]) for s in files]
    radec = np.array([hp.pix2ang(64, ipix, nest=True, lonlat=True) for ipix in ipix_list])
    heal_ra =  radec[:,0]
    heal_dec = radec[:,1]

    # Get center of 7DT tile and set matching radius
    center = dh.get_tiles(tile_idx, center=True)

    # find healpix tiles
    reference_coord = SkyCoord(ra=center['ra']*u.deg, dec=center['dec']*u.deg, frame='icrs')
    heal_coords = SkyCoord(ra=heal_ra*u.deg, dec=heal_dec*u.deg, frame='icrs')
    distances = reference_coord.separation(heal_coords)
    matched_files = np.array(files)[distances < matching_r*u.deg]
    
    return matched_files


def get_abpa_from_mvee(center, radii, rotation):
    '''unused'''

    semi_major_idx = np.argmax(radii)  # Index of the semi-major axis
    semi_minor_idx = 1 - semi_major_idx  # Index of the semi-minor axis

    semi_major = radii[semi_major_idx]
    semi_minor = radii[semi_minor_idx]

    # Ensure the position angle corresponds to the semi-major axis
    if semi_major_idx == 0:
        position_angle = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        position_angle = np.degrees(np.arctan2(rotation[1, 1], rotation[0, 1]))
    return center, semi_major, semi_minor, position_angle


def select_sources_in_ellipse(sources_df, center, a, b, pa):
    # Convert source positions to SkyCoord objects
    source_coords = SkyCoord(ra=sources_df['ra'].values, dec=sources_df['dec'].values, unit='deg')

    # Calculate separations and position angles relative to the center
    sep = center.separation(source_coords)  # .deg  rad
    pas = center.position_angle(source_coords)  # .deg  rad
    das = pas - Angle(pa, unit='deg')

    # # Convert position angles to the rotated frame of the ellipse
    # delta_angle = position_angles - position_angle
    # x_prime = separations * np.cos(np.radians(delta_angle))
    # y_prime = separations * np.sin(np.radians(delta_angle))
    # inside_ellipse = (x_prime / semi_major)**2 + (y_prime / semi_minor)**2 <= 1

    sep_ellip = np.linalg.norm([a * np.cos(das), b * np.sin(das)], axis=0)
    inside_ellipse = sep.deg < sep_ellip  # comparison in deg
    return sources_df[inside_ellipse]


def _plot_example(tile_idx=1000):
    tile_points = tile_vertex(tile_idx)  # get 7DT tile vertices
    tile_points = np.array(tile_points)
    center, radii, rotation = mvee(tile_points)

    cen = dh.get_tiles(tile_idx, center=True)
    plt.scatter(cen['ra'], cen['dec'], s=100)

    matched_files = find_healpix_tiles(tile_idx, 2)
    print(len(matched_files))
    dfs = []
    for f in matched_files:
        df = read_healpix_gaia(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    plt.scatter(df['ra'], df['dec'], s=1, alpha=.1)
    dh.overlay_tiles()

    # Generate points on a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.array([np.cos(theta), np.sin(theta)])  # Shape (2, 100)

    # Scale by the radii (ellipse axes lengths)
    ellipse = np.diag(radii) @ circle  # Shape (2, 100)

    # Rotate by the rotation matrix
    ellipse = rotation @ ellipse

    # Shift by the center
    ellipse[0, :] += center[0]
    ellipse[1, :] += center[1]

    # Plot the points and the ellipse
    plt.plot(ellipse[0, :], ellipse[1, :], 'r-', linewidth=2, label="Minimum Enclosing Ellipse")
    plt.plot(center[0], center[1], 'ro', label="Center")
    return


def save_as_fits_ldac(outbl, tablename):
    hdu_list = fits.HDUList()

    primary_hdu = fits.PrimaryHDU()
    hdu_list.append(primary_hdu)

    # bintable_hdu = fits.BinTableHDU(outbl.as_array(), name='LDAC_OBJECTS')
    bintable_hdu = fits.BinTableHDU(outbl, name='LDAC_OBJECTS')
    hdu_list.append(bintable_hdu)

    hdu_list.writeto(tablename, overwrite=True)


# def transform_to_new_center(skycoords, center_ra, center_dec):
#     """
#     Transform SkyCoord objects to a new coordinate system centered at the given RA and Dec.
    
#     Parameters:
#     skycoords (SkyCoord): Input SkyCoord objects to transform.
#     center_ra (float): Right Ascension of the new center in degrees.
#     center_dec (float): Declination of the new center in degrees.
    
#     Returns:
#     SkyCoord: Transformed SkyCoord objects in the new system.
#     """
#     from astropy.coordinates import SkyCoord, frame_transform_graph, BaseCoordinateFrame
#     from astropy.coordinates.matrix_utilities import rotation_matrix
#     import astropy.units as u
    
#     # Define the new center
#     new_center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg, frame='icrs')

#     # Convert the center to Cartesian
#     new_center_cartesian = new_center.cartesian.xyz
#     print(new_center, new_center_cartesian)

#     # Create rotation matrices for the transformation
#     dec_rotation = rotation_matrix(-center_dec * u.deg, axis='x')
#     ra_rotation = rotation_matrix(-center_ra * u.deg, axis='z')
#     total_rotation = dec_rotation @ ra_rotation 

#     # Transform the input coordinates to Cartesian
#     skycoords_cartesian = skycoords.icrs.cartesian.xyz

#     # Apply rotation matrix to all points
#     rotated_coords = total_rotation @ skycoords_cartesian

#     # Convert rotated Cartesian coordinates back to spherical
#     transformed_coords = SkyCoord(
#         x=rotated_coords[0], 
#         y=rotated_coords[1], 
#         z=rotated_coords[2], 
#         representation_type='cartesian',
#         frame='icrs'
#     ).spherical

#     # Normalize RA to be close to 0 degrees
#     transformed_ra = (transformed_coords.lon.deg + 180) % 360 - 180
#     transformed_dec = transformed_coords.lat.deg

#     return transformed_ra, transformed_dec


def run_single(tile_idx, matching_r=4, show=False):
    # matching_r  in deg
    tablename = f'/lyman/data1/factory/catalog/gaia_dr3_7DT/T{tile_idx:05}.fits'
    if os.path.isfile(tablename):
        return

    fpaths = find_healpix_tiles(tile_idx, matching_r)  # relevant hp tile filenames
    # load all as a single df
    dfs = []
    for f in fpaths:
        df = read_healpix_gaia(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    del dfs

    center = dh.get_tiles(tile_idx, center=True)
    center = SkyCoord(ra=center['ra'], dec=center['dec'], unit='deg')
    semi_major = 0.98
    semi_minor = 0.82
    position_angle = 0
    if tile_idx == 0:
        position_angle = 90
    df_sel = select_sources_in_ellipse(df, center, semi_major, semi_minor, position_angle)

    # ra dec err mas to deg
    df_sel.loc[:, 'ra_error'] = df_sel['ra_error'] / (3600 * 10**3)
    df_sel.loc[:, 'dec_error'] = df_sel['dec_error'] / (3600 * 10**3)

    if show:
        import ligo.skymap.plot

        plt.figure(dpi=300)
        ax = plt.axes(projection='astro zoom', center='9h -90d', radius='5 deg')

        ra, dec = (df_sel['ra'], df_sel['dec'])
        ax.scatter(ra, dec, transform=ax.get_transform('world'))
        ax.grid()
        ax.coords[0].set_format_unit(u.deg)  # ra axis hour to deg

        # dh.set_xylim(ax, 129, 141, -4.5, 5)
        dh.overlay_tiles(fontsize = 6, color='k', fontweight='bold')
        plt.xlabel('RA (deg)')
        plt.ylabel('Dec (deg)')

    outbl = Table.from_pandas(df_sel)  # to astropy Table
    
    save_as_fits_ldac(outbl, tablename)
    del df_sel
    del outbl


#%%

if __name__ == '__main__':
    import multiprocessing

    # run_single(0, show=True)
    start = time()
    tiles = list(dh.get_tiles(center=True)['id'])

    nthread = 96
    with multiprocessing.Pool(processes=nthread) as pool:
        # Map the function across the tiles
        # results = pool.map(run_single, tiles)
        results = list(tqdm(pool.imap(run_single, tiles), total=len(tiles)))

    print(time() - start, 's elapsed')