import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from spherical_geometry.polygon import SphericalPolygon
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from scipy.optimize import minimize
import os
from astropy.io import ascii
import ligo.skymap.plot
import glob
import sds as dhutil
import healpy as hp
import time
import argparse

GAIA_ROOT_DIR = '/lyman/data1/factory/catalog/gaia_source_dr3/healpix_nside64'

def path_set():
    path_thisfile = Path(os.path.abspath('')).resolve()
    path_src = path_thisfile.parent  # absolute path of dhutil
    path_root = path_src.parent  # parent of dhutil
    if path_root not in map(Path, sys.path):
        sys.path.append(str(path_root))
    return path_src


def get_tiles(indices, center=False):
    path_src = path_set()
    if center:
        center = ascii.read(str(path_src / "displaycenter.txt"))
        # center = ascii.read("/Users/dhhyun/VSCode/7DT/displaycenter.txt")
        return center[indices]
    vertices = ascii.read(str(path_src / "displayfootprint.txt"))
    # vertices = ascii.read("/Users/dhhyun/VSCode/7DT/displayfootprint.txt")
    return vertices[indices]


def tile_vertex(tile_num):
    tile_vertice = get_tiles(tile_num)
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
    table = ascii.read(str(path_set() / "displayfootprint.txt"))

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

def find_healpix_tile(tile_idx, matching_r, verbose=False):
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

    savepath = '/lyman/data1/factory/catalog/gaia_source_dr3/column_names.txt'
    with open(savepath, 'r') as f:
        colnames = [line.strip() for line in f]
    
    # get 7DT tile vertices
    idx_df = tile_vertex(tile_idx)
    
    # load healpix ids
    files = sorted(glob.glob(f"{GAIA_ROOT_DIR}/tile*.csv"))
    ipix_list = [s.split('_')[-1].replace('.csv', '') for s in files]
    dec_ra = [hp.pix2ang(64, int(ipix), nest=True, lonlat=True) for ipix in ipix_list]
    
    heal_df = pd.DataFrame(columns=['path', 'ra', 'dec'])
    heal_df['path'] = files
    heal_df['ra'] = np.array(dec_ra)[:,0]
    heal_df['dec'] = np.array(dec_ra)[:,1]

    heal_ra = heal_df['ra'].to_numpy()
    heal_dec = heal_df['dec'].to_numpy()
    heal_points = np.column_stack((heal_ra, heal_dec))

    # Get center of 7DT tile and set matching radius
    ra_center = get_tiles(tile_idx, center=True)['ra']
    dec_center = get_tiles(tile_idx, center=True)['dec']
    a = matching_r * np.pi / 180  # [deg]
    b = matching_r * np.pi / 180  # [deg]

    # find healpix tiles
    inside_h_mask = is_points_in_ellipse_skycoord(heal_points, ra_center, dec_center, a, b, theta=0)
    inside_h_points = heal_points[inside_h_mask]
    
    # merging selected healpix tiles
    for i, coord in enumerate(inside_h_points):
        co_ra = coord[0]; co_dec = coord[1]
        
        # Convert RA/Dec to HEALPix pixel index in nested ordering
        _idx = hp.ang2pix(64, co_ra, co_dec, nest=True, lonlat=True)
        if verbose: print(f'Matched Healpix ID : {_idx}')
        try:
            temp_df = pd.read_csv(f'{GAIA_ROOT_DIR}/tile_{_idx}.csv', dtype={151: 'str'}, names=colnames)
        except FileNotFoundError:  # exception for -0 > 0
            temp_df = pd.read_csv(f'{GAIA_ROOT_DIR}/tile_{_idx}.csv', dtype={151: 'str'}, names=colnames)
        
        if i == 0:
            dr3 = temp_df
        else:
            dr3 = pd.concat([dr3, temp_df], axis=0)
        # returns merged healpix tile sources
    
    # elliptical fitting for 7DT tile
    tile_ra = np.array(idx_df['ra'])
    tile_dec = np.array(idx_df['dec'])
    tile_points = np.vstack((tile_ra, tile_dec)).T
    center, radii, rotation = mvee(tile_points)

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
    
    ra = dr3['ra'].to_numpy()
    dec = dr3['dec'].to_numpy()
    points = np.column_stack((ra, dec))

    ra_center = center[0]
    dec_center = center[1]
    a = (radii[0] + 0.1) * np.pi / 180
    b = (radii[1] + 0.1) * np.pi / 180
    theta = 90 - np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))

    inside_mask = is_points_in_ellipse_skycoord(points, ra_center, dec_center, a, b, theta)
    inside_points = dr3[inside_mask]

    df = inside_points[columns_to_extract]
    df['phot_g_mean_mag_error'] = 2.5 / np.log(10) * df['phot_g_mean_flux_error'] / df['phot_g_mean_flux']
    # drop flux cols
    df = df.drop(columns=['phot_g_mean_flux_error', 'phot_g_mean_flux'])
    # reorder
    cols = [col for col in df.columns if col != 'ref_epoch'] + ['ref_epoch']
    df = df[cols]
    
    if verbose:
        print(f"Total points Number: {len(points)}")
        print(f"Total points inside the ellipse: {len(inside_points)}")
    return df.reset_index()


if __name__ == '__main__':
    # Default tile & matching radius
    id_7dt = 0
    mat_r = 2.0
    verbose = False

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile", "-t", type=int, nargs='?', help="Tile number in 7DS")
    parser.add_argument("--radius", "-r", type=float, nargs='?', help="Matching radius for Healpix tiles")
    parser.add_argument("--verbose", "-v", type=bool, nargs='?', help="Print outputs")

    args = parser.parse_args()
    if args.tile:
        id_7dt = args.tile
    if args.radius:
        mat_r = args.radius
    if args.verbose:
        verbose = args.verbose

    # matching tiles
    start_time = time.time()
    table = find_healpix_tile(tile_idx=id_7dt, matching_r=mat_r, verbose=verbose)
    print(table)
    time_end = time.time() - start_time
    print(f'Time : {time_end} sec')

    # print the figures for visual inspection
    ra_center = Angle(get_tiles(id_7dt, center=True)['ra'] * u.deg).hour
    dec_center = get_tiles(id_7dt, center=True)['dec']

    plt.figure(dpi=100)
    ax = plt.axes(projection="astro zoom", center=f"{ra_center}h {dec_center}d", radius="2 deg")
    ax.scatter(table.ra, table.dec, linewidth=0, s=3, color='#33333388', label='GAIA sources', transform=ax.get_transform("world"))
    ax.grid()
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    ax.legend()
    ax.set_title(f'7DT Tile: {id_7dt}, R = {mat_r}')
    # ax.coords[0].set_format_unit(u.deg)
    dhutil.overlay_tiles(fontsize=10, color="b", fontweight="bold")
    plt.savefig(f'testfield_{id_7dt}_{mat_r}.png')

    # for i in [int(s.split('_')[1].replace('.csv', '')) for s in sorted(glob.glob("files"))]:
    #     find_healpix_tile(tile_idx=i, matching_r=1.5)