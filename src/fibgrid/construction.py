# Copyright (c) 2021, TU Wien, Department of Geodesy and Geoinformation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of TU Wien, Department of Geodesy and Geoinformation
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL TU WIEN DEPARTMENT OF GEODESY AND
# GEOINFORMATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Construct Fibonacci grid.
"""

import netCDF4
import numpy as np
from numba import jit
from pygeogrids.grids import BasicGrid


@jit(nopython=True, cache=True)
def compute_fib_grid(n):
    """
    Computation of Fibonacci lattice on a sphere.

    Parameters
    ----------
    n : int
        Number of grid points in the Fibonacci lattice.

    Returns
    -------
    points : numpy.ndarray
        Point number from -n to +n.
    gpi : numpy.ndarray
        Grid point index starting at 0.
    lon : numpy.ndarray
        Longitude coordinate.
    lat : numpy.ndarray
        Latitude coordinate.
    """
    points = np.arange(-n, n+1)
    gpi = np.arange(points.size)
    lat = np.empty(points.size)
    lon = np.empty(points.size)
    phi = (1. + np.sqrt(5))/2.

    for i in points:
        lat[i] = np.arcsin((2*i)/(2*n+1)) * 180./np.pi
        lon[i] = np.mod(i, phi) * 360./phi
        if lon[i] < -180:
            lon[i] += 360.
        if lon[i] > 180:
            lon[i] -= 360.

    return points, gpi, lon, lat


def compute_grid_stats(n, k=5, geodatum='WGS84'):
    """
    Compute grid statistics of k nearest neighbors.

    Parameters
    ----------
    n : int
        Number of grid points in the Fibonacci lattice.
    k : int, optional
        Nearest neighbor (default: 5).
    geodatum : str, optional
        Geodatum (default: 'WGS84')
    """
    points, gpi, lon, lat = compute_fib_grid(n)

    grid = BasicGrid(lon, lat, geodatum=geodatum)
    nn, dist = grid.find_k_nearest_gpi(lon, lat, k=5)

    print('Min distance:    {:10.4f}'.format(dist[:, 1:].min()))
    print('Max distance:    {:10.4f}'.format(dist[:, 1:].max()))
    print('Mean distance:   {:10.4f}'.format(dist[:, 1:].mean()))
    print('Median distance: {:10.4f}'.format(np.median(dist[:, 1:])))
    print('Std distance:    {:10.4f}'.format(dist[:, 1:].std()))

    for i in range(1, k):
        print('\n')
        print('------- Neighbor #{:} -------'.format(i))
        print('Min distance:    {:10.4f}'.format(dist[:, i].min()))
        print('Max distance:    {:10.4f}'.format(dist[:, i].max()))
        print('Mean distance:   {:10.4f}'.format(dist[:, i].mean()))
        print('Median distance: {:10.4f}'.format(np.median(dist[:, i])))
        print('Std distance:    {:10.4f}'.format(dist[:, i].std()))

    print('\n')
    lat_bands = range(-90, 90, 5)
    for band in lat_bands:
        subgrid = grid.subgrid_from_gpis(grid.get_bbox_grid_points(
            latmin=band, latmax=band+5))
        nn, dist_subgrid = grid.find_k_nearest_gpi(
            subgrid.arrlon, subgrid.arrlat, k=k)
        print('Band [{} -- {}] deg: {:10.4f}'.format(
            band, band+5, dist_subgrid[:, 1:].mean()))


def write_grid(filename, n, nc_fmt='NETCDF4_CLASSIC', nc_zlib=True,
               nc_complevel=2):
    """
    Write grid file for Fibonacci lattice.

    Parameters
    ----------
    filename : str
        Grid filename.
    n : int
        Number of grid points in the Fibonacci lattice.
    nc_fmt : str, optional
        NetCDF4 file format (default: 'NETCDF4_CLASSIC').
    nc_zlib : bool, optional
        If the optional keyword zlib is True, the data will be compressed in
        the netCDF file using gzip compression (default: True).
    nc_complevel : int, optional
        The optional keyword complevel is an integer between 1 and 9
        describing the level of compression desired (default 2).
    """
    points, gpi, lon, lat = compute_fib_grid(n)

    with netCDF4.Dataset(filename, 'w', format=nc_fmt) as fp:

        fp.createDimension('locations', gpi.size)

        gpi_var = fp.createVariable('gpi', np.int32, ('locations',),
                                    zlib=nc_zlib, complevel=nc_complevel)
        gpi_var[:] = gpi

        gpi_attr = {'name': 'gpi',
                    'long_name': 'grid point index',
                    'coordinates': 'lat lon',
                    'valid_range': (0, gpi.size-1),
                    'missing_value': np.iinfo(np.int32).max}

        gpi_var.setncatts(gpi_attr)

        lon_var = fp.createVariable('lon', np.float32, ('locations',),
                                    zlib=nc_zlib, complevel=nc_complevel)
        lon_var[:] = lon

        lon_attr = {'standard_name': 'longitude',
                    'long_name': 'location longitude',
                    'units': 'degrees_east',
                    'valid_range': (-180.0, 180.0)}
        lon_var.setncatts(lon_attr)

        lat_var = fp.createVariable('lat', np.float32, ('locations',),
                                    zlib=nc_zlib, complevel=nc_complevel)
        lat_var[:] = lat

        lat_attr = {'standard_name': 'latitude',
                    'long_name': 'location latitude',
                    'units': 'degrees_north',
                    'valid_range': (-90.0, 90.0)}
        lat_var.setncatts(lat_attr)

        global_ncatts = {'creator': 'fibgrid'}
        fp.setncatts(global_ncatts)


def read_grid(filename, variables=['gpi', 'lon', 'lat']):
    """
    Read grid file stored as NetCDF.

    Parameters
    ----------
    filename : str
        Grid filename.
    variables : list of str, optional
        Variables to be read (default: ['gpi', 'lon', 'lat'])

    Returns
    -------
    data : dict
        Grid file information.
    """
    data = {}
    with netCDF4.Dataset(filename) as fp:
        for var in variables:
            data[var] = fp.variables[var][:]

    return data
