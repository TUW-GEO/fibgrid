# Copyright (c) 2026, TU Wien
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

"""Construct Fibonacci grid."""

import numpy as np
from numba import jit
from pyproj import CRS, Transformer


@jit(nopython=True, cache=True)
def compute_fib_grid(n: int) -> tuple:
    """Compute Fibonacci lattice on a sphere.

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
    points = np.arange(-n, n + 1)
    gpi = np.arange(points.size)
    lat = np.empty(points.size, dtype=np.float64)
    lon = np.empty(points.size, dtype=np.float64)
    phi = (1.0 + np.sqrt(5)) / 2.0

    for i in points:
        lat[i] = np.arcsin((2 * i) / (2 * n + 1)) * 180.0 / np.pi
        lon[i] = np.mod(i, phi) * 360.0 / phi
        if lon[i] < -180:
            lon[i] += 360.0
        if lon[i] > 180:
            lon[i] -= 360.0

    return points, gpi, lon, lat


def compute_fib_grid_wgs84(n: int) -> tuple:
    """Compute Fibonacci lattice on a sphere and transform coordinates (WGS84).

    Parameters
    ----------
    n : int
        Number of grid points in the Fibonacci lattice.

    """
    crs_wgs84 = CRS.from_epsg(4326)
    crs_sphere = CRS.from_proj4("+proj=lonlat +ellps=sphere +R=6370997 +towgs84=0,0,0")

    points, gpi, sphere_lon, sphere_lat = compute_fib_grid(n)
    transformer = Transformer.from_crs(crs_sphere, crs_wgs84)

    wgs84_lon = np.zeros(sphere_lon.size, dtype=np.float64)
    wgs84_lat = np.zeros(sphere_lat.size, dtype=np.float64)

    i = 0
    for lon, lat in zip(sphere_lon, sphere_lat):
        wgs84_lat[i], wgs84_lon[i] = transformer.transform(lon, lat)
        i = i + 1

    return points, gpi, wgs84_lon, wgs84_lat
