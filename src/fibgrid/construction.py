# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

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
