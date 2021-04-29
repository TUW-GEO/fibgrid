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
Read pre-computed Fibonacci grid files.
"""

import os
import netCDF4
import numpy as np

from pygeogrids.grids import CellGrid


def read_grid_file(n):
    """
    Read pre-computed grid files.

    Parameters
    ----------
    n : int
        Number of grid in the Fibonacci lattice used to identify
        a pre-computed grid.

    Returns
    -------
    lon : numpy.ndarray
        Longitude coordinate.
    lat : numpy.ndarray
        Latitude coordinate.
    cell : numpy.ndarray
        Cell number.
    gpi : numpy.ndarray
        Grid point index starting at 0.
    metadata : dict
        Metadata information of the grid.
    """
    filename = os.path.join(os.path.dirname(__file__),
                            'files', 'fibgrid_n{}.nc'.format(n))

    metadata_fields = ['land_frac_fw', 'land_frac_hw',
                       'land_mask_hw', 'land_mask_fw']
    metadata_list = []
    with netCDF4.Dataset(filename) as fp:
        lon = fp.variables['lon'][:]
        lat = fp.variables['lat'][:]
        cell = fp.variables['cell'][:]
        gpi = fp.variables['gpi'][:]
        for f in metadata_fields:
            metadata_list.append(fp.variables[f][:])

    metadata = np.rec.fromarrays(metadata_list, names=metadata_fields)

    return lon, lat, cell, gpi, metadata


class FibGrid(CellGrid):

    """
    Fibonacci grid.
    """

    def __init__(self, res, geodatum='WGS84'):
        """
        Initialize FibGrid.

        Parameters
        ----------
        res : int
            Sampling.
        geodatum : str, optional
            Geodatum (default: 'WGS84')
        """
        if res == 6.25:
            n = 6600000
        elif res == 12.5:
            n = 1650000
        elif res == 25:
            n = 430000
        else:
            raise ValueError('Resolution unknown')

        lon, lat, cell, gpi, self.metadata = read_grid_file(n)
        super().__init__(lon, lat, cell, gpi, geodatum=geodatum)
