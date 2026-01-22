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
"""Download and read pre-computed Fibonacci grid files."""

from pathlib import Path
from platformdirs import user_cache_dir
import urllib.request
import warnings


import netCDF4
import numpy as np
from pygeogrids.grids import CellGrid

from fibgrid import __version__

DATA_URL = {
    "n430000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_sphere_n430000.nc",
    "n430000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_wgs84_n430000.nc",
    "n1650000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_sphere_n1650000.nc",
    "n1650000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_wgs84_n1650000.nc",
    "n6600000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_sphere_n6600000.nc",
    "n6600000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.8/fibgrid_wgs84_n6600000.nc",
}

CACHE_DIR = Path(user_cache_dir("fibgrid")) / __version__


def read_grid_file(n: int, geodatum: str = "WGS84", sort_order: str = "none") -> tuple:
    """Read pre-computed grid files.

    Parameters
    ----------
    n : int
        Number of grid in the Fibonacci lattice used to identify
        a pre-computed grid.
    geodatum : str, optional
        Definition of geodatum.
    sort_order : str, optional
        Choose sort order of gridpoints:
        "none": original order
        "latband": sorting in latitude bands starting from 90S to 90N.

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
    filename = CACHE_DIR / f"fibgrid_{geodatum.lower()}_n{n}.nc"

    if not filename.exists():
        warnings.warn(
            "You are about to download the fibonacci grid file: {filename.name}",
            UserWarning,
        )
        filename.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL[f"n{n}_{geodatum.lower()}"], filename)

    if geodatum not in ["sphere", "WGS84"]:
        raise ValueError(f"Geodatum unknown: {geodatum}")

    if sort_order not in ["none", "latband"]:
        raise ValueError(f"Grid point sort order unknown: {sort_order}")

    metadata_fields = [
        "land_frac_fw",
        "land_frac_hw",
        "land_mask_hw",
        "land_mask_fw",
        "land_flag",
    ]

    metadata_list = []

    with netCDF4.Dataset(filename) as fp:
        lon = fp.variables["lon"][:].data
        lat = fp.variables["lat"][:].data
        cell = fp.variables["cell"][:].data
        gpi = fp.variables["gpi"][:].data
        if sort_order != "none":
            idx = fp.variables[f"{sort_order}_sorting"][:].data
            lon = lon[idx]
            lat = lat[idx]
            cell = cell[idx]
            gpi = gpi[idx]
            for f in metadata_fields:
                metadata_list.append(fp.variables[f][:].data[idx])
        else:
            for f in metadata_fields:
                metadata_list.append(fp.variables[f][:].data)

    metadata = np.rec.fromarrays(metadata_list, names=metadata_fields)

    return lon, lat, cell, gpi, metadata


class FibGrid(CellGrid):
    """Fibonacci grid."""

    def __init__(
        self, res: float, geodatum: str = "WGS84", sort_order: str = "none"
    ) -> None:
        """Initialize FibGrid.

        Parameters
        ----------
        res : int
            Sampling.
        geodatum : str, optional
            Geodatum (default: 'WGS84')
        sort_order : str, optional
            Sort order of grid points (default: "none")
            Available options
              - "none": original order
              - "latband": points are ordered in latitude bands
        """
        self.res = res
        self.lut = None

        if self.res == 6.25:
            n = 6600000
        elif self.res == 12.5:
            n = 1650000
        elif self.res == 25:
            n = 430000
        else:
            raise ValueError("Resolution unknown")

        lon, lat, cell, gpi, self.metadata = read_grid_file(
            n, geodatum=geodatum, sort_order=sort_order
        )

        super().__init__(lon, lat, cell, gpi, geodatum=geodatum)


class FibLandGrid(CellGrid):
    """Fibonacci grid with active points over land defined by land fraction."""

    def __init__(
        self, res: float, geodatum: str = "WGS84", sort_order: str = "none"
    ) -> None:
        """Initialize FibGrid.

        Parameters
        ----------
        res : int
            Sampling.
        geodatum : str, optional
            Geodatum (default: 'WGS84')
        sort_order : str, optional
            Choose sort order of gridpoints:
            - "none": original order
            - "latband": sorting in latitude bands starting from 90S to 90N.
        """
        self.res = res

        if self.res == 6.25:
            n = 6600000
        elif self.res == 12.5:
            n = 1650000
        elif self.res == 25:
            n = 430000
        else:
            raise ValueError("Resolution unknown")

        lon, lat, cell, gpi, self.metadata = read_grid_file(
            n, geodatum=geodatum, sort_order=sort_order
        )

        subset = np.nonzero(self.metadata["land_flag"])[0]

        super().__init__(lon, lat, cell, gpi, subset=subset, geodatum=geodatum)
