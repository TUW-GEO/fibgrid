# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""Download and read pre-computed Fibonacci grid files."""

from pathlib import Path
from platformdirs import user_cache_dir
import urllib.request
import warnings
import zipfile


import numpy as np
import zarr
from pygeogrids.grids import CellGrid

from fibgrid import __version__

METADATA_FIELDS = [
    "land_frac_fw",
    "land_frac_hw",
    "land_mask_hw",
    "land_mask_fw",
    "land_flag",
]

DATA_URL_ZARR = {
    "n430000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_sphere_n430000.zarr.zip",
    "n430000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_wgs84_n430000.zarr.zip",
    "n1650000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_sphere_n1650000.zarr.zip",
    "n1650000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_wgs84_n1650000.zarr.zip",
    "n6600000_sphere": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_sphere_n6600000.zarr.zip",
    "n6600000_wgs84": "https://github.com/TUW-GEO/fibgrid/releases/download/v0.0.9/fibgrid_wgs84_n6600000.zarr.zip",
}

CACHE_DIR = Path(user_cache_dir("fibgrid")) / __version__


def read_grid_file(n: int, geodatum: str = "WGS84", sort_order: str = "none") -> tuple:
    """Read a pre-computed grid from a hosted Zarr artifact.

    The grid is distributed as a single ``.zarr.zip`` file. On first use the
    archive is downloaded and extracted into a Zarr directory store inside the
    cache directory; subsequent reads use that on-disk store directly.

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
    metadata : numpy.recarray
        Metadata information of the grid.
    """
    if geodatum not in ["sphere", "WGS84"]:
        raise ValueError(f"Geodatum unknown: {geodatum}")

    if sort_order not in ["none", "latband"]:
        raise ValueError(f"Grid point sort order unknown: {sort_order}")

    zarr_dir = CACHE_DIR / f"fibgrid_{geodatum.lower()}_n{n}.zarr"

    if not zarr_dir.exists():
        zip_path = CACHE_DIR / f"fibgrid_{geodatum.lower()}_n{n}.zarr.zip"
        if not zip_path.exists():
            url = DATA_URL_ZARR.get(f"n{n}_{geodatum.lower()}", "")
            if not url:
                raise ValueError(f"No download URL for {zip_path.name}")
            warnings.warn(
                f"You are about to download the fibonacci grid file: {zip_path.name}",
                UserWarning,
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, zip_path)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zarr_dir)
        zip_path.unlink()

    grid = zarr.open_group(str(zarr_dir), mode="r")
    lon = np.asarray(grid["lon"][:])
    lat = np.asarray(grid["lat"][:])
    cell = np.asarray(grid["cell"][:])
    gpi = np.asarray(grid["gpi"][:])
    metadata_list = [np.asarray(grid[f][:]) for f in METADATA_FIELDS]

    if sort_order != "none":
        idx = np.asarray(grid[f"{sort_order}_sorting"][:])
        lon = lon[idx]
        lat = lat[idx]
        cell = cell[idx]
        gpi = gpi[idx]
        metadata_list = [m[idx] for m in metadata_list]

    metadata = np.rec.fromarrays(metadata_list, names=METADATA_FIELDS)

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
