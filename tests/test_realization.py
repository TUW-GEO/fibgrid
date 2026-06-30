# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test pre-computed Fibonacci grid files.
"""

import unittest

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

import fibgrid.realization as realization
from fibgrid.realization import (
    METADATA_FIELDS,
    FibGrid,
    FibLandGrid,
    read_grid_file,
)


def _write_grid_zip(zip_path, size):
    """Write a minimal but valid fibgrid ``.zarr.zip`` artifact.

    Mirrors the layout produced by ``helper.convert_to_zarr`` (one unchunked
    array per variable) so it can drive the download/extract code path.
    """
    arrays = {
        "lon": np.linspace(-180, 180, size, dtype=np.float32),
        "lat": np.linspace(-90, 90, size, dtype=np.float32),
        "cell": np.arange(size, dtype=np.int16),
        "gpi": np.arange(size, dtype=np.int32),
    }
    for field in METADATA_FIELDS:
        arrays[field] = np.zeros(size, dtype=np.int8)

    with ZipStore(str(zip_path), mode="w") as store:
        root = zarr.open_group(store=store, mode="w")
        for name, data in arrays.items():
            arr = root.create_array(
                name, shape=data.shape, dtype=data.dtype, chunks=data.shape
            )
            arr[:] = data


class TestGrid(unittest.TestCase):
    def setUp(self):
        """
        Define grids.
        """
        self.res = [6.25, 12.5, 25]
        self.n = [6600000, 1650000, 430000]
        self.geodatum = ["sphere", "WGS84"]

    def test_grid_files(self):
        """
        Test read grid files.
        """
        for geodatum in self.geodatum:
            for res, n in zip(self.res, self.n, strict=True):
                fb = FibGrid(res, geodatum=geodatum)
                assert fb.gpis.size == 2 * n + 1


class TestReadGridFile(unittest.TestCase):
    def test_returns_arrays_and_metadata(self):
        """read_grid_file returns the coordinates and land metadata."""
        n = 430000
        lon, lat, cell, gpi, metadata = read_grid_file(n, geodatum="WGS84")
        size = 2 * n + 1
        for arr in (lon, lat, cell, gpi):
            assert arr.shape == (size,)
        assert tuple(metadata.dtype.names) == tuple(METADATA_FIELDS)
        assert metadata.shape == (size,)

    def test_latband_sorting(self):
        """latband sorting reorders the same points from south to north."""
        _, lat_none, _, _, _ = read_grid_file(1650000, "sphere", "none")
        _, lat_sorted, _, _, _ = read_grid_file(1650000, "sphere", "latband")
        assert lat_sorted[0] < lat_sorted[-1]
        # latband is a permutation: same set of latitudes, different order
        np.testing.assert_array_equal(np.sort(lat_none), np.sort(lat_sorted))

    def test_unknown_geodatum(self):
        """An unknown geodatum raises a ValueError."""
        with pytest.raises(ValueError, match="Geodatum unknown"):
            read_grid_file(430000, geodatum="mars")

    def test_unknown_sort_order(self):
        """An unknown sort order raises a ValueError."""
        with pytest.raises(ValueError, match="sort order unknown"):
            read_grid_file(430000, geodatum="WGS84", sort_order="bogus")

    def test_latband_unavailable(self):
        """latband is not available for the 25 km grid and errors clearly."""
        with pytest.raises(ValueError, match="not available"):
            read_grid_file(430000, geodatum="WGS84", sort_order="latband")


class TestDownload:
    """Exercise the download / extract code path in read_grid_file."""

    def test_missing_download_url(self, tmp_path, monkeypatch):
        """A grid with no registered URL raises a clear ValueError."""
        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(realization, "CACHE_DIR", cache)
        monkeypatch.setattr(realization, "DATA_URL_ZARR", {})
        with pytest.raises(ValueError, match="No download URL"):
            read_grid_file(2, geodatum="WGS84")

    def test_download_extract_and_reuse(self, tmp_path, monkeypatch):
        """First read downloads + extracts; the second reuses the store."""
        size = 5
        src = tmp_path / "artifact.zarr.zip"
        _write_grid_zip(src, size)

        cache = tmp_path / "cache"
        cache.mkdir()
        monkeypatch.setattr(realization, "CACHE_DIR", cache)
        monkeypatch.setattr(
            realization, "DATA_URL_ZARR", {"n2_wgs84": src.as_uri()}
        )

        with pytest.warns(UserWarning, match="about to download"):
            lon, lat, cell, gpi, metadata = read_grid_file(2, geodatum="WGS84")

        for arr in (lon, lat, cell, gpi):
            assert arr.shape == (size,)
        assert tuple(metadata.dtype.names) == tuple(METADATA_FIELDS)
        # the archive was extracted and the downloaded zip removed
        assert (cache / "fibgrid_wgs84_n2.zarr").is_dir()
        assert not (cache / "fibgrid_wgs84_n2.zarr.zip").exists()

        # second call must not download again: drop the URL so any attempt fails
        monkeypatch.setattr(realization, "DATA_URL_ZARR", {})
        lon2, *_ = read_grid_file(2, geodatum="WGS84")
        np.testing.assert_array_equal(lon, lon2)


class TestResolutionMapping:
    """Cover the res->n mapping and the unknown-resolution errors."""

    def test_fibgrid_unknown_resolution(self):
        with pytest.raises(ValueError, match="Resolution unknown"):
            FibGrid(99)

    def test_fibland_unknown_resolution(self):
        with pytest.raises(ValueError, match="Resolution unknown"):
            FibLandGrid(99)

    @pytest.mark.parametrize(
        "res,n", [(6.25, 6600000), (12.5, 1650000), (25, 430000)]
    )
    def test_fibland_resolution_mapping(self, res, n):
        land = FibLandGrid(res, geodatum="WGS84")
        assert 0 < land.activegpis.size < 2 * n + 1


class TestFibLandGrid(unittest.TestCase):
    def test_land_subset(self):
        """FibLandGrid keeps only the land points of the full grid."""
        n = 430000
        land = FibLandGrid(25, geodatum="WGS84")
        _, _, _, _, metadata = read_grid_file(n, geodatum="WGS84")
        expected = int(np.count_nonzero(metadata["land_flag"]))
        assert land.activegpis.size == expected
        assert 0 < expected < 2 * n + 1


if __name__ == "__main__":
    unittest.main()
