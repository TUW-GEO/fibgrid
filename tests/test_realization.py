# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test pre-computed Fibonacci grid files.
"""

import unittest

import numpy as np
import pytest

from fibgrid.realization import (
    METADATA_FIELDS,
    FibGrid,
    FibLandGrid,
    read_grid_file,
)


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
