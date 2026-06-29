# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test Fibonacci grid construction.
"""

import unittest
import numpy as np
from fibgrid.construction import compute_fib_grid


class TestConstruction(unittest.TestCase):
    def setUp(self):
        """
        Define grids.
        """
        self.n = [6600000, 1650000, 430000]

    def test_fibgrid(self):
        """
        Test Fibonacci grid construction.
        """
        for n in self.n:
            points, gpi, lon, lat = compute_fib_grid(n)
            np.testing.assert_equal(points, np.arange(-n, n + 1))
            np.testing.assert_equal(gpi, np.arange(points.size))


if __name__ == "__main__":
    unittest.main()
