# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 TU Wien
# SPDX-FileContributor: For a full list of authors, see the AUTHORS file.

"""
Test pre-computed Fibonacci grid files.
"""

import unittest

from fibgrid.realization import FibGrid


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
            for res, n in zip(self.res, self.n):
                fb = FibGrid(res, geodatum=geodatum)
                assert fb.gpis.size == 2 * n + 1


if __name__ == "__main__":
    unittest.main()
