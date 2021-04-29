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
Test Fibonacci grid construction.
"""

import os
import unittest
from tempfile import mkdtemp

import numpy as np

from fibgrid.construction import compute_fib_grid, write_grid, read_grid


class TestConstruction(unittest.TestCase):

    def setUp(self):
        """
        Define grids.
        """
        self.test_path = mkdtemp()
        self.n = [6600000, 1650000, 430000]

    def test_fibgrid(self):
        """
        Test Fibonacci grid construction.
        """
        for n in self.n:
            points, gpi, lon, lat = compute_fib_grid(n)
            np.testing.assert_equal(points, np.arange(-n, n+1))
            np.testing.assert_equal(gpi, np.arange(points.size))

    def test_read_write_fibgrid(self):
        """
        Test read/write Fibonacci grid.
        """
        for n in self.n:
            filename = os.path.join(self.test_path, 'fibgrid_{}.NC'.format(n))
            write_grid(filename, n)
            data = read_grid(filename)
            np.testing.assert_equal(data['gpi'], np.arange(n*2+1))


if __name__ == '__main__':
    unittest.main()
