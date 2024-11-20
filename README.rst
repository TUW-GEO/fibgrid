=======
fibgrid
=======

.. image:: https://github.com/TUW-GEO/fibgrid/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/TUW-GEO/fibgrid/actions/workflows/ci.yml

.. image:: https://coveralls.io/repos/github/TUW-GEO/fibgrid/badge.svg?branch=master
   :target: https://coveralls.io/github/TUW-GEO/fibgrid?branch=master

.. image:: https://badge.fury.io/py/fibgrid.svg
    :target: http://badge.fury.io/py/fibgrid

.. image:: https://readthedocs.org/projects/fibgrid/badge/?version=latest
   :target: http://fibgrid.readthedocs.org/

Fibonacci grid
==============

The Fibonacci Grid is a method for distributing points uniformly across the surface of a sphere. It is inspired by the Fibonacci sequence and the golden angle to ensure a near-equal area distribution of points, making it ideal for applications that require unbiased global sampling. Points are systematically spaced in latitude and longitude, avoiding clustering at the poles, which is a common issue with traditional latitude-longitude grids.

Constructing the grid involves distributing points along the vertical axis of the sphere (latitude) and rotating them around the sphere’s horizontal axis (longitude) based on the golden angle. This deterministic approach is computationally efficient and scales easily by adjusting the number of points. The resulting grid is particularly suitable for representing global data with uniform coverage, making it useful for tasks like climate modeling, ocean simulation, and satellite data representation.

The Fibonacci Grid is valued for its simplicity and uniformity, offering an effective balance between computational efficiency and spatial accuracy. While it does not achieve exact equal-area partitioning or hierarchical refinement like some specialized grids, its versatility and ease of implementation make it a popular choice for applications requiring discrete global grids.

In the context of transforming Fibonacci grid points to an ellipsoid, the coordinates are first calculated on a sphere and then each point is projected onto the ellipsoidal surface. This two-step process ensures that the initial uniform distribution of points across the sphere is preserved while adapting the grid to the Earth's ellipsoidal shape. Once transformed to the ellipsoid, the grid becomes compatible with real-world geodetic Coordinate Reference Systems (CRS) like WGS84, allowing seamless integration with e.g. satellite data. However, a disadvantage is that the transformation may introduce slight non-uniformities, as the ellipsoid's curvature differs from that of the sphere. These distortions are typically minor and acceptable for most applications but could be a limitation in scenarios requiring exact equal-area distributions.

Grid construction
-----------------

Creating a Fibonacci grid based on a given number of points.

.. code-block:: python

    from fibgrid.construction import compute_fib_grid

    n = 6600000
    points, gpi, lon, lat = compute_fib_grid(n)

Grid realization
----------------

Three different Fibonacci grids can be directly loaded with different sampling distances (6.25 km, 12.5 km and 25 km). Each grid is given in spherical and WGS84 latitude and longitude coordinates.

- approx. 6.25 km (N=6,600,000)
- approx. 12.5 km (N=1,650,000)
- approx. 25 km (N=430,000)

.. code-block:: python

    from fibgrid.realization import FibGrid

    sampling = 12.5
    sphere_fb = FibGrid(sampling, geodatum="sphere")
    wgs84_fb = FibGrid(sampling, geodatum="WGS84")

Citation
========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.14187002.svg
   :target: https://doi.org/10.5281/zenodo.14187002

If you use the software in a publication then please cite it using the Zenodo
DOI. Be aware that this badge links to the latest package version.

Please select your specific version at https://doi.org/10.5281/zenodo.14187002 to
get the DOI of that version. You should normally always use the DOI for the
specific version of your record in citations. This is to ensure that other
researchers can access the exact research artefact you used for reproducibility.

You can find additional information regarding DOI versioning at
http://help.zenodo.org/#versioning
