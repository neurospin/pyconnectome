##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System imports
from __future__ import division
import os
import sys
import argparse
import shutil
import datetime
import glob
import time

# Third party imports
import progressbar
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import rotate
from scipy.ndimage import label

# Global parameters
ITERATION = 1
FVAL = 0


def central_moment_order_pqr(arr, p, q, r):
    """ Compute the input array central moment of order p, q, and r.

    Parameters
    ----------
    arr: array (X, Y, Z)
        an image array from which we want to extract the interhemispheric
        plane.
    p, q, r: int
        momentum orders

    Returns
    -------
    m_pqr: float
        the requested moment.
    xc, yc, zc: float
        the center of mass coordiantes.
    """
    (sx, sy, sz) = arr.shape
    cumulate_intensity = arr.sum()

    # Center of mass of coordinates (xc, yc, zc)
    x, y, z = np.where(arr >= arr.min())
    arr_flat = arr.flatten()
    xc = np.sum(arr_flat * x) / cumulate_intensity
    yc = np.sum(arr_flat * y) / cumulate_intensity
    zc = np.sum(arr_flat * z) / cumulate_intensity

    # Central moment p+q+rth-order of arr
    m_pqr = arr_flat * (x - xc)**p
    m_pqr = m_pqr * (y - yc)**q
    m_pqr = np.sum(m_pqr * (z - zc)**r)

    return m_pqr, xc, yc, zc


def head_interhemi_plane_init(arr):
    """ Find a good initilization for the interhemispheric plane.

    The interhemispheric plane is defined as:
        ax + by + cz + d = 0

    Reference: Tuzikov, Alexander V., Olivier Colliot, and Isabelle Bloch.
    Evaluation of the symmetry plane in 3D MR brain images. Pattern Recognition
    Letters 24.14 (2003): 2219-2233.

    Parameters
    ----------
    arr: array (X, Y, Z)
        an image array from which we want to extract the interhemispheric
        plane.

    Returns
    -------
    x0: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    """
    # Computing momentum
    m200, xc, yc, zc = central_moment_order_pqr(arr, 2, 0, 0)
    m110, xc, yc, zc = central_moment_order_pqr(arr, 1, 1, 0)
    m101, xc, yc, zc = central_moment_order_pqr(arr, 1, 0, 1)
    m020, xc, yc, zc = central_moment_order_pqr(arr, 0, 2, 0)
    m011, xc, yc, zc = central_moment_order_pqr(arr, 0, 1, 1)
    m002, xc, yc, zc = central_moment_order_pqr(arr, 0, 0, 2)

    # Computing plane normal
    cov_matrix = np.array([[m200, m110, m101],
                           [m110, m020, m011],
                           [m101, m011, m002]])
    eigval, eigvec = np.linalg.eigh(cov_matrix)
    a_mass = [xc, yc, zc]
    costs = []
    x_init = []
    with progressbar.ProgressBar(max_value=3) as bar:
        for index in range(3):
            bar.update(index)
            x = eigvec[index].tolist()
            x.append(np.dot(x, a_mass))
            x_init.append(x)
            costs.append(head_interhemi_plane_cost(x, arr))
    min_index = np.argmin(costs)
    x0 = x_init[min_index]

    return x0


def image_symmetry(x, arr):
    """ This function compute the symmetrized image array from an
    interhemispheric plane is defined as:
        ax + by + cz + d = 0

    Reference: Tuzikov, Alexander V., Olivier Colliot, and Isabelle Bloch.
    Evaluation of the symmetry plane in 3D MR brain images. Pattern Recognition
    Letters 24.14 (2003): 2219-2233.

    Parameters
    ----------
    x: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    arr: array (X, Y, Z)
        an image array from which we want to extract the interhemispheric
        plane.

    Returns
    -------
    arr_sym: array (X, Y, Z)
        the symmetrized input image array.
    """
    # Detect all available voxel indices: points Ai
    xa, ya, za = np.where(arr >= arr.min())
    a_i = [xa, ya, za]

    # Compute the projections AiAproj vec normal = 0
    a, b, c, d = x
    z_proj = a * a * za - a * c * xa + c * d - b * c * ya + b * b * za
    y_proj = (c * ya - b * za + b * z_proj) / c
    x_proj = (d - b * y_proj - c * z_proj) / a

    # Compute the symetries: 2 * AiProj = AiAsym
    x_sym = (2 * x_proj - xa).astype(int)
    y_sym = (2 * y_proj - ya).astype(int)
    z_sym = (2 * z_proj - za).astype(int)
    a_sym = [x_sym, y_sym, z_sym]

    # Remove symetric points out of the FOV
    arr_sym = np.zeros(arr.shape, dtype=arr.dtype)
    for axis in range(arr.ndim):
        valid_condition = (a_sym[axis] < arr.shape[axis]) & (a_sym[axis] > 0)
        for axis in range(arr.ndim):
            a_sym[axis] = a_sym[axis][valid_condition]
            a_i[axis] = a_i[axis][valid_condition]
    arr_sym[tuple(a_i)] = arr[tuple(a_sym)]

    return arr_sym


def head_interhemi_plane_cost(x, arr):
    """ Cost function to determine the interhemispheric plane.

    The interhemispheric plane is defined as:
        ax + by + cz + d = 0

    Reference: Tuzikov, Alexander V., Olivier Colliot, and Isabelle Bloch.
    Evaluation of the symmetry plane in 3D MR brain images. Pattern Recognition
    Letters 24.14 (2003): 2219-2233.

    Parameters
    ----------
    x: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    arr: array (X, Y, Z)
        an image array from which we want to extract the interhemispheric
        plane.

    Returns
    -------
    cost: float
        the cost function value.
    """
    # Compute the image symmetic at the current stage of the optimization
    arr_sym = image_symmetry(x, arr)

    # Compute the nse metric (normalized square error)
    cost = np.mean((arr - arr_sym)**2)
    global FVAL
    FVAL = cost

    return cost


def plot_slice(x, arr, zcut, output_file=None):
    """ Display the interhemispheric plane at a specific slice.

    Parameters
    ----------
    x: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    arr: array (X, Y, Z)
        an image array from which we want to extract the interhemispheric
        plane.
    zcut: int
        the desired slice. The last dimension must match with the slice axis.
    output_file: str, optional
        the name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg.

    Returns
    -------
    fig:
        the matplotlib figure.
    """
    fig = plt.figure()
    a, b, c, d = x
    plt.imshow(rotate(arr[..., zcut], 90), cmap="gray", interpolation="none")
    y_plane = np.asarray(range(arr.shape[1]))
    x_plane = (- b * y_plane - c * zcut + d) / a
    plt.plot(x_plane, y_plane, "r")
    plt.axis("off")
    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    return fig


def callback(x):
    """ Simple optimized callback to display optimization status at each
    iteration.
    """
    global ITERATION, FVAL
    print("{0:4d} {1: 3.6f} {2: 3.6f} {3: 3.6f} {4: 3.6f} {5: 3.6f}".format(
        ITERATION, x[0], x[1], x[2], x[3], FVAL))
    ITERATION += 1


def head_interhemi_plane(input_file, verbose=0, output_dir=None):
    """ Compute the interhemispheric plane.

    The interhemispheric plane is defined as:
        ax + by + cz + d = 0

    Parameters
    ----------
    input_file: str
        an input image file.
    verbose: int, optional
        control the verbosity level.
    output_dir: str, optional
        an existing directory where some QC files will be saved.

    Returns
    -------
    x: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    snaps: list of str
        the generated QC files.
    """
    # Initialisation of the plane using the ellipsoid of inertia
    if verbose > 0:
        print("---")
        print("Starting plane initialization")
    snaps = []
    tic = time.clock()
    im = nib.load(input_file)
    arr = im.get_data().astype(float)
    zcut = arr.shape[-1] // 2
    x0 = head_interhemi_plane_init(arr)
    toc = time.clock()
    if verbose > 0:
        print("x0 = ", x0)
        print("Ellapsed time:", toc - tic)
    if output_dir is not None:
        snaps.append(os.path.join(output_dir, "x0.png"))
        plot_slice(x0, arr, zcut=zcut, output_file=snaps[-1])
    elif verbose > 1:
        plot_slice(x0, arr, zcut=zcut)

    # Optimisation using downhill simplex method (Nelder-Mead)
    if verbose > 0:
        print("---")
        print("Starting optimization")
    options = {
        "xtol": 1e-2,
        "disp": True,
        "maxiter": 500}
    print("{0:4s} {1:9s} {2:9s} {3:9s} {4:9s} {5:9s}".format(
        "Iter", "a", "b", "c", "d", "cost"))
    res = minimize(head_interhemi_plane_cost, x0, method="nelder-mead",
                   args=(arr, ), callback=callback, options=options)
    toc = time.clock()
    if verbose > 0:
        print("x = ", res.x)
        print("Ellapsed time:", toc - tic)
    if output_dir is not None:
        snaps.append(os.path.join(output_dir, "x.png"))
        plot_slice(res.x, arr, zcut=zcut, output_file=snaps[-1])
    elif verbose > 1:
        plot_slice(res.x, arr, zcut=zcut)
        plt.show()

    return res.x, snaps


def head_interhemi_distances(input_file, mask_file, x, verbose=0):
    """ Compute the minimum distance between each connected components of the
    mask and a plane.

    The plane is defined as:
        ax + by + cz + d = 0

    Parameters
    ----------
    input_file: str
        an input image file.
    mask_file: str
        a mask file containing the ROI (one ROI = one connected component).
    x: 4-uplet
        vector containing the plane parameters a, b, c, and d.
    verbose: int, optional
        control the verbosity level.

    Returns
    -------
    dists: dict
        the blobs distances.
    """
    # Load the data
    im = nib.load(input_file)
    mask = nib.load(mask_file)
    spacing_square = np.asarray(im.header.get_zooms())**2
    im_arr = im.get_data().astype(float)
    mask_arr = mask.get_data().astype(int)

    # Connected components to extract blobs from mask
    blobs, nb_components = label(mask_arr > 0)

    # Compute distances
    dists = {}
    for labl in range(1, nb_components + 1):

        # Detect all available voxel in blob
        xa, ya, za = np.where(blobs == labl)

        # Compute the projections
        a, b, c, d = x
        z_proj = a * a * za - a * c * xa + c * d - b * c * ya + b * b * za
        y_proj = (c * ya - b * za + b * z_proj) / c
        x_proj = (d - b * y_proj - c * z_proj) / a

        # Compute distances
        dist = np.sqrt(spacing_square[0] * (xa - x_proj)**2 +
                       spacing_square[1] * (ya - y_proj)**2 +
                       spacing_square[2] * (za - z_proj)**2)
        dists[labl] = {
            "centroid_vox": [np.mean(xa), np.mean(ya), np.mean(za)],
            "min_dist_mm": np.min(dist)}

    return dists
