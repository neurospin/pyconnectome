##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Extract diffusion metrics along the human folds.
"""

# System import
import os
import numpy
import nibabel

# Package import
from pyconnectome.utils.filetools import load_folds

# Third party import
import progressbar
from pyfreesurfer.utils.surftools import TriSurface
from pyfreesurfer.utils.surftools import apply_affine_on_mesh


def convert_folds(folds_file, graph_file, t1_file):
    """ Convert the folds in physical morphological space to NIFTI voxel
    space.

    Parameters
    ----------
    folds_file: str
        the folds '.gii' file.
    graph_file: str (optional, default None)
        the path to a morphologist '.arg' graph file.
    t1_file: str
        the t1 NIFTI file.

    Returns
    -------
    folds: dict with TriSurface
        all the loaded folds. The fold names are stored in the metadata.
        Vertices are in NIFTI voxel space.
    """
    # Load folds
    folds = load_folds(folds_file, graph_file=graph_file)

    # Load image
    t1im = nibabel.load(t1_file)
    affine = t1im.affine
    shape = t1im.shape

    # Generate affine trf in morphologist coordinates
    morphcoord = numpy.eye(4)
    morphcoord[0, 0] = -1
    morphcoord[1, 1] = 1
    morphcoord[2, 2] = 1
    morphcoord[0, 3] = affine[0, 3]
    morphcoord[1, 3] = -affine[1, 3]
    morphcoord[2, 3] = -affine[2, 3]
    morphaffine = numpy.dot(morphcoord, affine)

    # Deal with axis inversion
    inv_morphaffine = numpy.linalg.inv(morphaffine)
    inv_morphaffine[1, 1] = -inv_morphaffine[1, 1]
    inv_morphaffine[2, 2] = -inv_morphaffine[2, 2]
    inv_morphaffine[1, 3] = shape[1]
    inv_morphaffine[2, 3] = shape[2]

    # Set folds vertices in voxel Nifti space
    for labelindex, surf in folds.items():
        surf.vertices = apply_affine_on_mesh(surf.vertices, inv_morphaffine)

    return folds


def sphere_integration(t1_file, folds, scalars, seg_file=None, radius=2,
                       wm_label=200, gm_label=100):
    """ Compute some measures attached to vertices using a sphere integration
    strategy.

    Parameters
    ----------
    t1_file: str
        the reference anatomical file.
    folds: dict with TriSurface
        all the loaded folds. The fold names are stored in the metadata.
        Vertices are in NIFTI voxel space.
    scalars: list of str
        a list of scalar map that will be intersected with the vertices.
    seg_file: str, default None
        the white/grey matter segmentation file.
    radius: float, default 2
        the sphere radius defines in the scalar space and expressed in voxel.
    wm_label: int, default 200
        the label for the white matter in the segmentation mask
    gm_label : int, default 200
        the label for the grey matter in the segmentation mask

    Returns
    -------
    measures: dict
        the different scalar measures computed along the vertices.
    """
    # Check inputs
    if len(scalars) == 0:
        raise ValueError("At least one scalar map is expected.")

    # Load the anatomical image
    t1im = nibabel.load(t1_file)
    t1affine = t1im.affine

    # Load segmentation file and extract wm/gm coordinates
    if seg_file is not None:
        segim = nibabel.load(seg_file)
        if not numpy.allclose(segim.affine, t1affine, 3):
            raise ValueError("The white/grey matter image must be in the same "
                             "space than the anatomical image.")
        condition = (segim.get_data() == wm_label)
        points_in_wm = numpy.argwhere(condition)

        # Check if there is any point in white/grey matter
        if points_in_wm.shape[0] == 0:
            points_in_wm = None
        condition = (segim.get_data() == gm_label)
        points_in_gm = numpy.argwhere(condition)
        if points_in_gm.shape[0] == 0:
            points_in_gm = None

    # Load all scalars' image files and check they are all in the same space
    scalarims = {}
    scalaraffine = None
    for path in scalars:
        name = os.path.basename(path).split(".")[0]
        scalarims[name] = nibabel.load(path)
        if scalaraffine is None:
            scalaraffine = scalarims[name].affine
        elif not numpy.allclose(scalarims[name].affine, scalaraffine):
            raise ValueError("The scalar images must be in the same space.")

    # Compute the voxel anatomical to voxel scalar coordinates transformation.
    trf = numpy.dot(numpy.linalg.inv(scalaraffine), t1affine)

    # Go through each fold
    measures = {}
    for labelindex, surf in folds.items():

        # Set the vertices to the scalar space
        vertices = apply_affine_on_mesh(surf.vertices, trf)

        # For each vertex compute the sphere intersection with all the scalar
        # maps
        measures[labelindex] = {}
        with progressbar.ProgressBar(max_value=len(vertices),
                                     redirect_stdout=True) as bar:
            for cnt, vertex in enumerate(vertices):
                key = repr(vertex.tolist())
                measures[labelindex][key] = {}
                for name, image in scalarims.items():
                    # Initialize mean and median values
                    wm_mean, gm_mean = None, None
                    wm_median, gm_median = None, None
                    if name in measures[labelindex][key]:
                        raise ValueError("All the scalar map must have "
                                         "different names.")
                    int_points = inside_sphere_points(
                        center=vertex, radius=radius, shape=image.shape)
                    wm_points = points_intersection(int_points, points_in_wm)
                    gm_points = points_intersection(int_points, points_in_gm)
                    if wm_points is not None and len(wm_points) != 0:
                        wm_mean = float(numpy.mean(image.get_data()
                                        [wm_points]))
                        wm_median = float(numpy.median(image.get_data()
                                          [wm_points]))
                    if gm_points is not None and len(gm_points) != 0:
                        gm_mean = float(numpy.mean(image.get_data()
                                        [gm_points]))
                        gm_median = float(numpy.median(image.get_data()
                                          [gm_points]))

                    measures[labelindex][key][name] = {
                        "global_mean": float(numpy.mean(image.get_data()
                                             [int_points])),
                        "wm_mean": wm_mean,
                        "gm_mean": gm_mean,
                        "global_median": float(numpy.median(image.get_data()
                                               [int_points])),
                        "wm_median": wm_median,
                        "gm_median": gm_median
                    }
                bar.update(cnt)

    return measures


def points_intersection(points1, points2):
    """ Return the intersection of two arrays of points

    Parameters
    ----------
    points1 : array, shape (n,3)
       first array of points.
    points2 : array, shape (m,3)
       second array of points.

    Returns
    -------
    xyz : array, shape (N,3)
       the array of the intersecting points
    """
    if points1 is None or points2 is None:
        return None

    points1_set = set([tuple(point) for point in points1.tolist()])
    points2_set = set([tuple(point) for point in points2.tolist()])
    intersection = points1_set.intersection(points2_set)
    intersection = numpy.array([list(point) for point in intersection])
    return intersection


def inside_sphere_points(center, radius, shape):
    """ Return all the points within a sphere of a specified
    center and radius.
    Mathematicaly this can be simply described by $|x-c|\le r$ where $x$ a
    point $c$ the center of the sphere and $r$ the radius of the sphere.

    Parameters
    ----------
    center: array, shape (3,)
       center of the sphere.
    radius: float
       radius of the sphere.
    shape: 3-uplet
        the reference shape.

    Returns
    -------
    xyz : array, shape (N,3)
       array representing x,y,z of the N points inside the sphere.
    """
    # Construct the mesh grid from shape
    nx, ny, nz = shape
    gridx, gridy, gridz = numpy.meshgrid(numpy.linspace(0, nx - 1, nx),
                                         numpy.linspace(0, ny - 1, ny),
                                         numpy.linspace(0, nz - 1, nz))
    xyz = numpy.concatenate((
        gridx.flatten().reshape(-1, 1), gridy.flatten().reshape(-1, 1),
        gridz.flatten().reshape(-1, 1)), axis=1)
    xyz = xyz.astype(int)

    # Compute shpere intersection
    return xyz[numpy.sqrt(numpy.sum((xyz - center)**2, axis=1)) <= radius]
