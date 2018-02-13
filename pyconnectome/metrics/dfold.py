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
import nibabel.gifti.giftiio as gio

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

    Returnsfrom pyfreesurfer.utils.surftools import apply_affine_on_mesh
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


def convert_pits(pits_file, mesh_file, t1_file):
    """ Extract pits coordinates from white matter mesh in physical
    morphological space and put them in NIFTI voxel space.

    Parameters
    ----------
    pits_file: str
        the pits '.gii' file.
    mesh_file: str
        the path to white matter '.gii' mesh file.
    t1_file: str
        the t1 NIFTI file.

    Returns
    -------
    mesh_pits: ndarray (shape (N,3))
        all pits vertices in mesh.
        Vertices are in NIFTI voxel space.
    pits_indexes : ndarray (shape (N,1))
        all the pits indexes in the pits_file
    """
    # Load pits and mesh file
    pits_gii = gio.read(pits_file)
    mesh_gii = gio.read(mesh_file)

    # Get mesh vertices and pits' mask array and check data adequacy
    pits_vertices = pits_gii.darrays[0].data
    mesh_vertices = mesh_gii.darrays[0].data

    if mesh_vertices.shape[0] != pits_vertices.shape[0]:
        raise ValueError("Surface pits file and white matter surface file\
                          should have the same number of vertices")

    pits_indexes = numpy.argwhere(pits_vertices == 1).flatten()

    # Get the mesh vertices which are pits
    mesh_pits = mesh_vertices[pits_indexes]

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

    # Set pits vertices in voxel Nifti space
    mesh_pits = apply_affine_on_mesh(mesh_pits, inv_morphaffine)

    return mesh_pits, pits_indexes


def sphere_integration(t1_file, folds, scalars, pits=None, pits_ind=None,
                       seg_file=None, radius=2, wm_label=200, gm_label=100):
    """ Compute some measures attached to vertices using a sphere integration
    strategy.

    Parameters
    ----------
    t1_file: str
        the reference anatomical file.
    folds: dict with TriSurface
        all the loaded folds. The fold names are stored in the metadata.
        Vertices are in NIFTI voxel space.
    pits: ndarray (shape (N,3))
        all pits vertices in white mesh.
        Vertices are in NIFTI voxel space.
        If pits is different from None, scalars will be computed on pits only
        and not on the folds.
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

    # Load all scalars' image files and check they are all in the same space
    scalarims = {}
    scalaraffine = None
    for path in scalars:
        name = os.path.basename(path).split(".")[0]
        scalarims[name] = nibabel.load(path)
        if scalaraffine is None:
            scalaraffine = scalarims[name].affine
            scalarshape = scalarims[name].get_data().shape
        elif not numpy.allclose(scalarims[name].affine, scalaraffine):
            raise ValueError("The scalar images must be in the same space.")

    # Compute the voxel anatomical to voxel scalar coordinates transformation.
    trf = numpy.dot(numpy.linalg.inv(scalaraffine), t1affine)

    # Load segmentation file and extract wm/gm coordinates
    if seg_file is not None:
        segim = nibabel.load(seg_file)
        if not numpy.allclose(segim.affine, t1affine, 3):
            raise ValueError("The white/grey matter image must be in the same "
                             "space than the anatomical image.")
        condition = (segim.get_data() == wm_label)
        points_in_wm = numpy.argwhere(condition)

        # Switch to voxel scalar coordinates
        points_in_wm = apply_affine_on_points(points_in_wm, trf)
        points_in_wm = points_in_wm.astype(int)

        # Put in neurological convention
        points_in_wm[:, 0] = scalarshape[0] - points_in_wm[:, 0]

        if points_in_wm.shape[0] == 0:
            points_in_wm = None

        condition = (segim.get_data() == gm_label)
        points_in_gm = numpy.argwhere(condition)

        # Switch to voxel scalar coordinates
        points_in_gm = apply_affine_on_points(points_in_gm, trf)
        points_in_gm = points_in_gm.astype(int)

        # Put in neurological convention
        points_in_gm[:, 0] = scalarshape[0] - points_in_gm[:, 0]

        if points_in_gm.shape[0] == 0:
            points_in_gm = None

    # Go through each fold/pits
    measures = {}
    if pits is None:
        for labelindex, surf in folds.items():

            # Set the vertices to the scalar space
            vertices = apply_affine_on_mesh(surf.vertices, trf)

            # For each vertex compute the sphere intersection with all the
            # scalar maps
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
                        wm_points = points_intersection(int_points,
                                                        points_in_wm)
                        gm_points = points_intersection(int_points,
                                                        points_in_gm)
                        if wm_points is not None and len(wm_points) != 0:
                            wm_points_x = tuple(wm_points[:, 0])
                            wm_points_y = tuple(wm_points[:, 1])
                            wm_points_z = tuple(wm_points[:, 2])
                            wm_mean = float(numpy.mean(image.get_data()
                                            [wm_points_x, wm_points_y,
                                             wm_points_z]))
                            wm_median = float(numpy.median(image.get_data()
                                              [wm_points_x, wm_points_y,
                                               wm_points_z]))
                        if gm_points is not None and len(gm_points) != 0:
                            gm_points_x = tuple(gm_points[:, 0])
                            gm_points_y = tuple(gm_points[:, 1])
                            gm_points_z = tuple(gm_points[:, 2])
                            gm_mean = float(numpy.mean(image.get_data()
                                            [gm_points_x, gm_points_y,
                                             gm_points_z]))
                            gm_median = float(numpy.median(image.get_data()
                                              [gm_points_x, gm_points_y,
                                               gm_points_z]))
                        int_points_x = tuple(int_points[:, 0])
                        int_points_y = tuple(int_points[:, 1])
                        int_points_z = tuple(int_points[:, 2])
                        measures[labelindex][key][name] = {
                            "global_mean": float(numpy.mean(image.get_data()
                                                 [int_points_x, int_points_y,
                                                  int_points_z])),
                            "wm_mean": wm_mean,
                            "gm_mean": gm_mean,
                            "global_median": float(numpy.median(
                                                   image.get_data()
                                                   [int_points_x,
                                                    int_points_y,
                                                    int_points_z])),
                            "wm_median": wm_median,
                            "gm_median": gm_median
                        }
                    bar.update(cnt)
    else:
        # Set the pits to the scalar space
        pits = apply_affine_on_mesh(pits, trf)
        # For each vertex compute the sphere intersection with all the scalar
        # maps
        with progressbar.ProgressBar(max_value=len(pits),
                                     redirect_stdout=True) as bar:
            for cnt, vertex in enumerate(pits):
                coord = repr(vertex.tolist())
                pit_index = pits_ind[cnt]
                measures[pit_index] = {}
                for name, image in scalarims.items():
                    # Initialize mean and median values
                    wm_mean, gm_mean = None, None
                    wm_median, gm_median = None, None
                    if name in measures[pit_index]:
                        raise ValueError("All the scalar map must have "
                                         "different names.")
                    int_points = inside_sphere_points(
                        center=vertex, radius=radius, shape=image.shape)
                    wm_points = points_intersection(int_points, points_in_wm)
                    gm_points = points_intersection(int_points, points_in_gm)
                    if wm_points is not None and len(wm_points) != 0:
                        wm_points_x = tuple(wm_points[:, 0])
                        wm_points_y = tuple(wm_points[:, 1])
                        wm_points_z = tuple(wm_points[:, 2])
                        wm_mean = float(numpy.mean(image.get_data()
                                        [wm_points_x, wm_points_y,
                                         wm_points_z]))
                        wm_median = float(numpy.median(image.get_data()
                                          [wm_points_x, wm_points_y,
                                           wm_points_z]))
                    if gm_points is not None and len(gm_points) != 0:
                        gm_points_x = tuple(gm_points[:, 0])
                        gm_points_y = tuple(gm_points[:, 1])
                        gm_points_z = tuple(gm_points[:, 2])
                        gm_mean = float(numpy.mean(image.get_data()
                                        [gm_points_x, gm_points_y,
                                         gm_points_z]))
                        gm_median = float(numpy.median(image.get_data()
                                          [gm_points_x, gm_points_y,
                                           gm_points_z]))
                    int_points_x = tuple(int_points[:, 0])
                    int_points_y = tuple(int_points[:, 1])
                    int_points_z = tuple(int_points[:, 2])
                    measures[pit_index][name] = {
                        "coord_scalar_space": coord,
                        "global_mean": float(numpy.mean(image.get_data()
                                             [int_points_x, int_points_y,
                                              int_points_z])),
                        "wm_mean": wm_mean,
                        "gm_mean": gm_mean,
                        "global_median": float(numpy.median(
                                               image.get_data()
                                               [int_points_x,
                                                int_points_y,
                                                int_points_z])),
                        "wm_median": wm_median,
                        "gm_median": gm_median
                    }

                bar.update(cnt)

    return measures


def apply_affine_on_points(points, trf):
    """ Return a transformed list of points

    Parameters
    ----------
    points : array, shape (m,3)
       array of points to be transformed.
    trf : affine matrix, shape (t,4)
       transformation matrix.

    Returns
    -------
    points : array, shape (N,3)
       the array of the transformed points
    """
    M = trf[:3, :3]
    abc = trf[:3, 3]
    for i in range(points.shape[0]):
        points[i, :] = numpy.add(numpy.dot(M, points[i, :]), abc)
    return points


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
