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
from __future__ import print_function
from __future__ import division
import os
import json
import numpy
import collections


# Package import
from pyconnectome.utils.filetools import load_folds

# Third party import
import collections
import nibabel
import progressbar
import nibabel.gifti.giftiio as gio
import nibabel.freesurfer as fio
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist
from pyfreesurfer.utils.surftools import TriSurface
from pyfreesurfer.utils.surftools import apply_affine_on_mesh
from pyfreesurfer.utils.regtools import tkregister_translation

# Global parameters
SPHERE_INTEGRATION_METRICS = (
    "global_mean", "global_median", "wm_mean", "wm_median", "gm_mean",
    "gm_median")


def convert_mesh(texture_file, mesh_file, t1_file, outpattern=None,
                 mgz_file=None, freesurfer_conformed=True,
                 freesurfer_native_t1_file=None):
    """ Extract texture coordinates from white matter mesh in physical
    morphological space and put them in NIFTI voxel space.

    Parameters
    ----------
    texture_file: str
        the pits or parcellations '.gii' file. The parcellation can also be
        given as an annation '.annot' file.
    mesh_file: str
        the path to white matter '.gii' mesh file.
    t1_file: str
        the t1 NIFTI file.
    outpattern: str, default None
        if set, save the mesh in native space concatenating this patern with
        'mesh.native.nii.gz'.
    mgz_file: str, default None
        a FreeSurfer '.mgz' file.
    freesurfer_conformed: bool, default True
        if set apply the translation to go from the conformed to native space.
    freesurfer_native_t1_file: str, default None
        if set, consider the input mesh as a FreeSurfer mesh in the conformed
        space, otherwise a morphologist mesh.

    Returns
    -------
    mesh_vertices: ndarray (shape (N,3))
        all mesh vertices in NIFTI voxel space.
    texture_label_indices: list of ndarray (shape (N, 1))
        the texture labels locations that can be applied on the mesh vertices.
    texture_label_names: list of str
        the name of the texture labels.
    """
    # Load pits and mesh file
    if texture_file.endswith(".gii"):
        texture_gii = gio.read(texture_file)
        texture = texture_gii.darrays[0].data
        labels = None
    elif texture_file.endswith(".annot"):
        texture, _, labels = fio.read_annot(texture_file)
    texture = texture.astype(int)
    mesh_gii = gio.read(mesh_file)
    mesh_vertices = mesh_gii.darrays[0].data

    # Get mesh vertices and pits' mask array and check data adequacy
    if mesh_vertices.shape[0] != texture.shape[0]:
        raise ValueError("Texture file and white matter surface file "
                         "should have the same number of vertices.")
    unique_labels = numpy.unique(texture).tolist()
    if 0 in unique_labels:
        unique_labels.remove(0)
    texture_label_indices = []
    texture_label_names = []
    for cnt, label in enumerate(sorted(unique_labels)):
        if labels is not None:
            texture_label_names.append("{0}-{1}".format(label, labels[cnt]))
        else:
            texture_label_names.append("{0}".format(label))
        texture_label_indices.append(numpy.where(texture == label))

    # Load image
    t1im = nibabel.load(t1_file)
    affine = t1im.affine
    shape = t1im.shape

    # Realign the mesh in voxel Nifti space
    # Morphologist mesh
    if freesurfer_native_t1_file is None:
        # > generate affine trf in morphologist coordinates
        morphcoord = numpy.eye(4)
        morphcoord[0, 0] = -1
        morphcoord[1, 1] = 1
        morphcoord[2, 2] = 1
        morphcoord[0, 3] = affine[0, 3]
        morphcoord[1, 3] = -affine[1, 3]
        morphcoord[2, 3] = -affine[2, 3]
        morphaffine = numpy.dot(morphcoord, affine)
        # > deal with axis inversion
        inv_morphaffine = numpy.linalg.inv(morphaffine)
        inv_morphaffine[1, 1] = -inv_morphaffine[1, 1]
        inv_morphaffine[2, 2] = -inv_morphaffine[2, 2]
        inv_morphaffine[1, 3] = shape[1]
        inv_morphaffine[2, 3] = shape[2]
        mesh_vertices = apply_affine_on_mesh(mesh_vertices, inv_morphaffine)
    # FreeSurfer mesh
    else:
        fs_t1_image = nibabel.load(freesurfer_native_t1_file)
        # > FreeSurfer resample the T1 image to 1iso
        freesurfer_to_original_trf = numpy.dot(
            numpy.linalg.inv(t1im.affine), fs_t1_image.affine)
        # > Deal with FreeSurfer inner conformed space
        physical_to_index = numpy.linalg.inv(fs_t1_image.get_affine())
        if freesurfer_conformed:
            translation = tkregister_translation(mgz_file)
        else:
            translation = numpy.eye(4)
        conformed_to_native_trf = numpy.dot(physical_to_index, translation)
        conformed_to_original_trf = numpy.dot(
            freesurfer_to_original_trf, conformed_to_native_trf)
        mesh_vertices = apply_affine_on_mesh(
            mesh_vertices, conformed_to_original_trf)

    # Save the vertices as an image
    if outpattern is not None:
        overlay_file = outpattern + "mesh.native.nii.gz"
        overlay = numpy.zeros(t1im.shape, dtype=numpy.uint)
        indices = numpy.round(mesh_vertices).astype(int).T
        indices[0, numpy.where(indices[0] >= t1im.shape[0])] = 0
        indices[1, numpy.where(indices[1] >= t1im.shape[1])] = 0
        indices[2, numpy.where(indices[2] >= t1im.shape[2])] = 0
        overlay[indices.tolist()] = 1
        overlay_image = nibabel.Nifti1Image(overlay, t1im.affine)
        nibabel.save(overlay_image, overlay_file)

    return mesh_vertices, texture_label_indices, texture_label_names


def intersect_tractogram(tractogram_file, rois, t1_file, nodiff_file, outdir,
                         tol=3, verbose=0):
    """ Get the streamlines near the requested region.

    Parameters
    ----------
    tractogram_file: str
        the path to the tractogram in diffusion coordinates (in mm).
    rois: list of ndarray (N, 3)
        the points in the T1 voxel NIFTI space associated to the region of
        interests.
    t1_file: str
        the path to the t1 NIFTI file.
    nodiff_file: str
        the path or the no diffusion NIFTI file.
    outdir: str
        the destination folder.
    tol: float
        Distance between the end points coordinate in the streamline and the
        center of any voxel in the ROI.
    verbose: int, default 0
        control the verbosity level.

    Returns
    -------
    bundles_file: str
        a list of bundles associated with each ROI.
    """
    # Load antomical references
    if verbose > 0:
        print("[info] Loading anatomical references...")
    t1_image = nibabel.load(t1_file)
    nodiff_image = nibabel.load(nodiff_file)

    # Load the tractogram
    if verbose > 0:
        print("[info] Loading tractogram...")
    tracks = nibabel.streamlines.load(tractogram_file)

    # Downsample & check alignment of the tractogram using voxel coordinates
    if verbose > 0:
        print("[info] Checking tractogram...")
    streamlines = numpy.concatenate(
        [arr[[0, -1]] for arr in tracks.streamlines])
    vox_streamlines = apply_affine_on_mesh(
        streamlines, numpy.linalg.inv(nodiff_image.affine)).astype(int)
    connection_map = numpy.zeros(nodiff_image.shape, dtype=int)
    connection_map[vox_streamlines.T.tolist()] = 1
    connection_map_file = os.path.join(outdir, "connection_map.nii.gz")
    connection_map_image = nibabel.Nifti1Image(
        connection_map, nodiff_image.affine)
    connection_map_image.to_filename(connection_map_file)
    if verbose > 0:
        print("[info] Number of tracks: {0}".format(len(streamlines) / 2))

    # Put the ROI in the diffusion space using voxel coordinates
    if verbose > 0:
        print("[info] Putting ROI in diffusion space...")
    affine = numpy.dot(numpy.linalg.inv(nodiff_image.affine), t1_image.affine)
    vox_diff_rois = []
    roi_map = numpy.zeros(nodiff_image.shape, dtype=int)
    for roi in rois:
        vox_diff_rois.append(apply_affine_on_mesh(roi, affine))
        roi_map[vox_diff_rois[-1].T.tolist()] = 1
    roi_map_file = os.path.join(outdir, "roi_map.nii.gz")
    roi_map_image = nibabel.Nifti1Image(roi_map, nodiff_image.affine)
    roi_map_image.to_filename(roi_map_file)

    # Get the ROI associated fibers
    if verbose > 0:
        print("[info] Filter tractogram using requested ROI...")
    bundles = {}
    with progressbar.ProgressBar(max_value=len(vox_diff_rois),
                                 redirect_stdout=True) as bar:
        for cnt, roi in enumerate(vox_diff_rois):

            # Downsample ROI vertices
            ms = MeanShift(bandwidth=tol / 2, bin_seeding=True)
            ms.fit(roi)
            cluster_centers = ms.cluster_centers_

            # Select fibers
            bundles[cnt] = []
            dist = cdist(vox_streamlines[::2], cluster_centers, "euclidean")
            fibers_indices = numpy.argwhere(
                numpy.min(dist, -1) <= tol).squeeze().tolist()
            if not isinstance(fibers_indices, list):
                fibers_indices = [fibers_indices]
            bundles[cnt].extend(fibers_indices)
            dist = cdist(vox_streamlines[1::2], cluster_centers, "euclidean")
            fibers_indices = numpy.argwhere(
                numpy.min(dist, -1) <= tol).squeeze().tolist()
            if not isinstance(fibers_indices, list):
                fibers_indices = [fibers_indices]
            bundles[cnt].extend(fibers_indices)
            bundles[cnt] = list(set(bundles[cnt]))
            bar.update(cnt)

    # Save the bundles result
    bundles_file = os.path.join(outdir, "bundles.json")
    with open(bundles_file, "wt") as open_file:
        json.dump(bundles, open_file, indent=4)

    return bundles_file


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


def sphere_integration(t1_file, scalars, points, seg_file=None, radius=2,
                       wm_label=200, gm_label=100, average=False,
                       outpattern=None):
    """ Compute some measures attached to vertices using a sphere integration
    strategy.

    Parameters
    ----------
    t1_file: str
        the reference anatomical file.
    scalars: list of str
        a list of scalar map that will be intersected with the vertices.
    points: dict of the form {label : vertices (array(N,3))}.
        all the loaded parcellations, pits or folds' vertices.
        Vertices are in NIFTI voxel space.
    seg_file: str, default None
        the white/grey matter segmentation file.
    radius: float, default 2
        the sphere radius defines in the scalar space and expressed in voxel.
        If None, consider the sphere center only.
    wm_label: int, default 200
        the label for the white matter in the segmentation mask.
    gm_label : int, default 100
        the label for the grey matter in the segmentation mask.
    average: bool, default False
        if set, average the scalar values computed for each element in the
        input points paarameter.
    outpattern: str, default None
        if set, save the points in native space concatenating this patern with
        '<label>.points.scalar.native.nii.gz'.

    Returns
    -------
    measures: dict
        the different scalar measures computed along the vertices.
    scalar_names: list of str
        the different scalar measures names.
    """
    # Check inputs
    if len(scalars) == 0:
        raise ValueError("At least one scalar map is expected.")

    # Load the anatomical image
    t1im = nibabel.load(t1_file)
    t1affine = t1im.affine

    # Load all scalars' image files and check they are all in the same space
    scalarims = collections.OrderedDict()
    scalaraffine = None
    scalarshape = None
    for path in scalars:
        name = os.path.basename(path).split(".")[0]
        scalarims[name] = nibabel.load(path)
        if scalaraffine is None:
            scalaraffine = scalarims[name].affine
            scalarshape = scalarims[name].get_data().shape
        elif not numpy.allclose(scalarims[name].affine, scalaraffine):
            raise ValueError("The scalar images must be in the same space.")
    if scalarshape is None:
        raise ValueError("Need to specify at least one scalar image.")

    # Compute the voxel anatomical to voxel scalar coordinates transformation.
    trf = numpy.dot(numpy.linalg.inv(scalaraffine), t1affine)

    # Load segmentation file and extract wm/gm coordinates
    if seg_file is not None:
        segim = nibabel.load(seg_file)
        if not numpy.allclose(segim.affine, t1affine):
            print(t1affine)
            print(segim.affine)
            raise ValueError("The white/grey segmentation image must be in "
                             "the same space than the anatomical image.")

        # White matter
        condition = (segim.get_data() == wm_label)
        points_in_wm = numpy.argwhere(condition)
        # > switch to voxel scalar coordinates
        points_in_wm = apply_affine_on_mesh(points_in_wm, trf)
        points_in_wm = points_in_wm.astype(int)
        # > put in neurological convention: for morphologist only
        points_in_wm[:, 0] = scalarshape[0] - points_in_wm[:, 0]
        # > check that some points are found
        if points_in_wm.shape[0] == 0:
            points_in_wm = None

        # Gray matter
        condition = (segim.get_data() == gm_label)
        points_in_gm = numpy.argwhere(condition)
        # > switch to voxel scalar coordinates
        points_in_gm = apply_affine_on_mesh(points_in_gm, trf)
        points_in_gm = points_in_gm.astype(int)
        # > put in neurological convention: for morphologist only
        points_in_gm[:, 0] = scalarshape[0] - points_in_gm[:, 0]
        # > check that some points are found
        if points_in_gm.shape[0] == 0:
            points_in_gm = None
    else:
        points_in_wm = None
        points_in_gm = None

    # Go through each list of vertices
    measures = collections.OrderedDict()
    with progressbar.ProgressBar(max_value=len(points),
                                 redirect_stdout=True) as bar:

        # Save the vertices as an image
        if outpattern is not None:
            overlay_file = outpattern + "points.scalar.native.nii.gz"
            overlay = numpy.zeros(scalarshape, dtype=numpy.uint)

        count = 0
        for label, vertices in points.items():

            # Vertices in scalar space
            vertices = apply_affine_on_mesh(vertices, trf)

            # Save the vertices as an image
            if outpattern is not None:
                indices = numpy.round(vertices).astype(int).T
                indices[0, numpy.where(indices[0] >= scalarshape[0])] = 0
                indices[1, numpy.where(indices[1] >= scalarshape[1])] = 0
                indices[2, numpy.where(indices[2] >= scalarshape[2])] = 0
                overlay[indices.tolist()] = 1

            # For each vertex compute the sphere intersection with all the
            # scalar maps
            _measures = collections.OrderedDict()
            for cnt, vertex in enumerate(vertices):
                key = repr(vertex.tolist())
                _measures[key] = collections.OrderedDict()

                # For each scalar map
                for name, image in scalarims.items():
                    wm_mean, gm_mean = None, None
                    wm_median, gm_median = None, None
                    if name in _measures[key]:
                        raise ValueError(
                            "All the scalar map must have different names.")

                    # Compute points indices
                    if radius is None:
                        int_points = vertex.reshape(1, 3).astype(int)
                    else:
                        int_points = inside_sphere_points(
                            center=vertex, radius=radius, shape=image.shape)
                    wm_points = points_intersection(int_points, points_in_wm)
                    gm_points = points_intersection(int_points, points_in_gm)

                    # Intersect white matter
                    if wm_points is not None:
                        wm_points_x = tuple(wm_points[:, 0])
                        wm_points_y = tuple(wm_points[:, 1])
                        wm_points_z = tuple(wm_points[:, 2])
                        wm_mean = float(
                            numpy.mean(
                                image.get_data()
                                [wm_points_x, wm_points_y, wm_points_z]))
                        wm_median = float(
                            numpy.median(
                                image.get_data()
                                [wm_points_x, wm_points_y, wm_points_z]))

                    # Intersect gray matter
                    if gm_points is not None:
                        gm_points_x = tuple(gm_points[:, 0])
                        gm_points_y = tuple(gm_points[:, 1])
                        gm_points_z = tuple(gm_points[:, 2])
                        gm_mean = float(
                            numpy.mean(
                                image.get_data()
                                [gm_points_x, gm_points_y, gm_points_z]))
                        gm_median = float(
                            numpy.median(
                                image.get_data()
                                [gm_points_x, gm_points_y, gm_points_z]))

                    # Whole points intersection
                    int_points_x = tuple(int_points[:, 0])
                    int_points_y = tuple(int_points[:, 1])
                    int_points_z = tuple(int_points[:, 2])
                    global_mean = float(
                        numpy.mean(
                            image.get_data()
                            [int_points_x, int_points_y, int_points_z]))
                    global_median = float(
                        numpy.median(
                            image.get_data()
                            [int_points_x, int_points_y, int_points_z]))
                    _measures[key][name] = collections.OrderedDict()
                    _measures[key][name]["global_mean"] = global_mean
                    _measures[key][name]["global_median"] = global_median
                    _measures[key][name]["wm_mean"] = wm_mean
                    _measures[key][name]["wm_median"] = wm_median
                    _measures[key][name]["gm_mean"] = gm_mean
                    _measures[key][name]["gm_median"] = gm_median

            # Deal with the average option
            if average:
                measures[label] = collections.OrderedDict()
                measures[label]["average"] = collections.OrderedDict()
                avg_measure = measures[label]["average"]
                for name in scalarims:
                    avg_measure[name] = collections.OrderedDict()
                    for metric in SPHERE_INTEGRATION_METRICS:
                        mean_measure = 0
                        for key in _measures:
                            measure = _measures[key][name][metric]
                            if measure is not None:
                                mean_measure += measure / len(_measures)
                        if mean_measure == 0:
                            mean_measure = None

                        avg_measure[name][metric] = mean_measure
            else:
                measures[label] = _measures

            count += 1
            bar.update(count)

        # Save the vertices as an image
        if outpattern is not None:
            overlay_image = nibabel.Nifti1Image(overlay, scalaraffine)
            nibabel.save(overlay_image, overlay_file)

    return measures, scalarims.keys()


def points_intersection(points1, points2):
    """ Return the intersection of two arrays of points

    Parameters
    ----------
    points1: array, shape (n, 3)
       first array of points.
    points2: array, shape (m, 3)
       second array of points.

    Returns
    -------
    xyz: array, shape (N, 3)
        the array of the intersecting points. If no intersection is found
        return None.
    """
    if points1 is None or points2 is None:
        return None
    points1_set = set([tuple(point) for point in points1.tolist()])
    points2_set = set([tuple(point) for point in points2.tolist()])
    intersection = points1_set.intersection(points2_set)
    intersection = numpy.array([list(point) for point in intersection])
    if len(intersection) == 0:
        intersection = None
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
