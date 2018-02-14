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
import os
import json
import numpy

# Package import
from pyconnectome.utils.filetools import load_folds

# Third party import
import nibabel
import progressbar
import nibabel.gifti.giftiio as gio
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist
from pyfreesurfer.utils.surftools import TriSurface
from pyfreesurfer.utils.surftools import apply_affine_on_mesh
from pyfreesurfer.utils.regtools import tkregister_translation


def convert_pits(pits_file, mesh_file, t1_file, outdir=None, mgz_file=None,
                 freesurfer_native_t1_file=None):
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
    outdir: str, default None
        if set, save the mesh in native space.
    mgzfile: str, default None
        a FreeSurfer '.mgz' file.
    freesurfer_native_t1_file: str, default None
        if set, consider the input mesh as a FreeSurfer mesh in the conformed
        space, otherwise a morphologist mesh.

    Returns
    -------
    mesh_vertices: ndarray (shape (N,3))
        all mesh vertices in NIFTI voxel space.
    pits_indexes : ndarray (shape (N,1))
        the pits locations that can be applied on the mesh vertices.
    """
    # Load pits and mesh file
    pits_gii = gio.read(pits_file)
    mesh_gii = gio.read(mesh_file)

    # Get mesh vertices and pits' mask array and check data adequacy
    pits_vertices = pits_gii.darrays[0].data
    mesh_vertices = mesh_gii.darrays[0].data
    if mesh_vertices.shape[0] != pits_vertices.shape[0]:
        raise ValueError("Surface pits file and white matter surface file "
                         "should have the same number of vertices.")
    pits_indexes = numpy.argwhere(pits_vertices == 1).flatten()

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
        translation = tkregister_translation(mgz_file)
        conformed_to_native_trf = numpy.dot(physical_to_index, translation)
        conformed_to_original_trf = numpy.dot(
            freesurfer_to_original_trf ,conformed_to_native_trf)
        mesh_vertices = apply_affine_on_mesh(
            mesh_vertices, conformed_to_original_trf)
    # Save the vertices as an image    
    if outdir is not None:
        overlay_file = os.path.join(outdir, "mesh.native.nii.gz")
        overlay = numpy.zeros(t1im.shape, dtype=numpy.uint)
        indices = numpy.round(mesh_vertices).astype(int).T
        indices[0, numpy.where(indices[0] >= t1im.shape[0])] = 0
        indices[1, numpy.where(indices[1] >= t1im.shape[1])] = 0
        indices[2, numpy.where(indices[2] >= t1im.shape[2])] = 0
        overlay[indices.tolist()] = 1
        overlay_image = nibabel.Nifti1Image(overlay, t1im.affine)
        nibabel.save(overlay_image, overlay_file)

    return mesh_vertices, pits_indexes


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


def sphere_integration(t1_file, folds, scalars, wm_file=None, gm_file=None,
                       radius=2):
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
    wm_file: str, default None
        the white matter scalar map.
    gm_file: str, default None
        the gray matter scalar map.
    radius: float, default 2
        the sphere radius defines in the scalar space and expressed in voxel.

    Returns
    -------
    measures: dict
        the different scalar measures computed along the vertices.
    """
    # Check inputs
    if len(scalars) == 0:
        raise ValueError("At least one scalar map is expected.")

    # Load the images
    t1im = nibabel.load(t1_file)
    t1affine = t1im.affine
    wmim, gmim = None, None
    if wm_file is not None:
        wmim = nibabel.load(wm_file)
        if not numpy.allclose(wmim.affine, t1affine):
            raise ValueError("The white matter image must be in the same "
                             "space than the anatomical image.")
    if gm_file is not None:
        gmim = nibabel.load(gm_file)
        if not numpy.allclose(gmim.affine, t1affine):
            raise ValueError("The gray matter image must be in the same "
                             "space than the anatomical image.")
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
                    if name in measures[labelindex][key]:
                        raise ValueError("All the scalar map must have "
                                         "different names.")
                    int_points = inside_sphere_points(
                        center=vertex, radius=radius, shape=image.shape)
                    measures[labelindex][key][name] = {
                        "global_mean": numpy.mean(image.get_data()[int_points])
                    }
                bar.update(cnt)

    return measures


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


def inside_sphere_locations(xyz, center, radius):
    """ Return the locations of a track that intersects with a sphere of a
    specified center and radius.

    Parameters
    ----------
    xyz: array, shape (N,3)
       representing x,y,z of the N points of the track
    center: array, shape (3,)
       center of the sphere
    radius: float
       radius of the sphere

    Returns
    -------
    locations : list (M, )
       the matched locations.
    """
    return (numpy.sqrt(numpy.sum((xyz - center)**2, axis=1)) <= radius)
