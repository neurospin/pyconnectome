#! /usr/bin/env python
# -*- coding: utf-8 -*

##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
##########################################################################

# Standard import
import os
import nibabel
import subprocess


# Third-party import
from .slicer import plot_image
from pyfreesurfer.wrapper import FSWrapper
from pyfreesufer import DEFAULT_FREESURFER_PATH


def freeview_snapshot(base, outdir, overlay=None, outline_overlay=False,
                      overlay_opacity=0.5, overlay_colormap="jet",
                      basename=None, view="y", index=None,
                      view_size=(500, 500), shfile=DEFAULT_FREESURFER_PATH):
    """
    Create a PNG snapshot using Freeview. Requires FreeSurfer version >= 6.0.

    Parameters
    ----------
    base: str
        Path to the image to be plotted, on which an overlay can be added.
    outdir: str
        Directory where to output.
    overlay: str, default None
        Path to an image to be overlayed on the base image.
    outline_overlay: bool, default False
        Outline the overlay image to show only contours of mask or labels.
    overlay_opacity: float, default 0.5
        Opacity of the overlay.
    overlay_colormap: str, default "jet"
        Colormap of the overlay. According to Freeview doc:
        "Valid names are grayscale/lut/heat/jet/gecolor/nih/pet".
    basename: str, default None
        Basename without extension of the output PNG. If None use the same
        name as the base image.
    view: str, default "y"
        According to Freeview doc: "Accepted names are 'sagittal' or 'x',
       'coronal' or 'y', 'axial' or 'z' and '3d'"
    index: int, default None
        Position of the slice in the <view> direction to be plotted.
        If None the middle slice is used.
    size: couple of ints, default (500, 500)
        Resolution in pixels (nrows, ncols) of the PNG snapshot.
    shfile: str, default <NeuroSpin path>
        Path to the FreeSurfer 'SetUpFreeSurfer.sh' configuration file.

    Returns
    -------
    snapshot: str
        Path to the output snapshot: <outdir>/<basename>.png
    """
    # Index argument not implemented: control the position in the volume
    raise NotImplementedError()

    if basename is None:
        basename = os.path.basename(base)

    if index is None:
        pass

    # Snapshot path
    snapshot = os.path.join(outdir, basename + ".png")

    # Create Freeview command
    cmd = ["freeview", "--viewport", view, "-v", base,
           "--screenshot", snapshot]
    if overlay is not None:
        args = (overlay, outline_overlay, overlay_opacity, overlay_colormap)
        cmd += ["%s:outline:%i:opacity=%f:colormap=%s" % args]

    # Run Freeview
    FSWrapper(cmd=cmd, shfile=shfile)

    return snapshot


def fiber_density_map(tracks, template, outdir, basename=None,
                      fiber_ends_only=False, create_pdf=True,
                      overlay=False):
    """
    Create a density map of fibers or fiber-ends as a Nifti, along with a
    PDF file if requested.

    Parameters
    ----------
    tracks: str
        Path to the streamlines. It has to be loadable by nibabel.
    template: str
        Path to Nifti defining the space of output density map (e.g. nodif).
    outdir: str
        Path to directory where to output.
    basename: str, default None
        Filename without extension of output files. By default None, means
        "fiber_ends_density" if 'fiber_ends_only' is True else "fiber_density".
    fiber_ends_only: bool, default False
        If true, create a density map of fiber ends.
    create_pdf: bool, default True
        If true create a pdf containing snapshots of the density map.
    overlay: bool, default False
        If True, fibers are overlayed on the template in the PDF snapshot.
        Otherwise the fiber density is plotted alone.

    Return
    ------
    fiber_density_map: str
        Path to the output Nifti density map.
    pdf: str or None
        Path to the output PDF file, if requested (create_pdf = True).
    """
    # Import here to avoid dependency on dipy for the whole package
    from dipy.tracking.utils import density_map

    if basename is None:
        basename = "fiber_ends_density" if fiber_ends_only else "fiber_density"

    # Load tracks
    tracks_obj = nibabel.streamlines.load(tracks)

    # Load template image: needed to get affine matrix and shape
    template_obj = nibabel.load(template)

    # If user has requested density of fiber ends
    if fiber_ends_only:
        # Keep only the last point of the fibers (but keep a 2d-array)
        streamlines = [arr[-1:, :] for arr in tracks_obj.streamlines]
    else:
        streamlines = tracks_obj.streamlines

    # Compute fiber count in each voxel
    density_map_arr = density_map(streamlines=streamlines,
                                  vol_dims=template_obj.get_shape(),
                                  affine=tracks_obj.affine)

    # Output paths
    fiber_density_map = os.path.join(outdir, basename + ".nii.gz")
    pdf = os.path.join(outdir, basename + ".pdf") if create_pdf else None

    # Create and write Nifti
    density_map_nii = nibabel.Nifti1Image(density_map_arr.astype("int"),
                                          template_obj.get_affine())
    density_map_nii.to_filename(fiber_density_map)

    # If requested create PDF file
    if create_pdf:
        # Common args between the 2 calls (with or without overlay)
        kwargs = dict(snap_file=pdf, cut_coords=template_obj.get_shape()[2],
                      figsize=(10, 10), nbcols=8)
        # If requested use fibers as overlay on the template
        if overlay:
            plot_image(input_file=template, overlay_file=fiber_density_map,
                       **kwargs)
        else:
            plot_image(input_file=fiber_density_map, **kwargs)

    return fiber_density_map, pdf


def fiber_length_histogram(tracks, outdir, basename="hist_fiber_lengths",
                           ext=".png", bins=20):
    """
    Create a snapshot showing the histogram of fiber lengths.

    Parameters
    ----------
    tracks: str
        Path to the streamlines. It has to be loadable by nibabel.
    outdir: str
        Path to directory where to output.
    basename: str, default "hist_fiber_lengths"
        Filename without extension of output files.
    ext: str, default ".png"
        Snapshot extension, used to specify the output format.
    bins: int, default 20
        Number of bins in the histogram.

    Return
    ------
    snapshot: str
        Path to the output snapshot.
    """
    # Import here to avoid dependencies on dipy and matplotlib
    # for the whole package
    from dipy.tracking.utils import length
    import matplotlib.pyplot as plt

    # Load tracks
    tracks_obj = nibabel.streamlines.load(tracks)

    # Code inspired by dipy documentation, seen at:
    # http://nipy.org/dipy/examples_built/streamline_length.html

    fiber_lengths = list(length(tracks_obj.streamlines))

    # Create plot with Matplotlib
    snapshot = os.path.join(outdir, basename + ext)
    fig, ax = plt.subplots(1)
    ax.hist(fiber_lengths, color='burlywood', bins=bins)
    ax.set_xlabel('Length')
    ax.set_ylabel('Count')
    fig.savefig(snapshot)

    return snapshot


def fsleyes_tracto_mask_snapshot(tracto_mask, nodif, outdir,
                                 basename="tractography_mask",
                                 mask_alpha=0.5):
    """
    Create a tri-view snapshot of the tracto_mask overlayed on the diffusion,
    using FSLeyes. Used to check the Connectomist tractography mask.

    Parameters
    ----------
    tracto_mask: str
        Path to the tractography mask.
    nodif: str
        Path to the diffusion reference on which the tracto_mask is overlayed.
    outdir: str
        Directoy where to output.
    basename: str, default "tractography_mask"
        Basename without extension of the output snapshot.

    Return
    ------
    snapshot: str
        Path to the output snapshot: <outdir>/<basename>.png
    """
    snapshot = os.path.join(outdir, basename + ".png")

    cmd = ["python", "-m", "fsleyes.main", "render",
           "--hideCursor", "--outfile", snapshot, nodif, tracto_mask,
           "--alpha", "%i" % mask_alpha, "--cmap", "red"]
    subprocess.check_call(cmd)

    return snapshot
