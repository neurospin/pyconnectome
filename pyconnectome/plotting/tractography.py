#! /usr/bin/env python
# -*- coding: utf-8 -*
##########################################################################
# NSAp - Copyright (C) CEA, 2017
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html for details.
##########################################################################

"""
Modules that provides tractography renderings.
"""


# Standard import
import os
import nibabel
import subprocess

# Package import
from pyconnectome.utils.reorient import reorient_image

# Third party import
from nilearn import plotting


def fiber_density_map(tracks, template, outdir, basename=None,
                      fiber_ends_only=False, overlay=False, overlay_alpha=None,
                      ext=".png", axes="RAS"):
    """ Create a density map of fibers or fiber-ends as a Nifti, along with a
    snap file if requested.

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
    overlay: bool, default False
        If True, fibers are overlayed on the template in the PDF snapshot.
        Otherwise the fiber density is plotted alone.
    overlay_alpha: float, default None
        fix the overlay alpha value (0-1).
    ext: str, default '.png'
        Snapshot extension, used to specify the output format.
    axes: str, default 'RAS'
        orientation of the original axes X, Y, and Z
        specified with the following convention: L=Left, R=Right,
        A=Anterion, P=Posterior, I=Inferior, S=Superior.

    Return
    ------
    fiber_density_map: str
        Path to the output Nifti density map.
    fiber_density_snap: str or None
        Path to the output PDF file, if requested (create_pdf = True).
    """
    # Import here to avoid dependency on dipy & pydcmio for the whole package
    from dipy.tracking.utils import density_map
    from pydcmio.plotting.slicer import mosaic

    # Create the default output file name
    if basename is None:
        basename = "fiber_ends_density" if fiber_ends_only else "fiber_density"

    # Load tracks
    tracks = nibabel.streamlines.load(tracks)

    # Load template image: needed to get affine matrix and shape
    template_im = nibabel.load(template)

    # If user has requested density of fiber ends: keep only the end points
    # of the fibers as a 2d-array
    if fiber_ends_only:
        streamlines = [arr[[0, -1]] for arr in tracks.streamlines]
    else:
        streamlines = tracks.streamlines

    # Compute fiber count in each voxel
    density_map_arr = density_map(streamlines=streamlines,
                                  vol_dims=template_im.get_shape(),
                                  affine=tracks.affine)

    # Create and write Nifti
    fiber_density_map = os.path.join(outdir, basename + ".nii.gz")
    density_map_nii = nibabel.Nifti1Image(density_map_arr.astype("int"),
                                          template_im.get_affine())
    density_map_nii.to_filename(fiber_density_map)

    # Swap axes
    if axes != "RAS":
        reorient_image(
            in_file=fiber_density_map,
            axes=axes,
            prefix="",
            output_directory=None,
            is_direct=False)

    # If requested use fibers as overlay on the template
    kwargs = dict(outdir=outdir, title="Density Map", basename=basename,
                  ext=ext, overlay_alpha=overlay_alpha)
    if overlay:
        fiber_density_snap = mosaic(impath=template, overlay=fiber_density_map,
                                    **kwargs)
    else:
        fiber_density_snap = mosaic(impath=fiber_density_map, **kwargs)

    return fiber_density_map, fiber_density_snap


def fiber_length_histogram(tracks, outdir, basename="hist_fiber_lengths",
                           ext=".png", bins=20):
    """ Create a snapshot showing the histogram of fiber lengths.

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


def fsleyes_snapshot(inputfile, outdir, overlayfile=None,
                     basename="fsleyes_snap", mask_alpha=50,
                     cmap="greyscale", mask_cmap="red",
                     dr=None, mask_dr=None):
    """ Create a tri-view snapshot with one overlay.

    Parameters
    ----------
    inputfile: str
        path to the input image to snap.
    outdir: str
        directoy where to output.
    overlayfile: str, default None
        path to the image to overlay.
    basename: str, default 'fsleyes_snap'
        basename without extension of the output snapshot.
    mask_alpha: int, default 50
        a overlay alpha value (0-100).
    cmap: str, default 'greyscale'
        a valid FSLeyes color map for the input file.
    mask_cmap: str, default 'red'
        a valid FSLeyes color map for the mask file.
    dr: 2-uplet, default None
        the display range (LO, HI) for the input file.
    mask_dr: 2-uplet, default None
        the display range (LO, HI) for the mask file.

    Return
    ------
    snap: str
        path to the output snapshot: <outdir>/<basename>.png
    """
    snap = os.path.join(outdir, basename + ".png")
    cmd = ["python", "-m", "fsleyes.main", "render",
           "--hideCursor", "--outfile", snap, inputfile, "--cmap", cmap]
    if dr is not None:
        cmd += ["-dr", str(dr[0]), str(dr[1])]
    if overlayfile is not None:
        cmd += [overlayfile, "--alpha", str(mask_alpha), "--cmap", mask_cmap]
        if mask_dr is not None:
            cmd += ["-dr", str(mask_dr[0]), str(mask_dr[1])]
    subprocess.check_call(cmd)

    return snap


def nilearn_snapshot(inputfile, outdir, overlayfile=None,
                     basename="nilearn_snap", mask_alpha=50,
                     cmap=None, mask_cmap=None,
                     dr=None, mask_dr=None, black_bg=True):
    """ Create a tri-view snapshot with one overlay.

    Parameters
    ----------
    inputfile: str
        path to the input image to snap.
    outdir: str
        directoy where to output.
    overlayfile: str, default None
        path to the image to overlay.
    basename: str, default 'nilearn_snap'
        basename without extension of the output snapshot.
    mask_alpha: int, default 50
        a overlay alpha value (0-100).
    cmap: str, default None
        a valid nilearn color map for the input file.
    mask_cmap: str, default None
        a valid nilearn color map for the mask file.
    dr: 2-uplet, default None
        athe display range (LO, HI) for the input file.
    mask_dr: 2-uplet, default None
        athe display range (LO, HI) for the mask file.
    black_bg: bool, default True
        If True, the background of the image is set to be black

    Return
    ------
    snap: str
        path to the output snapshot: <outdir>/<basename>.png
    """
    # Display the image
    kwargs = {}
    if cmap is not None:
        try:
            kwargs["cmap"] = getattr(plotting.cm, cmap)
        except:
            raise ValueError("Invalid color map '{0}'.".format(cmap))
    if dr is not None:
        kwargs["vmin"] = dr[0]
        kwargs["vmax"] = dr[1]
    display = plotting.plot_anat(inputfile, black_bg=black_bg,
                                 draw_cross=False, **kwargs)

    # Add the overlay
    if overlayfile is not None:
        kwargs = {}
        if mask_cmap is not None:
            try:
                kwargs["cmap"] = getattr(plotting.cm, mask_cmap)
            except:
                raise ValueError("Invalid color map '{0}'.".format(mask_cmap))
        if mask_dr is not None:
            kwargs["vmin"] = mask_dr[0]
            kwargs["vmax"] = mask_dr[1]
        if mask_alpha is not None:
            kwargs["alpha"] = mask_alpha / 100.
        display.add_overlay(inputfile, **kwargs)

    # Save the result
    snap = os.path.join(outdir, basename + ".png")
    display.savefig(snap)
    display.close()

    return snap
