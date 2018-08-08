##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that defines simple image slicer methods.
"""

# System imports
import os

# Third party imports
import nibabel
import numpy
from nilearn import plotting
import matplotlib.pyplot as plt


def triplanar(input_file, output_fileroot, title=None, nb_slices=1,
              overlays=None, overlays_colors=None, overlay_opacities=None,
              contours=False, edges=False, marker_coords=None,
              resolution=300):
    """ Snap an image with edge/overlay/contour on top (useful for checking
    registration).

    Parameters
    ----------
    input_file: str
        the input image.
    output_fileroot: str
        output fileroot.
    title: str, default None
        the snap title.
    nb_slices: int
        number of slices outputted.
    overlays: array of str
        array of paths to overlay images
    overlays_colors: list of str or int
        overlay images color index.
    overlay_opacities: list of float
        overlay images opacities.
    contours: bool
        if set to True, add overlays as contours.
    edges: bool
        if set to True, add overlays as edges.
    marker_coords: 3-uplet
        Coordinates of the markers to plot.
    resolution: int
        png outputs resolution.

    Returns
    -------
    output_png_file: str
        the generated output snap.
    """
    # Load files
    input_img = nibabel.load(input_file)

    # Create the display
    if input_img.get_data().ndim == 3:
        display = plotting.plot_anat(
            input_img,
            vmin=0,
            vmax=numpy.percentile(input_img.get_data(), 98),
            display_mode="ortho",
            title=title,
            draw_cross=False if marker_coords is None else True,
            cut_coords=marker_coords)
    else:
        display = plotting.plot_epi(
            input_img,
            vmin=0,
            vmax=numpy.percentile(input_img.get_data(), 98),
            display_mode="ortho",
            draw_cross=False if marker_coords is None else True,
            cut_coords=marker_coords)

    # Add overlays
    if overlays is not None and len(overlays) > 0:

        # Get all available cmaps
        maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
        colors = "bgrcmy"

        # Add overlays
        if overlays_colors is None:
            overlays_colors = [None] * len(overlays)
        if overlay_opacities is None:
            overlay_opacities = [None] * len(overlays)
        for overlay, alpha, color in zip(overlays, overlay_opacities,
                                         overlays_colors):
            if isinstance(color, int):
                color = colors[color % len(colors)]
            elif color is not None and color not in maps:
                raise ValueError("Available cmap are: {0}.".format(maps))
            if contours:
                display.add_contours(
                    overlay,
                    threshold=1e-06,
                    colorbar=False,
                    alpha=alpha,
                    cmap=color)
            elif edges:
                display.add_edges(
                    overlay,
                    color=color or "r")
            else:
                display.add_overlay(
                    overlay,
                    threshold=1e-06,
                    colorbar=False,
                    alpha=alpha,
                    cmap=color)

        # Add markers
        if marker_coords is not None:
            marker_coords = numpy.asarray([marker_coords])
            display.add_markers(marker_coords, marker_color="y",
                                marker_size=30)

    # Save image
    output_png_file = output_fileroot + "_ortho.png"
    display.savefig(output_png_file, dpi=resolution)
    display.close()

    return output_png_file
