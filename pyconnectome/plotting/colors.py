##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that defines custom colors.
"""


# System import
import numpy


# Define some common colors
red = numpy.array([1, 0, 0])
green = numpy.array([0, 1, 0])
blue = numpy.array([0, 0, 1])
yellow = numpy.array([1, 1, 0])
cyan = numpy.array([0, 1, 1])
azure = numpy.array([0, 0.49, 1])
golden = numpy.array([1, 0.84, 0])
white = numpy.array([1, 1, 1])
black = numpy.array([0, 0, 0])


def line_colors(streamlines):
    """ Create colors for streamlines.

    Parameters
    ----------
    streamlines: list of array
        some streamlines.

    Returns
    -------
    colors: array
        streamline colors based on the orientation.
    """
    colors = []
    for track in streamlines:
        orient = track[-1] - track[0]
        orient = numpy.abs(orient / numpy.linalg.norm(orient))
        colors.append(orient)
    return colors
