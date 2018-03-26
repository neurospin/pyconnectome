##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that provides tools to display networks.
"""

# System import
from __future__ import division
import os
import numpy
from collections import OrderedDict

# Package import
from pyconnectome.plotting import pvtk
from pyconnectome.plotting.scenes import network


def dict2list(adict):
    """ Convert a dict to a list sorting by keys.

    Parameters
    ----------
    adict: dict
        a dictionary to be converted.

    Returns
    -------
    out: list
        a sorted list.
    """
    return [adict[k] for k in sorted(adict, reverse=False)]


def plot_network(nodes, labels, weights=None, edges=None, lh_surf=None,
                 rh_surf=None, weight_node_by_color=False,
                 weight_node_by_size=True, edge_weights=None,
                 weight_edge_by_color=False, weight_edge_by_size=True,
                 interactive=True, snap=False, animate=False, outdir=None,
                 name="network", actor_ang=(0., 0., 0.)):
    """ Diplay a network with VTK.

    Parameters
    ----------
    nodes: list of array (3, )
        the network nodes.
    labels: list of str
        the nodes associated labels.
    weights: list or array (N,) (optional, default None)
        the nodes associated weights.
    edges: list of array (2, 3) (optional, default None)
        the network edges.
    *_surf: TriSurface (optional, default None)
        the right and left hemispheres surfaces.
    weight_node_by_color: bool (optional, default False)
        if True and 'weights' parameter specified, color the nodes depending
        on the associated weigths.
    weight_node_by_size: bool (optional, default True)
        if True and 'weights' parameter specified, define the nodes sizes
        depending on the associated weigths.
    edge_weights: list of float (optional, default None)
        the edges associated connection weights.
    weight_edge_by_color: bool (optional, default False)
        if True and 'edge_weights' parameter specified, color the edges
        depending on the associated weigths.
    weight_edge_by_size: bool (optional, default True)
        if True and 'edge_weights' parameter specified, define the edge sizes
        depending on the associated weigths.
    interactive: bool (optional, default True)
        if True display the renderer.
    snap: bool (optional, default False)
        if True create a snap of the scene: need a valid outdir.
    animate: bool (optional, default False)
        if True create a 360 degrees animated gif of the scene: need a valid
        outdir.
    outdir: str (optional, default None)
        an existing directory.
    name: str (optional, default 'folds')
        the basename of the generated files.
    actor_ang: 3-uplet (optinal, default (0, 0, 0))
        the actors x, y, z position (in degrees).
    """
    # Check inputs
    if outdir is not None and not os.path.isdir(outdir):
        raise ValueError("'{0}' is not a valid directory.".format(outdir))

    # Create the renderer
    ren = pvtk.ren()
    ren.SetBackground(1, 1, 1)
    actors, observer = network(nodes, labels, weights=weights, edges=edges,
                               lh_surf=lh_surf, rh_surf=rh_surf,
                               weight_node_by_color=weight_node_by_color,
                               weight_node_by_size=weight_node_by_size,
                               edge_weights=edge_weights,
                               weight_edge_by_color=weight_edge_by_color,
                               weight_edge_by_size=weight_edge_by_size)
    for a in actors:
        if "vtkActor2D" not in repr(a):
            a.RotateX(actor_ang[0])
            a.RotateY(actor_ang[1])
            a.RotateZ(actor_ang[2])
        pvtk.add(ren, a)

    # Show the renderer
    if interactive:
        pvtk.show(ren, title="Network", observers=[observer])

    # Create a snap
    if snap:
        pvtk.record(ren, outdir, name, n_frames=1)

    # Create an animation
    if animate:
        pvtk.record(ren, outdir, name, n_frames=36, az_ang=10, animate=True,
                    delay=25)


def get_surface_parcellation_centroids(lh_surf, rh_surf, labels):
    """ Compute the centroids of a surface parcellation. The result will be
    saved in an label-ordered dictionary.

    Parameters
    ----------
    *_surf: TriSurface
        the right and left hemispheres surface parcellations.

    Returns
    -------
    centroids: OrderedDict
        a dictionnary with region names as keys and associated centroids as
        values.
    """
    # Parameters
    centroids = OrderedDict()

    # Initilized structure
    for key in labels:
        centroids[key.rstrip("\n")] = numpy.array([0., 0., 0.])

    # Go through each parcell and compute the associated centroid
    for surf, hemi in [(lh_surf, "lh"), (rh_surf, "rh")]:
        for index, meta in surf.metadata.items():
            vertices = surf.vertices[numpy.where(surf.labels == index)]
            if vertices.shape[0] != 0:
                key = "ctx-" + hemi + "-" + meta["region"]
                if key not in centroids:
                    raise ValueError("'Unexpected parcell '{0}'.".format(key))
                centroids[key] = numpy.mean(vertices, axis=0)

    return centroids


def matrix(matrix, snapshot, labels=None, transform=None,
           colorbar_title="", dpi=200, labels_size=4,
           vmin=None, vmax=None):
    """
    Create a PNG snapshot of a matrix.

    Parameters
    ----------
    matrix: ndarray
        The connectivity matrix.
    snapshot: str
        Path to the output snapshot.
    labels: str, default None
        The label names. By default no labels.
        Should be ordered like the rows of the connectivity matrix.
    transform: callable, default None
        A Callable function to apply on the matrix (e.g. numpy.log1p).
        By default no transformation is applied.
    colorbar_title: str, default ""
        How to interpret the values of the connectivity,
        e.g. "Log(# of tracks)" or "% of tracks"
    dpi: int, default 200
        "Dot Per Inch", set higher for better resolution.
    labels_size: int, default 4
        The label font size.
    vmin, vmax: float, default None
        The display range.

    Returns
    -------
    snapshot: str
        Path to the output connectome snapshot.
    """
    # Import in function, so that the rest of the module can be used even
    # if matplotlib is not available
    import matplotlib.pyplot as plt

    # Check matrix dimensions
    if matrix.ndim != 2:
        raise ValueError("Connectivity matrix should be a square matrix."
                         "Shape of matrix: {}".format(matrix.shape))
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T

    # Apply transformation if requested
    if transform is not None:
        matrix = transform(matrix)

    # Create the figure with matplotlib
    size_x = 50
    size_y = min(matrix.shape[0] / matrix.shape[1] * 50, 150)
    fig, ax = plt.subplots(figsize=(size_x, size_y))
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(numpy.arange(0.5, matrix.shape[1]))
    ax.tick_params(which="both", axis="x", width=0, length=0)

    # Add the labels if passed
    if labels is not None:
        if len(labels) != matrix.shape[1]:
            raise ValueError(
                "Wrong number of labels: {0}. Should be {1}.".format(
                    len(labels), matrix.shape[1]))
        ax.set_xticklabels(labels, size=labels_size, rotation=90)

    # Set display options
    # ax.set_aspect("equal")
    kwargs = {}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Reds, **kwargs)
    # colorbar = fig.colorbar(heatmap)
    # colorbar.set_label(colorbar_title, rotation=270, labelpad=20)
    fig.tight_layout()

    # Save to PNG file
    if not snapshot.endswith(".png"):
        snapshot += ".png"
    fig.savefig(snapshot, dpi=dpi)

    # Release memory
    fig.clear()
    plt.close()

    return snapshot
