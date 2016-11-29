##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
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
