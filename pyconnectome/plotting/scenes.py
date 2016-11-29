##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import numpy
import logging
import vtk

# Define the logger
logger = logging.getLogger(__name__)

# Caps import
import pvtk
import colors


class LabelsOnPick(object):
    """ Create a picker callback to display some labels.
    """
    default_message = "Press <p> to pick an object"

    def __init__(self, textactor, picker=None, actors=None,
                 static_position=True, to_keep_actors=None,
                 highlight_selection=False):
        """ Initialize the LabelsOnPick class.

        Parameters
        ----------
        textactor: vtkActor (mandatory)
            an actor with a text mapper.
        picker: vtkPropPicker (optional, default None)
            a picker.
        actors: vtkActor (optional, default None)
            all the actors to interact with.
        static_position: bool (optional, default True)
            if True the text labels will be displayed on the lower left corner,
            otherwise at the picked object position.
        to_keep_actors: list (optional, default None)
            a list of actor labels to keep visibile when an object is picked.
        highlight_selection: bool (optional, default False)
            if True highlight the picked actor and set the other 'actors'
            visibility to false.
        """
        self.textactor = textactor
        self.textmapper = textactor.GetMapper()
        self.picker = picker
        self.actors = None
        self.static_position = static_position
        self.to_keep_actors = to_keep_actors or []
        self.highlight_selection = highlight_selection

    def __call__(self, obj, event):
        """ When an actor is picked, display its label and focus on this
        actor only.
        """
        # Pick an actor
        actor = self.picker.GetProp3D()

        # Restore all actors visibilities
        if actor is None:
            if self.highlight_selection:
                for act in self.actors:
                    act.SetVisibility(True)
            self.textmapper.SetInput(self.default_message)
            self.textactor.VisibilityOn()
        # Focus on the picked actor and display the fold label
        else:
            if self.highlight_selection:
                for act in self.actors:
                    if act.label not in self.to_keep_actors:
                        act.SetVisibility(False)
                actor.SetVisibility(True)
            self.textmapper.SetInput(actor.label)
            self.textactor.VisibilityOn()
            if not self.static_position:
                point = self.picker.GetSelectionPoint()
                self.textactor.SetPosition(point[:2])


def network(nodes, labels, weights=None, edges=None, lh_surf=None,
            rh_surf=None, weight_node_by_color=False, weight_node_by_size=True,
            edge_weights=None, weight_edge_by_color=False,
            weight_edge_by_size=True):
    """ Scene that shows a network.

    Parameters
    ----------
    nodes: list of array (3, )
        the network nodes.
    labels: list of str
        the nodes associated labels.
    weights: list or array (N,) (optional, default None)
        the nodes associated weights.
    edges: list of 2-uplet (optional, default None)
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

    Returns
    ----------
    actors: list of vtkActor
        the scene actors.
    observer: LabelsOnPick
        pickle event callback.
    """
    # Parameters
    actors = []
    label_actors = []

    # Check inputs
    if len(nodes) != len(labels):
        raise ValueError("Nodes and labels must have the same size.")
    if weights is not None and len(nodes) != len(weights):
        raise ValueError("Nodes and associated weights must have the same "
                         "size.")
    if (edges is not None and edge_weights is not None and
            len(edges) != len(edge_weights)):
        raise ValueError("Edges and associated weights must have the same "
                         "size.")

    # Create actor for label and associated callback
    txtactor = pvtk.text(LabelsOnPick.default_message, font_size=15,
                         position=(10, 10), is_visible=True)
    observer = LabelsOnPick(txtactor, static_position=True,
                            to_keep_actors=["right hemisphere surface",
                                            "left hemisphere surface"],
                            highlight_selection=False)
    actors.append(txtactor)

    # Create actors for the surfaces
    for surf, hemi in ((lh_surf, "left"), (rh_surf, "rigth")):
        if surf is not None:
            ctab = [[120., 120., 120., 255.], ] * len(surf.metadata)
            actor = pvtk.surface(
                surf.vertices, surf.triangles, surf.labels,
                ctab, opacity=0.2, decimation_ratio=0.98)
            actor.GetProperty().SetRepresentationToWireframe()
            actor.label = "{0} hemisphere surface".format(hemi)
            actors.append(actor)

    # Create actors for the nodes
    if weights is None:
        for n, l in zip(nodes, labels):
            actor = pvtk.dots(n, color=colors.blue, psize=20, opacity=1)
            actor.label = l
            actors.append(actor)
    else:
        if weight_node_by_color:
            lut = vtk.vtkColorTransferFunction()
            lut.AddRGBPoint(weights.min(), 0.0, 0.0, 1.0)
            lut.AddRGBPoint(weights.max(), 1.0, 0.0, 0.0)
        for n, l, w in zip(nodes, labels, weights):
            if weight_node_by_color:
                color = lut.GetColor(w)
            else:
                color = colors.blue
            if weight_node_by_size:
                size = w
            else:
                size = 20
            actor = pvtk.dots(n, color=color, psize=size, opacity=1)
            actor.label = l
            actors.append(actor)

    # Create actors for the edges
    if edges is not None:
        lut = vtk.vtkColorTransferFunction()
        if edge_weights is not None and weight_edge_by_color:
            lut.AddRGBPoint(0., 0.0, 0.0, 1.0)
            lut.AddRGBPoint(numpy.log(2) / 2., 0.0, 1.0, 0.0)
            lut.AddRGBPoint(numpy.log(2), 1.0, 0.0, 0.0)
        else:
            lut.AddRGBPoint(0, 0.47, 0.47, 0.47)
            lut.AddRGBPoint(1, 0.47, 0.47, 0.47)
        if edge_weights is None:
            for link in edges:
                line = numpy.asarray([nodes[link[0]], nodes[link[1]]])
                actor = pvtk.line([line], colors=0.5, lut=lut, opacity=0.8,
                                  linewidth=1)
                actor.label = "edge: {0} - {1}".format(
                    labels[link[0]], labels[link[1]])
                actors.append(actor)
        else:
            mean_w = numpy.mean(edge_weights)
            for link, w in zip(edges, edge_weights):
                line = numpy.asarray([nodes[link[0]], nodes[link[1]]])
                if weight_edge_by_color and weight_edge_by_size:
                    color = lut.GetColor(numpy.log(w / mean_w + 1.))
                elif weight_edge_by_color and not weight_edge_by_size:
                    color = numpy.log(w / mean_w + 1.)
                else:
                    color = [[120., 120., 120.]]
                if weight_edge_by_size:
                    linewidth = numpy.log(w / mean_w / 5. + 1.) + 0.1
                    actor = pvtk.tubes(
                        [line], numpy.asarray(color), opacity=1,
                        linewidth=linewidth, tube_sides=8, lod=True,
                        lod_points=10 ** 4, lod_points_size=5)
                else:
                    actor = pvtk.line([line], colors=color, lut=lut,
                                      opacity=0.8, linewidth=1)
                actor.label = "edge: {0} - {1}".format(
                    labels[link[0]], labels[link[1]])
                actors.append(actor)

    return actors, observer


def bundle_representative_track_scene(tracks, representative_track_indx):
    """ Scene that shows the bundle and its most representative element.

    Parameters
    ----------
    tracks : sequence (N, )
       of tracks as arrays, shape (N1,3) .. (Nm,3).
    representative_track_indx: int
       index of the representative track of the bundle.

    Returns
    ----------
    actors: list of vtkActor
        the scene actors.
    """
    bundle_actor = pvtk.line(tracks, 1)
    representative_track_actor = pvtk.line(
        tracks[representative_track_indx], 0, linewidth=2)
    return [bundle_actor, representative_track_actor]


def field_directions(field):
    """ Scene the shows the directions of a vector field.

    Parameters
    ----------
    field: array (X, Y, N, 3)
        the vector field to plot where N is the number of peaks.

    Returns
    ----------
    actors: list of vtkActor
        the scene actors.
    """
    actors = []
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            line = numpy.zeros((2, 3), dtype=numpy.single)
            for vector in field[x, y]:
                line[1] = vector
                actors.append(pvtk.line(line, 0, linewidth=2))
                actors[-1].SetPosition((x, y, 0))
    return actors
