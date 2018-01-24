##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that provides tools to create 3D rendering using VTK.
"""


# System import
from __future__ import print_function
import numpy
import types
import os

# Caps import
from .colors import *
from .animate import images_to_gif

# VTK import
try:
    import vtk
except ImportError:
    raise ImportError("VTK is not installed.")


def ren():
    """ Create a renderer

    Returns
    --------
    ren: vtkRenderer() object

    Examples
    --------
    >>> import pvtk
    >>> ren = pvtk.ren()
    >>> pvtk.add(ren, actor)
    >>> pvtk.show(ren)
    """
    return vtk.vtkRenderer()


def add(ren, actor):
    """ Add a specific actor
    """
    if isinstance(actor, vtk.vtkVolume):
        ren.AddVolume(actor)
    else:
        ren.AddActor(actor)


def rm(ren, actor):
    """ Remove a specific actor
    """
    ren.RemoveActor(actor)


def clear(ren):
    """ Remove all actors from the renderer
    """
    ren.RemoveAllViewProps()


def show(ren, title="pvtk", size=(300, 300), observers=None):
    """ Show window

    Parameters
    ----------
    ren: vtkRenderer() object
        as returned from function ren().
    title: string
        a string for the window title bar.
    size: (int, int)
        (width, height) of the window.
    observers: callable
        functions that will be called at the end of the pick event.
    """
    ren.ResetCameraClippingRange()

    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    window.SetWindowName(title)
    window.SetSize(size)

    iren = vtk.vtkRenderWindowInteractor()

    # picker = vtk.vtkCellPicker()
    picker = vtk.vtkPointPicker()
    if observers is not None:
        actors = ren.GetActors()
        for func in observers:
            func.picker = picker
            func.actors = [actors.GetItemAsObject(index)
                           for index in range(actors.GetNumberOfItems())]
            picker.AddObserver("EndPickEvent", func)

    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetRenderWindow(window)
    iren.SetPicker(picker)
    iren.SetInteractorStyle(style)
    iren.Initialize()

    window.Render()
    iren.Start()


def record(ren, outdir, prefix, cam_pos=None, cam_focal=None,
           cam_view=None, n_frames=1, az_ang=10, size=(300, 300),
           animate=False, delay=100, verbose=False):
    """ This will record a snap/video of the rendered objects.

    Records a video as a series of ".png" files by rotating the azimuth angle
    'az_ang' in every frame.

    Parameters
    ----------
    ren: vtkRenderer() object (mandatory)
        as returned from function ren()
    outdir: str (mandatory)
        the output directory.
    prefix: str (mandatory)
        the png snap base names.
    cam_pos: 3-uplet (optional, default None)
        the camera position.
    cam_focal: 3-uplet (optional, default None)
        the camera focal point.
    cam_view: 3-uplet (optional, default None)
        the camera view up.
    n_frames: int (optional, default 1)
        the number of frames to save.
    az_ang: float (optional, default 10)
        the azimuthal angle of camera rotation (in degrees).
    size: 2-uplet (optional, default (300, 300))
        (width, height) of the window.
    animate: bool (optional, default False)
        if True agglomerate the generated snaps in a Gif and delete the
        raw snaps.
    delay: int (optional, default 100)
        this option is useful for regulating the animation of image
        sequences ticks/ticks-per-second seconds must expire before the
        display of the next image. The default is no delay between each
        showing of the image sequence. The default ticks-per-second is 100.
    verbose: bool (optional, default False)
        if True display debuging message.

    Returns
    -------
    snaps: list of str
        the generated snaps.
    """
    # Create a window and a interactor
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren)
    window.SetSize(size)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(window)

    # Set the camera properties
    ren.ResetCamera()
    camera = ren.GetActiveCamera()
    if cam_pos is not None:
        camera.SetPosition(cam_pos)
    if cam_focal is not None:
        camera.SetFocalPoint(cam_focal)
    if cam_view is not None:
        camera.SetViewUp(cam_view)
    if verbose:
        print("Camera Position (%.2f,%.2f,%.2f)" % camera.GetPosition())
        print("Camera Focal Point (%.2f,%.2f,%.2f)" % camera.GetFocalPoint())
        print("Camera View Up (%.2f,%.2f,%.2f)" % camera.GetViewUp())

    # Create 'n_frames' by rotating each time the scene by 'az_ang' degrees
    writer = vtk.vtkPNGWriter()
    snaps = []
    for index in range(n_frames):
        render = vtk.vtkRenderLargeImage()
        render.SetInput(ren)
        render.SetMagnification(1)
        render.Update()
        writer.SetInputConnection(render.GetOutputPort())
        current_prefix = prefix
        if n_frames > 1:
            current_prefix += "-" + str(index + 1).zfill(8)
        snap_file = os.path.join(outdir, current_prefix + ".png")
        writer.SetFileName(snap_file)
        snaps.append(snap_file)
        writer.Write()
        camera.Azimuth(az_ang)

    # Create an animation
    if animate:
        giffile = os.path.join(outdir, prefix + ".gif")
        images_to_gif(snaps, giffile, delay=delay)
        for fname in snaps:
            os.remove(fname)
        snaps = [giffile]

    return snaps


def text(text, font_size=10, position=(0, 0), color=(0, 0, 0),
         is_visible=True):
    """ Generate a 2d text actor.
    """
    mapper = vtk.vtkTextMapper()
    mapper.SetInput(text)
    properties = mapper.GetTextProperty()
    properties.SetFontFamilyToArial()
    properties.SetFontSize(font_size)
    properties.BoldOn()
    properties.ShadowOn()
    properties.SetColor(color)

    actor = vtk.vtkActor2D()
    actor.SetPosition(position)
    if not is_visible:
        actor.VisibilityOff()
    actor.SetMapper(mapper)

    return actor


def tensor(coeff, order, position=(0, 0, 0),
           radius=0.5, thetares=20, phires=20, opacity=1, tessel=0):
    """ Generate a generic tensor actor.
    """
    from clindmri.estimation.gdti.monomials import (
        construct_matrix_of_monomials)

    # Create a sphere that we will deform
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(radius)
    sphere.SetLatLongTessellation(tessel)
    sphere.SetThetaResolution(thetares)
    sphere.SetPhiResolution(phires)

    # Get the polydata
    poly = sphere.GetOutput()
    poly.Update()

    # Get the mesh
    numPts = poly.GetNumberOfPoints()
    mesh = numpy.zeros((numPts, 3), dtype=numpy.single)
    for i in range(numPts):
        mesh[i, :] = (poly.GetPoint(i)[0], poly.GetPoint(i)[1],
                      poly.GetPoint(i)[2])

    # Deform mesh
    design_matrix = construct_matrix_of_monomials(mesh, order)
    signal = numpy.dot(design_matrix, coeff)
    # signal = np.maximum(signal, 0.0)
    signal /= signal.max()
    signal *= 0.5

    scalars = vtk.vtkFloatArray()
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(numPts)
    for i in range(numPts):
        pts.SetPoint(i, signal[i] * mesh[i, 0], signal[i] * mesh[i, 1],
                     signal[i] * mesh[i, 2])
        scalars.InsertTuple1(i, signal[i])

    poly.SetPoints(pts)
    poly.GetPointData().SetScalars(scalars)
    poly.Update()

    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.667, 0.0)
    lut.Build()

    spherem = vtk.vtkPolyDataMapper()
    spherem.SetInput(poly)
    spherem.SetLookupTable(lut)
    spherem.ScalarVisibilityOn()
    spherem.SetColorModeToMapScalars()
    spherem.SetScalarRange(0.0, 0.5)

    actor = vtk.vtkActor()
    actor.SetMapper(spherem)
    actor.SetPosition(position)
    actor.GetProperty().SetOpacity(opacity)

    return actor


def line(lines, colors, lut=None, opacity=1, linewidth=1):
    """ Create a line actor for one or more lines.

    Parameters
    ----------
    lines : list
        a list of array representing a line as 3d points (N, 3)
    colors : a float or a list of float
        0 <= scalar <= 1 to associate a color to the bloc of lines or
        a list of scalar to associate different color to lines.
    opacity : float (default = 1)
        the transparency of the bloc of lines: 0 <= transparency <= 1.
    linewidth : float (default = 1)
        the line thickness.

    Returns
    ----------
    actor: vtkActor
        the bloc of lines actor.
    """
    # Consteruct le lookup table if necessary
    if lut is None:
        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.667, 0.0)
        lut.Build()

    # If one line is passed as a numpy array, create virtually a list around
    # this structure
    if not isinstance(lines, types.ListType):
        lines = [lines]

    # If one color is passed, create virtually a list around this structure
    if not isinstance(colors, types.ListType):
        colors = [colors] * len(lines)

    # Create vtk structures
    vtk_points = vtk.vtkPoints()
    vtk_line = vtk.vtkCellArray()
    vtk_scalars = vtk.vtkFloatArray()

    # Go through all lines for the rendering
    point_id = 0
    for line, scalar in zip(lines, colors):

        # Get the line size
        nb_of_points, line_dim = line.shape

        # Associate one scalar to each point of the line for color rendering
        scalars = [scalar] * nb_of_points

        # Fill the vtk structure for the curretn line
        for point_position in range(nb_of_points - 1):

            # Get the segment [p0, p1]
            p0 = line[point_position]
            p1 = line[point_position + 1]

            # Set line points
            vtk_points.InsertNextPoint(p0)
            vtk_points.InsertNextPoint(p1)

            # Set color property
            vtk_scalars.SetNumberOfComponents(1)
            vtk_scalars.InsertNextTuple1(scalars[point_position])
            vtk_scalars.InsertNextTuple1(scalars[point_position])

            # Set line segment
            vtk_line.InsertNextCell(2)
            vtk_line.InsertCellPoint(point_id)
            vtk_line.InsertCellPoint(point_id + 1)

            point_id += 2

    # Create the line polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_line)
    polydata.GetPointData().SetScalars(vtk_scalars)

    # Create the line mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(polydata)
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarRange(0.0, 1.0)
    mapper.SetScalarModeToUsePointData()

    # Create the line actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetLineWidth(linewidth)

    return actor


def tubes(lines, colors, opacity=1, linewidth=0.15, tube_sides=8,
          lod=True, lod_points=10 ** 4, lod_points_size=5):
    """ Uses streamtubes to visualize polylines.


    Parameters
    ----------
    lines : list
        a list of array representing a line as 3d points (N, 3)
    colors : array (N, 3)
        rgb colors.
    opacity : float (default = 1)
        the transparency of the bloc of lines: 0 <= transparency <= 1.
    linewidth : float (default = 1)
        the line thickness.
    tube_sides: int
        the tube resolution.
    lod: bool
        use vtkLODActor rather than vtkActor.
    lod_points: int
        number of points to be used when LOD is in effect.
    lod_points_size: int
        size of points when lod is in effect.

    Returns
    ----------
    actor: vtkActor or vtkLODActor
        the bloc of tubes actor.
    """
    points = vtk.vtkPoints()

    colors = numpy.asarray(colors)
    if colors.ndim == 1:
        colors = numpy.tile(colors, (len(lines), 1))

    # Create the polyline.
    streamlines = vtk.vtkCellArray()

    cols = vtk.vtkUnsignedCharArray()
    cols.SetName("Cols")
    cols.SetNumberOfComponents(3)

    len_lines = len(lines)
    prior_line_shape = 0
    for i in range(len_lines):
        line = lines[i]
        streamlines.InsertNextCell(line.shape[0])
        for j in range(line.shape[0]):
            points.InsertNextPoint(*line[j])
            streamlines.InsertCellPoint(j + prior_line_shape)
            color = (255 * colors[i]).astype('ubyte')
            cols.InsertNextTuple3(*color)
        prior_line_shape += line.shape[0]

    profileData = vtk.vtkPolyData()
    profileData.SetPoints(points)
    profileData.SetLines(streamlines)
    profileData.GetPointData().AddArray(cols)

    # Add thickness to the resulting line.
    profileTubes = vtk.vtkTubeFilter()
    profileTubes.SetNumberOfSides(tube_sides)

    if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        profileTubes.SetInput(profileData)
    else:
        profileTubes.SetInputData(profileData)

    # profileTubes.SetInput(profileData)
    profileTubes.SetRadius(linewidth)

    profileMapper = vtk.vtkPolyDataMapper()
    profileMapper.SetInputConnection(profileTubes.GetOutputPort())
    profileMapper.ScalarVisibilityOn()
    profileMapper.SetScalarModeToUsePointFieldData()
    profileMapper.SelectColorArray("Cols")
    profileMapper.GlobalImmediateModeRenderingOn()

    if lod:
        profile = vtk.vtkLODActor()
        profile.SetNumberOfCloudPoints(lod_points)
        profile.GetProperty().SetPointSize(lod_points_size)
    else:
        profile = vtk.vtkActor()
    profile.SetMapper(profileMapper)

    profile.GetProperty().SetAmbient(0)  # .3
    profile.GetProperty().SetSpecular(0)  # .3
    profile.GetProperty().SetSpecularPower(10)
    profile.GetProperty().SetInterpolationToGouraud()
    profile.GetProperty().BackfaceCullingOn()
    profile.GetProperty().SetOpacity(opacity)

    return profile


def dots(points, color=(1, 0, 0), psize=1, opacity=1):
    """ Create one or more 3d dot points.

    Returns
    -------
    actor: vtkActor
        one actor handling all the points.
    """
    if points.ndim == 2:
        points_no = points.shape[0]
    else:
        points_no = 1

    polyVertexPoints = vtk.vtkPoints()
    polyVertexPoints.SetNumberOfPoints(points_no)
    aPolyVertex = vtk.vtkPolyVertex()
    aPolyVertex.GetPointIds().SetNumberOfIds(points_no)

    cnt = 0
    if points.ndim > 1:
        for point in points:
            polyVertexPoints.InsertPoint(cnt, point[0], point[1], point[2])
            aPolyVertex.GetPointIds().SetId(cnt, cnt)
            cnt += 1
    else:
        polyVertexPoints.InsertPoint(cnt, points[0], points[1], points[2])
        aPolyVertex.GetPointIds().SetId(cnt, cnt)
        cnt += 1

    aPolyVertexGrid = vtk.vtkUnstructuredGrid()
    aPolyVertexGrid.Allocate(1, 1)
    aPolyVertexGrid.InsertNextCell(aPolyVertex.GetCellType(),
                                   aPolyVertex.GetPointIds())

    aPolyVertexGrid.SetPoints(polyVertexPoints)
    aPolyVertexMapper = vtk.vtkDataSetMapper()
    if vtk.VTK_MAJOR_VERSION <= 5:
        aPolyVertexMapper.SetInput(aPolyVertexGrid)
    else:
        aPolyVertexMapper.SetInputData(aPolyVertexGrid)
    aPolyVertexActor = vtk.vtkActor()
    aPolyVertexActor.SetMapper(aPolyVertexMapper)

    aPolyVertexActor.GetProperty().SetColor(color)
    aPolyVertexActor.GetProperty().SetOpacity(opacity)
    aPolyVertexActor.GetProperty().SetPointSize(psize)
    return aPolyVertexActor


def surface(points, triangles, labels, ctab=None, opacity=1, set_lut=True,
            decimation_ratio=0.):
    """ Create a colored triangular surface.

    Parameters
    ----------
    points: array (n_vertices, 3)
        the surface vertices.
    triangles: array
        nfaces x 3 array defining mesh triangles.
    labels: array (n_vertices)
        Annotation id at each vertex.
        If a vertex does not belong to any label its id must be negative.
    ctab: ndarray (n_labels, 5) (optional, default None)
        RGBA + label id color table array. If None a default blue to red
        256 levels lookup table is used.
    opacity: float (optional, default 1)
        the actor global opacity.
    set_lut: bool (optional, default True)
        if True set a tuned lut.
    decimation_ratio: float (optional, default 0)
        how many triangles should reduced by specifying the percentage
        ([0,1]) of triangles to be removed.

    Returns
    -------
    actor: vtkActor
        one actor handling the surface.
    """
    # First setup points, triangles and colors
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(1)
    labels[numpy.where(labels < 0)] = 0
    for index in range(len(points)):
        vtk_points.InsertNextPoint(points[index])
        vtk_colors.InsertNextTuple1(labels[index])
    for cnt, triangle in enumerate(triangles):
        vtk_triangle = vtk.vtkTriangle()
        vtk_triangle.GetPointIds().SetId(0, triangle[0])
        vtk_triangle.GetPointIds().SetId(1, triangle[1])
        vtk_triangle.GetPointIds().SetId(2, triangle[2])
        vtk_triangles.InsertNextCell(vtk_triangle)

    # Make a lookup table using vtkColorSeries
    lut = vtk.vtkLookupTable()
    if ctab is not None:
        nb_of_labels = len(ctab)
        lut.SetNumberOfColors(nb_of_labels)
        lut.Build()
        for cnt, lut_element in enumerate(ctab):
            lut.SetTableValue(
                cnt, lut_element[0] / 255., lut_element[1] / 255.,
                lut_element[2] / 255., lut_element[3] / 255.)
        lut.SetNanColor(1, 0, 0, 1)
    # This creates a blue to red lut.
    else:
        nb_of_labels = 255
        lut.SetHueRange(0.667, 0.0)
        lut.SetNumberOfColors(nb_of_labels)
        lut.Build()

    # Create (geometry and topology) the associated polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.GetPointData().SetScalars(vtk_colors)
    polydata.SetPolys(vtk_triangles)

    # Decimate the mesh
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputConnection(polydata.GetProducerPort())
    decimate.SetTargetReduction(decimation_ratio)

    # Create the mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(decimate.GetOutputPort())
    if set_lut:
        mapper.SetLookupTable(lut)
        mapper.SetColorModeToMapScalars()
        mapper.SetScalarRange(0, nb_of_labels)
        mapper.SetScalarModeToUsePointData()
    else:
        mapper.ScalarVisibilityOff()

    # Create the actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(0.9, 0.9, 0.9)

    return actor
