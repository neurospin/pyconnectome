##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that provides tools to create 3D rendering using VTK.

https://gitlab.kitware.com/vtk/vtk/blob/master/IO/Image/
vtkNIFTIImageReader.cxx#L738

// === Image Orientation in NIfTI files ===
//
// The vtkImageData class does not provide a way of storing image
// orientation.  So when we read a NIFTI file, we should also provide
// the user with a 4x4 matrix that can transform VTK's data coordinates
// into NIFTI's intended coordinate system for the image.  NIFTI defines
// these coordinate systems as:
// 1) NIFTI_XFORM_SCANNER_ANAT - coordinate system of the imaging device
// 2) NIFTI_XFORM_ALIGNED_ANAT - result of registration to another image
// 3) NIFTI_XFORM_TALAIRACH - a brain-specific coordinate system
// 4) NIFTI_XFORM_MNI_152 - a similar brain-specific coordinate system
//
// NIFTI images can store orientation in two ways:
// 1) via a quaternion (orientation and offset, i.e. rigid-body)
// 2) via a matrix (used to store e.g. the results of registration)
//
// A NIFTI file can have both a quaternion (qform) and matrix (sform)
// stored in the same file.  The NIFTI documentation recommends that
// the qform be used to record the "scanner anatomical" coordinates
// and that the sform, if present, be used to define a secondary
// coordinate system, e.g. a coordinate system derived through
// registration to a template.
//
// -- Quaternion Representation --
//
// If the "quaternion" form is used, then the following equation
// defines the transformation from voxel indices to NIFTI's world
// coordinates, where R is the rotation matrix computed from the
// quaternion components:
//
//   [ x ]   [ R11 R12 R13 ] [ pixdim[1] * i        ]   [ qoffset_x ]
//   [ y ] = [ R21 R22 R23 ] [ pixdim[2] * j        ] + [ qoffset_y ]
//   [ z ]   [ R31 R32 R33 ] [ pixdim[3] * k * qfac ]   [ qoffset_z ]
//
// qfac is stored in pixdim[0], if it is equal to -1 then the slices
// are stacked in reverse: VTK will have to reorder the slices in order
// to maintain a right-handed coordinate transformation between indices
// and coordinates.
//
// Let's call VTK data coordinates X,Y,Z to distinguish them from
// the NIFTI coordinates x,y,z.  The relationship between X,Y,Z and
// x,y,z is expressed by a 4x4 matrix M:
//
//   [ x ]   [ M11 M12 M13 M14 ] [ X ]
//   [ y ] = [ M21 M22 M23 M24 ] [ Y ]
//   [ z ]   [ M31 M32 M33 M34 ] [ Z ]
//   [ 1 ]   [ 0   0   0   1   ] [ 1 ]
//
// where the VTK data coordinates X,Y,Z are related to the
// VTK structured coordinates IJK (i.e. point indices) by:
//
//   X = I*Spacing[0] + Origin[0]
//   Y = J*Spacing[1] + Origin[1]
//   Z = K*Spacing[2] + Origin[2]
//
// Now let's consider: when we read a NIFTI image, how should we set
// the Spacing, the Origin, and the matrix M?  Let's consider the
// cases:
//
// 1) If there is no qform, then R is identity and qoffset is zero,
//    and qfac will be 1 (never -1).  So:
//      I,J,K = i,j,k, Spacing = pixdim, Origin = 0, M = Identity
//
// 2) If there is a qform, and qfac is 1, then:
//
//    I,J,K = i,j,k (i.e. voxel order in VTK same as in NIFTI)
//
//    Spacing[0] = pixdim[1]
//    Spacing[1] = pixdim[2]
//    Spacing[2] = pixdim[3]
//
//    Origin[0] = 0.0
//    Origin[1] = 0.0
//    Origin[2] = 0.0
//
//        [ R11 R12 R13 qoffset_x ]
//    M = [ R21 R22 R23 qoffset_y ]
//        [ R31 R32 R33 qoffset_z ]
//        [ 0   0   0   1         ]
//
//    Note that we cannot store qoffset in the origin.  That would
//    be mathematically incorrect.  It would only give us the right
//    offset when R is the identity matrix.
//
// 3) If there is a qform and qfac is -1, then the situation is more
//    compilcated.  We have three choices, each of which is a compromise:
//    a) we can use Spacing[2] = qfac*pixdim[3], i.e. use a negative
//       slice spacing, which might cause some VTK algorithms to
//       misbehave (the VTK tests only use images with positive spacing).
//    b) we can use M13 = -R13, M23 = -R23, M33 = -R33 i.e. introduce
//       a flip into the matrix, which is very bad for VTK rendering
//       algorithms and should definitely be avoided.
//    c) we can reverse the order of the slices in VTK relative to
//       NIFTI, which allows us to preserve positive spacing and retain
//       a well-behaved rotation matrix, by using these equations:
//
//         K = number_of_slices - k - 1
//
//         M14 = qoffset_x - (number_of_slices - 1)*pixdim[3]*R13
//         M24 = qoffset_y - (number_of_slices - 1)*pixdim[3]*R23
//         M34 = qoffset_z - (number_of_slices - 1)*pixdim[3]*R33
//
//       This will give us data that will be well-behaved in VTK, at
//       the expense of making VTK slice numbers not match with
//       the original NIFTI slice numbers.  NIFTI slice 0 will become
//       VTK slice N-1, and the order will be reversed.
//
// -- Matrix Representation --
//
// If the "matrix" form is used, then pixdim[] is ignored, and the
// voxel spacing is implicitly stored in the matrix.  In addition,
// the matrix may have a negative determinant, there is no "qfac"
// flip-factor as there is in the quaternion representation.
//
// Let S be the matrix stored in the NIFTI header, and let M be our
// desired coordinate transformation from VTK data coordinates X,Y,Z
// to NIFTI data coordinates x,y,z (see discussion above for more
// information).  Let's consider the cases where the determinant
// is positive, or negative.
//
// 1) If the determinant is positive, we will factor the spacing
//    (but not the origin) out of the matrix.
//
//    Spacing[0] = pixdim[1]
//    Spacing[1] = pixdim[2]
//    Spacing[2] = pixdim[3]
//
//    Origin[0] = 0.0
//    Origin[1] = 0.0
//    Origin[2] = 0.0
//
//         [ S11/pixdim[1] S12/pixdim[2] S13/pixdim[3] S14 ]
//    M  = [ S21/pixdim[1] S22/pixdim[2] S23/pixdim[3] S24 ]
//         [ S31/pixdim[1] S32/pixdim[2] S33/pixdim[3] S34 ]
//         [ 0             0             0             1   ]
//
// 2) If the determinant is negative, then we face the same choices
//    as when qfac is -1 for the quaternion transformation.  We can:
//    a) use a negative Z spacing and multiply the 3rd column of M by -1
//    b) keep the matrix as is (with a negative determinant)
//    c) reorder the slices, multiply the 3rd column by -1, and adjust
//       the 4th column of the matrix:
//
//         M14 = S14 + (number_of_slices - 1)*S13
//         M24 = S24 + (number_of_slices - 1)*S23
//         M34 = S34 + (number_of_slices - 1)*S33
//
//       The third choice will provide a VTK image that has positive
//       spacing and a matrix with a positive determinant.
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
from vtk.util import numpy_support


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
    ren.ResetCamera()
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
            decimation_ratio=0., smooth=False):
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
    smooth: bool (optional, default False)
        if set smooth the mesh using a Laplacian smoothing.

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
        nb_of_labels = ctab[4].max()
        lut.SetNumberOfColors(nb_of_labels)
        lut.Build()
        for cnt, lut_element in enumerate(ctab):
            lut.SetTableValue(
                lut_element[4], lut_element[0] / 255., lut_element[1] / 255.,
                lut_element[2] / 255., lut_element[3] / 255.)
        lut.SetNanColor(0, 0, 0, 1)
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
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(decimation_ratio)

    # Smooth the mesh
    it = 0
    if smooth:
        it = 30
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(decimate.GetOutputPort())
    smooth.SetRelaxationFactor(0.1)
    smooth.SetNumberOfIterations(it)
    smooth.FeatureEdgeSmoothingOff()
    smooth.BoundarySmoothingOn()

    # Create the mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(smooth.GetOutputPort())
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


def skin(image_path, thres=500):
    """ Extract the skin and create a mesh.

    Parameters
    ----------
    image_path: str
        the MRI image we want to extract the skin.
    thres: int
        the skin threshold.

    Returns
    -------
    actor: vtkActor
        one actor handling the surface.
    """
    # Read the inputs Nifti Image
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(image_path)

    # An isosurface, or contour value of 500 is known to correspond to the
    # skin of the patient. Once generated, a vtkPolyDataNormals filter is
    # is used to create normals for smooth surface shading during rendering.
    # The triangle stripper is used to create triangle strips from the
    # isosurface these render much faster on may systems.
    skinExtractor = vtk.vtkContourFilter()
    skinExtractor.SetInputConnection(reader.GetOutputPort())
    skinExtractor.SetValue(0, thres)
    skinNormals = vtk.vtkPolyDataNormals()
    skinNormals.SetInputConnection(skinExtractor.GetOutputPort())
    skinNormals.SetFeatureAngle(60.0)
    skinStripper = vtk.vtkStripper()
    skinStripper.SetInputConnection(skinNormals.GetOutputPort())
    skinMapper = vtk.vtkPolyDataMapper()
    skinMapper.SetInputConnection(skinStripper.GetOutputPort())
    skinMapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(skinMapper)
    actor.GetProperty().SetDiffuseColor(1, .49, .25)
    actor.GetProperty().SetSpecular(.3)
    actor.GetProperty().SetSpecularPower(20)

    return actor


def mask_surface(mask, opacity=1, color=(1, 0, 0)):
    """ Extract a mesh from a binary image.

    Parameters
    ----------
    mask: str or ndarray
        the MRI mask image we want to extract the surface.
    opacity: float, default 1
        the actor global opacity.
    color: 3-uplet, default (1, 0, 0)
        the generated mesh color.

    Returns
    -------
    actor: vtkActor
        one actor handling the surface.
    """
    # Create mesh
    dmc = vtk.vtkDiscreteMarchingCubes()
    if isinstance(mask, numpy.ndarray):
        vtk_mask = numpy_support.numpy_to_vtk(
            num_array=mask.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        dmc.SetInputConnection(vtk_mask.Getoutput())
    else:
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(mask)
        dmc.SetInputConnection(reader.GetOutputPort())
    dmcMapper = vtk.vtkPolyDataMapper()
    dmcMapper.SetInputConnection(dmc.GetOutputPort())
    dmcMapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(dmcMapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(opacity)

    return actor
