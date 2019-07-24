##########################################################################
# NSAP - Copyright (C) CEA, 2013 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
File utilities.
"""

# System import
import os
import glob
import subprocess
import tempfile
import shutil

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper
from pyconnectome.utils.regtools import flirt2aff

# Third party
import numpy
import nibabel
import nibabel.gifti.giftiio as gio
from pyfreesurfer.utils.surftools import TriSurface


def load_folds(folds_file, graph_file=None):
    """ Load morphologist folds and associated labels.

    Parameters
    ----------
    folds_file: str( mandatory)
        the folds '.gii' file.
    graph_file: str (optional, default None)
        the path to a morphologist '.arg' graph file.

    Returns
    -------
    folds: dict with TriSurface
        all the loaded folds. The fold names are stored in the metadata.
    """
    # Load the labels
    if graph_file is not None:
        labels = parse_graph(graph_file)
    else:
        labels = {}

    # Load folds
    image = gio.read(folds_file)
    nb_of_surfs = len(image.darrays)
    if nb_of_surfs % 2 != 0:
        raise ValueError("Need an odd number of arrays (vertices, triangles).")
    folds = {}
    for vertindex in range(0, nb_of_surfs, 2):
        vectices = image.darrays[vertindex].data
        triangles = image.darrays[vertindex + 1].data
        labelindex = image.darrays[vertindex].get_metadata()["Timestep"]
        if labelindex != image.darrays[vertindex + 1].get_metadata()[
                "Timestep"]:
            raise ValueError("Gifti arrays '{0}' and '{1}' do not share the "
                             "same label.".format(vertindex, vertindex + 1))
        labelindex = int(labelindex)
        if labelindex in labels:
            label = labels[labelindex]
        else:
            label = "NC"
        metadata = {"fold_name": label}
        surf = TriSurface(vectices, triangles, labels=None, metadata=metadata)
        folds[labelindex] = surf

    return folds


def parse_graph(graph_file):
    """ Parse a Morphologist graph file to get the fold labels.

    Parameters
    ----------
    graph_file: str (mandatory)
        the path to a morphologist '.arg' graph file.

    Returns
    -------
    labels: dict
        a mapping between a mesh id and its label.
    """
    # Read all the lines in the graph file
    with open(graph_file) as open_file:
        lines = open_file.readlines()

    # Search labels
    infold = False
    meshid = None
    label = None
    labels = {}
    for line in lines:

        # Locate fold items
        if line.startswith("*BEGIN NODE fold"):
            infold = True
            continue
        if infold and line.startswith("*END"):
            if meshid in labels:
                raise ValueError("'{0}' mesh id already found.".format(meshid))
            labels[meshid] = label
            infold = False
            continue

        # In fold item detect the mesh id and the associated label
        if infold and line.startswith("label"):
            label = line.replace("label", "").strip()
        if infold and line.startswith("Tmtktri_label"):
            meshid = int(line.replace("Tmtktri_label", "").strip())

    return labels


def merge_fibers(tractograms, tempdir=None):
    """ Merge tractograms.

    Parameters
    ----------
    tractograms: list of str
        paths to the input tractograms.
    tempdir: str, default None
        a temporary directory to store intermediate tractogram.

    Returns
    -------
    merge_tractogram_file: str
        all the streamlines in one TRK file.
    """
    # Check existence of input file
    for path in tractograms:
        if not os.path.isfile(path):
            raise ValueError("File does not exist: {0}.".format(path))

    # Create a temporary directory to store an intermediate tractogram
    tempdir = tempfile.mkdtemp(prefix="tractconverter_", dir=tempdir)

    # Combine tractograms in one file
    trk = nibabel.streamlines.load(tractograms[0])
    for trk_path in tractograms[1:]:
        part_trk = nibabel.streamlines.load(trk_path)
        trk.streamlines.extend(part_trk.streamlines)
    merge_tractogram_file = os.path.join(tempdir, "tmp.trk")
    trk.save(merge_tractogram_file)

    return merge_tractogram_file


def convert_trk_fibers_to_tck(dwi, trk_tractograms, tck_tractogram,
                              tempdir=None):
    """
    Convert a list of TRK tractograms to a TCK tractogram (MRtrix format).
    The input tractogram is assumed to be in LAS convention.

    Parameters
    ----------
    dwi: str
        Path to dwi (or nodif_brain) to specify diffusion space.
    trk_tractogram: list of str
        Paths to the input TRK tractograms.
    tck_tractogram: str
        Path to the output TCK tractogram.
    tempdir: str, default None
        A temporary directory to store intermediate tractogram.
    """
    # Local import
    import tractconverter

    # Check existence of input file
    if not os.path.isfile(dwi):
        raise ValueError("File does not exist: {0}.".format(dwi))

    # Merge the input tractograms
    tmp_trk_tractogram = merge_fibers(trk_tractograms, tempdir=tempdir)

    # Convert TRK to TCK using tractconverter
    trk_fibers = tractconverter.TRK(tmp_trk_tractogram)
    tck_fibers = tractconverter.TCK.create(tck_tractogram, hdr=trk_fibers.hdr,
                                           anatFile=dwi)
    tractconverter.convert(trk_fibers, tck_fibers)

    # Clean temporary directory
    shutil.rmtree(os.path.dirname(tmp_trk_tractogram))

    return tck_tractogram


def convert_mitk_vtk_fibers_to_tck(vtk_tractogram, tck_tractogram):
    """
    Convert a .fib tractogram (VTK polydata format) generated by MITK to
    a .tck (MRtrix TCK format) tractogram.
    The input tractogram is assumed to be in LPS convention.

    Parameters
    ----------
    vtk_tractogram: str
        Path to the input .fib tractogram (VTK polydata format).
    tck_tractogram: str
        Path to the output TCK tractogram.
    """
    # Function import
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    # Check existence of input file
    if not os.path.isfile(vtk_tractogram):
        raise ValueError("File does not exist: %s" % vtk_tractogram)

    # Read the input data with VTK
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_tractogram)
    reader.Update()
    polydata = reader.GetOutput()

    # List of fibers = list of point arrays
    fibers = []
    nb_fibers = polydata.GetNumberOfCells()

    # For each fiber
    for i in range(nb_fibers):
        # Get the points of the fiber
        vtk_pts = polydata.GetCell(i).GetPoints().GetData()
        # Convert the coordinates into a numpy array
        fiber_pts = vtk_to_numpy(vtk_pts)
        # Create a copy of the array (vtk seems to keep a pointer and
        # overrides the previous result at the next call)
        fiber_pts_copy = numpy.array(fiber_pts)
        fibers.append(fiber_pts_copy)

    # LPS to RAS matrix: MITK (like ITK) uses the LPS convention.
    # TCK requires RAS
    lps_to_ras = numpy.diag([-1, -1, 1, 1])

    # Create a Nibabel tractogram
    tractogram = nibabel.streamlines.Tractogram(streamlines=fibers,
                                                affine_to_rasmm=lps_to_ras)

    # Create the TCK file with Nibabel
    nibabel.streamlines.TckFile(tractogram=tractogram).save(tck_tractogram)

    return tck_tractogram


def convert_probtrackx2_saved_paths_to_tck(dwi, saved_paths, tck_tractogram,
                                           tempdir=None, verbose=1):
    """
    Convert a TRK tractogram generated by Connectmist to a TCK tractogram
    (MRtrix format).
    The input tractogram is assumed to be in LAS convention.

    Parameters
    ----------
    dwi: str
        Path to dwi (or nodif_brain) to specify diffusion space (affine).
    saved_paths str
        Path to the 'saved_paths.txt' created by Probtrackx2 when requesting
        the fibers with the --savepaths option.
    tck_tractogram: str
        Path to the output TCK tractogram.
    verbose: int, default 1
        Control the verbosity level.
    """
    # Check existence of input files
    for path in [dwi] + saved_paths:
        if not os.path.isfile(path):
            raise ValueError("File does not exist: %s" % path)

    # Create a temporary directory to store an intermediate tractogram
    tempdir = tempfile.mkdtemp(prefix="tractconverter_", dir=tempdir)

    # Concatenate saved paths
    tractogram_file = os.path.join(tempdir, "tmp.txt")
    with open(tractogram_file, "wt") as tract_file:
        for path in saved_paths:
            if verbose > 0:
                print("[info] Concatenating '{0}'...".format(path))
            with open(path, "rt") as open_file:
                for line in open_file:
                    tract_file.write(line)

    # Fibers: each fiber is stored as a Nx3 numpy array (N-point coordinates)
    fibers = []
    with open(tractogram_file) as f:
        for cnt, line in enumerate(f):
            if verbose > 0 and (cnt % 1000000 == 0):
                print("[info] Converting chunk starting at index "
                      "{0}...".format(cnt))
            # '#' indicates the start of a fiber's list of points
            if line.startswith("#"):
                nb_fibers = int(line.strip().split()[1])
                if nb_fibers == 0:
                    continue
                fiber_points = numpy.zeros((nb_fibers, 3), dtype=float)
                fibers.append(fiber_points)
                index = 0  # row index in fiber_points
            else:
                fiber_points[index, :] = map(float, line.strip().split())
                index += 1

    # When a fiber is constructed from a constitutive point (not from an
    # endpoint) there are 2 parts (not always, sometimes one part is
    # discarded). In that case the seed point appears twice in the fiber's
    # list of points. For each fiber:
    for i, fiber in enumerate(fibers):
        if verbose > 0 and (i % 100000 == 0):
            print("[info] Postprocessing fibers starting at index "
                  "{0}...".format(i))
        seed_point = fiber[0, :]  # Coordinates of seed_point
        # Compare the seed coordinates to all other points to check whether
        # it appears again. If its the case inverse the order of points of
        # the 2nd part and put them at the beginning.
        for j, point in enumerate(fiber[1:], start=1):
            if all(point == seed_point):  # if seed point appears again
                fibers[i] = numpy.concatenate((fiber[:j:-1], fiber[:j]),
                                              axis=0)
                break

    # Read the affine matrix from the DWI reference
    affine = nibabel.load(dwi).affine
    # Create a Nibabel tractogram by combining the fibers and the affine
    tractogram = nibabel.streamlines.Tractogram(fibers, affine_to_rasmm=affine)
    # Save the result as a TCK file
    tck = nibabel.streamlines.TckFile(tractogram=tractogram)
    tck.save(tck_tractogram)

    # Clean temporary directory
    shutil.rmtree(tempdir)

    return tck_tractogram


def mrtrix_extract_b0s_and_mean_b0(dwi, b0s, mean_b0, bvals=None, bvecs=None,
                                   nb_threads=1):
    """ Extract b=0 (bvalue=0) volumes from DWI and compute mean b=0 volume.

    Parameters
    ----------
    dwi: str
        The diffusion file.
    b0s: str
        The b0 volumes file.
    mean_b0:
        The mean b0 volumes file.
    bvals, bvecs: str
        Path to bvals/bvecs, required if dwi is not a MIF file.
    nb_threads: int, default None
        Number of threads that MRtrix is allowed to use.
    """
    # Check arguments
    is_mif = dwi.endswith(".mif")
    if (not is_mif) and (None in {bvals, bvecs}):
        raise ValueError("bvals/bvecs required if DWI is not in MIF format.")

    # Extract the b0 volumes
    cmd_1 = ["dwiextract", "-bzero", dwi, b0s,
             "-nthreads", "%i" % nb_threads, "-failonwarn"]
    if not is_mif:
        cmd_1 += ["-fslgrad", bvecs, bvals]
    subprocess.check_call(cmd_1)

    # Average the b0 volumes
    cmd_2 = ["mrmath", b0s, "mean", mean_b0, "-axis", "3",
             "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_2)


def extract_image(in_file, index, out_file=None):
    """ Extract the image at 'index' position.

    Parameters
    ----------
    in_file: str (mandatory)
        the input image.
    index: int (mandatory)
        the index of last image dimention to extract.
    out_file: str (optional, default None)
        the name of the extracted image file.

    Returns
    -------
    out_file: str
        the name of the extracted image file.
    """
    # Set default output if necessary
    dirname = os.path.dirname(in_file)
    basename = os.path.basename(in_file).split(".")[0]
    if out_file is None:
        out_file = os.path.join(
            dirname, "extract{0}_{1}.nii.gz".format(index, basename))

    # Extract the image of interest
    image = nibabel.load(in_file)
    affine = image.get_affine()
    extracted_array = image.get_data()[..., index]
    extracted_image = nibabel.Nifti1Image(extracted_array, affine)
    nibabel.save(extracted_image, out_file)

    return out_file


def fslreorient2std(input_image, output_image, save_trf=True,
                    fslconfig=DEFAULT_FSL_PATH):
    """ Reorient an image to match the approximate orientation of the standard
    template image (MNI152).

    It only applies 0, 90, 180 or 270 degree rotations.
    It is not a registration tool.
    It requires NIfTI images with valid orientation information in them (seen
    by valid labels in FSLView). This tool assumes the labels are correct - if
    not, fix that before using this. If the output name is not specified the
    equivalent transformation matrix is written to the standard output.

    The basic usage is:
        fslreorient2std <input_image> [output_image]

    Parameters
    ----------
    input_image: str (mandatory)
        The image to reorient.
    output_image: str (mandatory)
        The reoriented image.
    save_trf: bool, default True
        If set save the reorientation matrix.
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.
    """
    # Check the input parameter
    if not os.path.isfile(input_image):
        raise ValueError("'{0}' is not a valid input file.".format(
                         input_image))
    if not output_image.endswith(".nii.gz"):
        if output_image.endswith(".nii"):
            output_image += ".gz"
        else:
            output_image += ".nii.gz"

    # Define the FSL commands
    cmd1 = ["fslreorient2std", input_image, output_image]
    cmd2 = ["fslreorient2std", input_image]

    # Call fslreorient2std
    fslprocess = FSLWrapper(shfile=fslconfig)
    fslprocess(cmd=cmd1)
    if save_trf:
        fslprocess(cmd=cmd2)
        fsl_trf_file = output_image.split(".")[0] + ".fsl.trf"
        with open(fsl_trf_file, "wt") as open_file:
            open_file.write(fslprocess.stdout.decode("utf-8"))
        trf_file = output_image.split(".")[0] + ".trf"
        numpy.savetxt(trf_file, flirt2aff(fsl_trf_file, output_image,
                                          input_image))

    return output_image


def surf2surf(input_surf, output_surf, fslconfig=DEFAULT_FSL_PATH):
    """ Convert an input surface in ASCI mode.

    Parameters
    ----------
    input_surf: str (mandatory)
        The input surface to convert.
    output_surf: str (mandatory)
        The converted surface.
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.
    """
    # check the input parameter
    if not os.path.isfile(input_surf):
        raise ValueError("'{0}' is not a valid input file.".format(
                         input_surf))

    # Define the FSL command
    cmd = [
        "surf2surf",
        "-i", input_surf,
        "-o", output_surf,
        "--outputtype=ASCII",
        "--values=1"]

    # Call fslreorient2std
    fslprocess = FSLWrapper(cmd, shfile=fslconfig)
    fslprocess(cmd=cmd)


def apply_mask(input_file, output_fileroot, mask_file,
               fslconfig=DEFAULT_FSL_PATH):
    """ Apply a mask to an image.

    Parameters
    ----------
    input_file: str (mandatory)
        The image to mask.
    output_fileroot: str (mandatory)
        The masked image root name.
    mask_file: str (mandatory)
        The mask image.
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    mask_file: str
        the masked input image.
    """
    # Check the input parameter
    for filename in (input_file, mask_file):
        if not os.path.isfile(filename):
            raise ValueError("'{0}' is not a valid input "
                             "file.".format(filename))

    # Define the FSL command
    # "-mas": use (following image>0) to mask current image.
    cmd = ["fslmaths", input_file, "-mas", mask_file, output_fileroot]

    # Call fslmaths
    fslprocess = FSLWrapper(shfile=fslconfig)
    fslprocess(cmd=cmd)

    return glob.glob(output_fileroot + ".*")[0]


def erode(input_file, output_file, radius, fslconfig=DEFAULT_FSL_PATH):
    """ Erode an image using a spherical kernel.

    Parameters
    ----------
    input_file: str (mandatory)
        The image to erode (binary or gray level).
    output_file: str (mandatory)
        The eroded image.
    radius: float (optional, default 2)
        The sphere kernel in mm.
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    output_file: str
        The eroded image.
    """
    # Check the input parameter
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(input_file))

    # Define the FSL command
    cmd = ["fslmaths", input_file, "-kernel", "sphere", str(radius), "-ero",
           output_file]

    # Call fslmaths
    fslprocess = FSLWrapper(shfile=fslconfig)
    fslprocess(cmd=cmd)

    return output_file


def monkeypatch(klass, methodname=None):
    """ Decorator extending class with the decorated callable.

    >>> class A:
    ...     pass
    >>> @monkeypatch(A)
    ... def meth(self):
    ...     return 12
    ...
    >>> a = A()
    >>> a.meth()
    12
    >>> @monkeypatch(A, 'foo')
    ... def meth(self):
    ...     return 12
    ...
    >>> a.foo()
    12

    Parameters
    ----------
    klass: class object
        the class to be decorated.
    methodname: str, default None
        the name of the decorated method. If None, use the function name.

    Returns
    -------
    decorator: callable
        the decorator.
    """
    def decorator(func):
        try:
            name = methodname or func.__name__
        except AttributeError:
            raise AttributeError(
                "{0} has no __name__ attribute: you should provide an "
                "explicit 'methodname'".format(func))
        setattr(klass, name, func)
        return func
    return decorator


class TempDir(object):
    """ Create a tempdir with the with synthax.
    """
    def __init__(self, dirname=None, basename=None):
        """ Initialize the TempDir class.

        Parameters
        ----------
        dirname: str, default None
            if set, the temporary directory is generated in this folder and
            will not be removed.
        basename: str, default None
            if set, use this name as a temporary folder prefix.
        """
        self.path = None
        self.dirname = dirname
        self.basename = basename
        return

    def __enter__(self):
        kwargs = {}
        if self.dirname is not None:
            kwargs["dir"] = self.dirname
        if self.basename is not None:
            kwargs["prefix"] = self.basename
        self.path = tempfile.mkdtemp(**kwargs)
        return self.path

    def __exit__(self, type, value, traceback):
        if self.dirname is None:
            shutil.rmtree(self.path)
