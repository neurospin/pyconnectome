##########################################################################
# NSAP - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wrappers for the FSL's segmentation utilities.
"""

# System import
import os
import glob

# Pyfsl import
from pyfsl import DEFAULT_FSL_PATH
from pyfsl.wrapper import FSLWrapper


def fast(input_file, out_fileroot, klass=3, im_type=1, segments=False,
         bias_field=True, bias_corrected_im=True, probabilities=False,
         shfile=DEFAULT_FSL_PATH):
    """ FAST (FMRIB's Automated Segmentation Tool) segments a 3D image of
    the brain into different tissue types (Grey Matter, White Matter, CSF,
    etc.), whilst also correcting for spatial intensity variations
    (also known as bias field or RF inhomogeneities).
    The underlying method is based on a hidden Markov random field model and
    an associated Expectation-Maximization algorithm.

    Usage: fast [options] file(s)

    Parameters
    ----------
    input_file: str (mandatory)
        the image to be segmented.
    out: str (mandatory)
        output basename.
    klass: int (optional, default 3)
        number of tissue-type classes.
    im_type: int (optional, default 1)
        type of image 1=T1, 2=T2, 3=PD.
    segments: bool (optional, default False)
        outputs a separate binary image for each tissue type.
    bias_field: bool (optional, default True)
        output estimated bias field.
    bias_corrected_im: bool (optional, default True)
        output bias-corrected image.
    probabilities: bool (optional, default False)
        outputs individual probability maps.
    shfile: str (optional, default local path)
        the path to the FSL 'fsl.sh' configuration file.

    Returns
    -------
    tpm: list of str
        the generated tissue preobaility maps.
    tsm: list of str
        the generated tissue segmentation maps.
    segmentation_file: str
        the segmented tissues.
    bias_file: str
        the bias field.
    biascorrected_file: str
        the bias corrected input image.
    """
    # Check input parameters
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(input_file))

    # Define the FSL command
    bool_params = {
        "-g": segments,
        "-b": bias_field,
        "-B": bias_corrected_im,
        "-p": probabilities
    }
    cmd = ["fast", "-o", out_fileroot, "-n", str(klass), "-t", str(im_type)]
    for name, value in bool_params.items():
        if value:
            cmd.append(name)
    cmd.append(input_file)

    # Call FSL fast
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not declared in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Format outputs
    image_ext = fslprocess.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]
    segmentation_file = out_fileroot + "_seg" + image_ext
    bias_file = out_fileroot + "_bias" + image_ext
    if not os.path.isfile(bias_file):
        bias_file = None
    biascorrected_file = out_fileroot + "_restore" + image_ext
    if not os.path.isfile(biascorrected_file):
        biascorrected_file = None
    tpm = glob.glob(out_fileroot + "_pve_*")
    tsm = glob.glob(out_fileroot + "_pve_*")

    return tpm, tsm, segmentation_file, bias_file, biascorrected_file


def bet2(input_file, output_fileroot, outline=False, mask=False,
         skull=False, nooutput=False, f=0.5, g=0, radius=None, smooth=None,
         c=None, threshold=False, mesh=False, shfile=DEFAULT_FSL_PATH):
    """ Wraps command bet2.

    Deletes non-brain tissue from an image of the whole head. It can also
    estimate the inner and outer skull surfaces, and outer scalp surface,
    if you have good quality T1 and T2 input images.

    The basic usage is:
        bet2 <input_fileroot> <output_fileroot> [options]

    Parameters
    ----------
    input_file: (mandatory)
        Input image.
    output_fileroot: (mandatory)
        Output image.
    outline: bool (optional, default False)
        Generate brain surface outline overlaid onto original image.
    mask: bool (optional, default False)
        Generate binary brain mask.
    skull: bool (optional, default False)
        Generate approximate skull image.
        (not as clean as what betsurf generates).
    nooutput: bool (optional, default False)
        Don't generate segmented brain image output.
    f: float (optional, default 0.5)
        Fractional intensity threshold (0->1).
        Smaller values give larger brain outline estimates.
    g: int (optional, default 0)
        Vertical gradient in fractional intensity threshold (-1->1).
        Positive values give larger brain outline at bottom, smaller at top.
    radius: (optional)
        Head radius (mm not voxels).
        Initial surface sphere is set to half of this.
    smooth : float (optional, default 1)
        Smoothness factor.
        Values smaller than 1 produce more detailed brain surface, values
        larger than one produce smoother, less detailed surface.
    c: (optional)
        Centre-of-gravity (voxels not mm) of initial mesh surface (x, y, z).
    threshold: bool (optional, default False)
        Apply thresholding to segmented brain image and mask.
    mesh: bool (optional, default False).
        Generates brain surface as mesh in .vtk format.
    shfile: (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.


    Returns
    -------
    output: str
        the extracted brain volume.
    mask_file: str
        the binary mask of the extracted brain volume.
    mesh_file: str
        the brain surface as a vtk mesh.
    outline_file: str
        the brain surface outline overlaid onto original image.
    inskull_mask_file, inskull_mesh_file,
    outskull_mask_file, outskull_mesh_file,
    outskin_mask_file, outskin_mesh_file,
    skull_mask_file: str
        rough skull image.
    shfile: str (optional, default local path)
        the path to the FSL 'fsl.sh' configuration file.
    """
    # Check the input parameter
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(
                         input_file))

    # Check that the output directory exists
    if not os.path.isdir(output_fileroot):
        os.mkdir(output_fileroot)

    # Define the FSL command
    cmd = ["bet2",
           input_file,
           output_fileroot,
           "-f", str(f),
           "-g", str(g),
           ]

    # Set optional arguments
    if outline:
        cmd += ["--outline"]
    if mask:
        cmd += ["--mask"]
    if skull:
        cmd += ["--skull"]
    if nooutput:
        cmd += ["--nooutput"]
    if mesh:
        cmd += ["--mesh"]
    if threshold:
        cmd += ["--threshold"]

    if c is not None:
        cmd += ["-c", c]
    if radius is not None:
        cmd += ["--radius", radius]
    if smooth is not None:
        cmd += ["--smooth", smooth]

    # Call bet2
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not declared in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Format outputs
    image_ext = fslprocess.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]

    # Create format outputs
    if outline:
        outline_file = output_fileroot + "_outline" + image_ext
    else:
        outline_file = None

    if mask:
        mask_file = output_fileroot + "_mask" + image_ext
    else:
        mask_file = None

    if skull:
        inskull_mask_file = output_fileroot + "_inskull_mask" + image_ext
        inskull_mesh_file = output_fileroot + "_inskull_mesh" + image_ext
        outskull_mask_file = output_fileroot + "_outskull_mask" + image_ext
        outskull_mesh_file = output_fileroot + "_outskull_mesh" + image_ext
        outskin_mask_file = output_fileroot + "_outskin_mask" + image_ext
        outskin_mesh_file = output_fileroot + "_outskin_mesh" + image_ext
        skull_mask_file = output_fileroot + "_skull_mask" + image_ext
    else:
        inskull_mask_file = None
        inskull_mesh_file = None
        outskull_mask_file = None
        outskull_mesh_file = None
        outskin_mask_file = None
        outskin_mesh_file = None
        skull_mask_file = None

    if nooutput:
        output = None
    else:
        output_fileroot += image_ext
        output = output_fileroot

    if mesh:
        mesh_file = output_fileroot + "_mesh.vtk"
    else:
        mesh_file = None

    return (output, mask_file, mesh_file, outline_file, inskull_mask_file,
            inskull_mesh_file, outskull_mask_file, outskull_mesh_file,
            outskin_mask_file, outskin_mesh_file, skull_mask_file)
