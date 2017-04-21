##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Registration utilities.
"""

# System import
import os
import glob
import numpy
import nibabel

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper


def mcflirt(in_file, out_fileroot, cost="normcorr", bins=256, dof=6,
            reg_to_mean=True, mats=False, plots=True, verbose=0,
            shfile=DEFAULT_FSL_PATH):
    """ Wraps command mcflirt.

    MCFLIRT is an intra-modal motion correction tool designed for use on
    fMRI time series and based on optimization and registration techniques
    used in FLIRT, a fully automated robust and accurate tool for linear
    (affine) inter- and inter-modal brain image registration.

    Parameters
    ----------
    in_file: str (mandatory)
        Input serie file path.
    out_fileroot: str (mandatory)
        Output serie file path without extension.
    cost: str(optional, default None)
        The optimization cost function.
        Choose the most appropriate option: "mutualinfo", "woods",
        "corratio", "normcorr", "normmi", "leastsquares".
    bins: int (optional, default 256)
        Number of histogram bins.
    dof: int (optional, default 6)
        Number of transform degrees of freedom.
    reg_to_mean: bool (optional, default True)
        If set, register to mean, otherwise to middle volume of the serie.
    mats: bool (optional, default False)
        If set save transformation matricies in subdirectory outfilename.mat
    plot: bool (optional, default True)
        If set save transformation parameters in file outputfilename.par
    verbose: int (optional)
        0 is least and default.
    shfile: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    func_file: str
        Output realigned serie.
    mean_file: str
        Mean serie tempalte.
    par_file: str
        The motion correction transformation parameters.
    """
    # Check the input parameters
    if not os.path.isfile(in_file):
        raise ValueError(
            "'{0}' is not a valid input file.".format(in_file))
    if cost not in ["mutualinfo", "woods", "corratio", "normcorr", "normmi",
                    "leastsquares"]:
        raise ValueError(
            "'{0}' is not a valid optimization cost function.".format(cost))

    # Define the FSL command
    cmd = ["mcflirt",
           "-in", in_file,
           "-out", out_fileroot,
           "-cost", cost,
           "-bins", str(bins),
           "-dof", str(dof),
           "-verbose", str(verbose)]
    if reg_to_mean:
        cmd.append("-meanvol")
    if mats:
        cmd.append("-mats")
    if plots:
        cmd.append("-plots")

    # Call mcflirt
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    # Get generated outputs
    func_files = [elem for elem in glob.glob(out_fileroot + ".*")
                 if not elem.endswith(".par") and os.path.isfile(elem)]
    if len(func_files) != 1:
        raise ValueError("Expect only one mcflirt output file, not {0}.".format(
            func_files))
    func_file = func_files[0]
    mean_file = None
    if reg_to_mean:
        mean_file = glob.glob(out_fileroot + "_mean_reg.*")[0]
    par_file = None
    if plots:
        par_file = out_fileroot + ".par"

    return func_file, mean_file, par_file


def flirt(in_file, ref_file, omat=None, out=None, init=None, cost="corratio",
          usesqform=False, displayinit=False, anglerep="euler", bins=256,
          interp="trilinear", dof=12, applyxfm=False, applyisoxfm=None,
          verbose=0, shfile=DEFAULT_FSL_PATH):
    """ Wraps command flirt.

    The basic usage is:
        flirt [options] -in <inputvol> -ref <refvol> -out <outputvol>
        flirt [options] -in <inputvol> -ref <refvol> -omat <outputmatrix>
        flirt [options] -in <inputvol> -ref <refvol> -applyxfm -init <matrix>
              -out <outputvol>

    Parameters
    ----------
    in_file: str (mandatory)
        Input volume.
    ref_file: str (mandatory)
        Reference volume.
    omat: str (optional, default None)
        Matrix filename. Output in 4x4 ascii format.
    out: str (optional, default None)
        Output volume.
    init: (optional, default None)
        Input 4x4 affine matrix
    cost: str (optional, default "corratio")
        Choose the most appropriate option: "mutualinfo", "corratio",
        "normcorr", "normmi", "leastsq", "labeldiff", "bbr".
    usesqform: bool (optional, default False)
        Initialise using appropriate sform or qform.
    displayinit: bool
        Display initial matrix.
    anglerep: str (optional default "euler")
        Choose the most appropriate option: "quaternion", "euler".
    bins: int (optional, default 256)
        Number of histogram bins
    interp: str (optional, default "trilinear")
        Choose the most appropriate option: "trilinear", "nearestneighbour",
        "sinc", "spline". (final interpolation: def - trilinear)
    dof: int (optional, default 12)
        Number of transform dofs.
    applyxfm: bool
        Applies transform (no optimisation) - requires -init.
    applyisoxfm: float (optional)
        The integer defines the scale. As applyxfm but forces isotropic
        resampling.
    verbose: int (optional)
        0 is least and default.
    shfile: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    out: str
        Output volume.
    omat: str
        Output matrix filename. Output in 4x4 ascii format.
    """
    # Check the input parameters
    for filename in (in_file, ref_file):
        if not os.path.isfile(filename):
            raise ValueError(
                "'{0}' is not a valid input file.".format(filename))

    # Define the FSL command
    cmd = ["flirt",
           "-in", in_file,
           "-ref", ref_file,
           "-cost", cost,
           "-anglerep", anglerep,
           "-bins", str(bins),
           "-interp", interp,
           "-dof", str(dof),
           "-verbose", str(verbose)]

    # Set default parameters
    if usesqform:
        cmd += ["-usesqform"]
    if displayinit:
        cmd += ["-displayinit"]
    if applyxfm:
        cmd += ["-applyxfm"]

    if init is not None:
        cmd += ["-init", init]
    if applyisoxfm is not None:
        cmd += ["-applyisoxfm", applyisoxfm]

    dirname = os.path.dirname(in_file)
    basename = os.path.basename(in_file).split(".")[0]
    if out is None:
        out = os.path.join(dirname, "flirt_out_{0}.nii.gz".format(basename))
        cmd += ["-out", out]
    else:
        cmd += ["-out", out]

    if omat is None:
        omat = os.path.join(dirname, "flirt_omat_{0}".format(basename))
        cmd += ["-omat", omat]
    else:
        cmd += ["-omat", omat]

    # Call flirt
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    return out, omat


def fnirt(in_file, ref_file, affine_file, outdir, inmask_file=None, verbose=0,
          shfile=DEFAULT_FSL_PATH):
    """ Wraps command fnirt.

    Parameters
    ----------
    in_file: str (mandatory)
        Input volume.
    ref_file: str (mandatory)
        Reference volume.
    affine_file: str (optional, default None)
        Affine matrix filename in 4x4 ascii format.
    outdir: str
        The destination folder.
    inmask_file: str (optional, default None)
        Name of file with mask in input image space.
    verbose: int (optional)
        0 is least and default.
    shfile: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.

    Returns
    -------
    cout: str
        Name of output file with field coefficients.
    iout: str
        Name of output image.
    fout: str
        Name of output file with field.
    jout: str
        Name of file for writing out the Jacobian of the field.
    refout: str
        Name of file for writing out intensity modulated.
    intout: str
        Name of files for writing information pertaining to intensity mapping
    logout: str
        Name of log-file.
    """
    # Check the input parameters
    for filename in (in_file, ref_file, affine_file, inmask_file):
        if filename is not None and not os.path.isfile(filename):
            raise ValueError(
                "'{0}' is not a valid input file.".format(filename))

    # Define the FSL command
    cmd = ["fnirt",
           "--ref={0}".format(ref_file),
           "--in={0}".format(in_file),
           "--aff={0}".format(affine_file),
           "--verbose={0}".format(verbose)]
    if inmask_file is not None:
        cmd += ["--inmask={0}".format(inmask_file)]
    basename = os.path.basename(in_file).split(".")[0]
    outputs = []
    for param in ("cout", "iout", "fout", "jout", "refout", "intout", "logout"):
        ext = ".nii.gz"
        if param in ("logout"):
            ext = ".txt"
        outputs.append(
            os.path.join(outdir, "{0}_{1}{2}".format(param, basename, ext)))
        cmd += ["--{0}={1}".format(param, outputs[-1])]

    # Call fnirt
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    return outputs


def flirt2aff(mat_file, in_file, ref_file):
    """ Map from 'in_file' image voxels to 'ref_file' voxels given `omat` FSL
    affine transformation.

    Parameters
    ------------
    mat_file: str (mandatory)
        filename of output '-omat' transformation file from FSL flirt.
    in_file: str (mandatory)
        filename of the image passed to flirt as the '-in' image.
    ref_file: str (mandatory)
        filename of the image passed to flirt as the '-ref' image.

    Returns
    -------
    omat: array (4, 4)
        array containing the transform from voxel coordinates in image
        for 'in_file' to voxel coordinates in image for 'ref_file'.
    """
    # Check the input parameters
    for filename in (mat_file, in_file, ref_file):
        if not os.path.isfile(filename):
            raise ValueError("'{0}' is not a valid input "
                             "file.".format(filename))

    # Load dataset
    flirt_affine = numpy.loadtxt(mat_file)
    in_img = nibabel.load(in_file)
    ref_img = nibabel.load(ref_file)
    in_hdr = in_img.get_header()
    ref_hdr = ref_img.get_header()

    # Define a function to flip x
    def _x_flipper(n):
        flipr = numpy.diag([-1, 1, 1, 1])
        flipr[0, 3] = n - 1
        return flipr

    # Check image orientation
    inspace = numpy.diag(in_hdr.get_zooms() + (1, ))
    refspace = numpy.diag(ref_hdr.get_zooms() + (1, ))
    if numpy.linalg.det(in_img.get_affine()) >= 0:
        inspace = numpy.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
    if numpy.linalg.det(ref_img.get_affine()) >= 0:
        refspace = numpy.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))

    # Get the voxel to voxel mapping
    omat = numpy.dot(
        numpy.linalg.inv(refspace), numpy.dot(flirt_affine, inspace))

    return omat
