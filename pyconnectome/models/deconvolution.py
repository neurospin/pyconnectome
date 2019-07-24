##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Estimating fiber orientations using model-based deconvolution.
"""

# System import
import os
import glob

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper


def bedpostx(subjectdir, n=3, w=1, b=1000, j=1250, s=25, model=2,
             g=False, c=False, rician=False, fslconfig=DEFAULT_FSL_PATH,
             fsl_parallel=False):
    """ Wraps command bedpostx.

    The basic usage is:
        bedpostx <subject_directory> [options]

    expects to find bvals and bvecs in subject directory
    expects to find data and nodif_brain_mask in subject directory
    expects to find grad_dev in subject directory, if g is set

    ALTERNATIVELY: you can pass on xfibres options onto directly bedpostx
    For example:  bedpostx <subject directory> --noard --cnonlinear
    Type 'xfibres --help' for a list of available options
    Default options will be bedpostx default (see above), and not xfibres
    default.

    Parameters
    ----------
    subjectdir: str (mandatory)
        The subject directory with the diffusion data.
    n: int (mandatory, default 3)
        Number of fibers per voxel.
    w: int (mandatory, default 1)
        ARD weight, more weight means less secondary fibers per voxel.
    b: int (mandatory, default 1000)
        Burnin period: number of iterations before starting the sampling.
        These might be increased if the data are noisy, and the MCMC needs more
        iterations to converge.
    j: int (mandatory, default 1250)
        Number of jumps to be made by MCMC.
    s: int (mandatory, default 25)
        Sample every.
    model: int (mandatory, default 2)
        Deconvolution model.
        1: single-shell, with sticks,
        2: multi-shell, with sticks with a range of diffusivities,
        3: multi-shell, with zeppelins.
    g: bool (optional, default False)
        Consider gradient nonlinearities. Instructs bedpostx to use the
        grad_dev.nii.gz file from the data folder and produce voxel-specific
        bvals and bvecs.
    c: (optional, default False)
        Do not use CUDA capable hardware/queue (if found).
    rician: bool (optional, default False)
        A Rician noise model.
    fslconfig: str (optional)
        The FSL configuration batch.
    fsl_parallel: (optional)
        If set use Condor to parallelize FSL on your local workstation.

    Returns
    -------
    outdir: str
        The bedpostx output directory
    merged_th<i>samples - 4D volume
        Samples from the distribution on theta
    merged_ph<i>samples - 4D volume
        Samples from the distribution on phi: theta and phi together represent
        the principal diffusion direction in spherical polar co-ordinates
    merged_f<i>samples - 4D volume
        Samples from the distribution on anisotropic volume fraction.
    mean_th<i>samples - 3D Volume
        Mean of distribution on theta
    mean_ph<i>samples - 3D Volume
        Mean of distribution on phi
    mean_f<i>samples - 3D Volume
        Mean of distribution on f anisotropy. Note that in each voxel, fibers
        are ordered according to a decreasing mean f-value
    mean_dsamples - 3D Volume
        Mean of distribution on diffusivity d
    mean_S0samples - 3D Volume
        Mean of distribution on T2w baseline signal intensity S0
    dyads<i>
        Mean of PDD distribution in vector form. Note that this file can be
        loaded into fslview for easy viewing of diffusion directions
    dyads_dispersion - 3D Volume
        Uncertainty on the estimated fiber orientation. Characterizes how wide
        the orientation distribution is around the respective PDD.
    nodif_brain_mask
        binary mask created from nodif_brain - copied from input directory
    bvecs
        contain a 3x1 vector for each gradient, indicating the gradient
        direction - copied from input directory
    bvals
        contain a scalar value for each applied gradient, corresponding to the
        respective bvalue - copied from input directory
    """
    # Check input parameters
    if not os.path.isdir(subjectdir):
        raise ValueError(
            "'{0}' is not a valid subject directory.".format(
                subjectdir))

    # Define the FSL command
    cmd = ["bedpostx",
           subjectdir,
           "-n", str(n),
           "-w", str(w),
           "-b", str(b),
           "-j", str(j),
           "-s", str(s),
           "-model", str(model)]

    # Add optional arguments
    if rician:
        cmd += ["--rician"]
    if g:
        cmd += ["-g"]
    if c:
        cmd += ["-c"]

    # Execute the FSL command
    fslprocess = FSLWrapper(shfile=fslconfig, fsl_parallel=fsl_parallel)
    fslprocess(cmd=cmd)

    # Format outputs
    outdir = subjectdir + ".bedpostX"
    merged_th = glob.glob(os.path.join(outdir, "merged_th*"))
    merged_ph = glob.glob(os.path.join(outdir, "merged_ph*"))
    merged_f = glob.glob(os.path.join(outdir, "merged_f*"))
    mean_th = glob.glob(os.path.join(outdir, "mean_th*"))
    mean_ph = glob.glob(os.path.join(outdir, "mean_ph*"))
    mean_f = glob.glob(os.path.join(outdir, "mean_f*"))
    mean_d = os.path.join(outdir, "mean_d*")
    mean_S0 = os.path.join(outdir, "mean_S0*")
    dyads = glob.glob(os.path.join(outdir, "dyads*"))

    return (outdir, merged_th, merged_ph, merged_f, mean_th, mean_ph, mean_f,
            mean_d, mean_S0, dyads)


def bedpostx_datacheck(data_dir, fslconfig=DEFAULT_FSL_PATH):
    """ Wraps bedpostx_datacheck.

    Usage: bedpostx_datacheck data_dir

    Parameters
    ----------
    data_dir: str (mandatory)
        The folder to check.
    fslconfig: str (optional)
        The FSL configuration batch.

    Returns
    -------
    is_valid: bool
        True if all the data are present in the data directory
    """
    # check data directory
    if not os.path.isdir(data_dir):
        raise ValueError("'{0}' is not a valid data directory.".format(
                         data_dir))

    # Define the FSL command
    cmd = ["bedpostx_datacheck", data_dir]

    # Execute the FSL command
    fslprocess = FSLWrapper(shfile=fslconfig)
    fslprocess(cmd=cmd)

    # Parse outputs
    is_valid = (
        fslprocess.stderr == "" and "does not exist" not in fslprocess.stdout)

    return is_valid
