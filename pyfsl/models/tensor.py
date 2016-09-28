##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Use FSL dtifit command for fitting a diffusion tensor model at each voxel.
"""

# System import
import os

# Pyfsl import
from pyfsl import DEFAULT_FSL_PATH
from pyfsl.wrapper import FSLWrapper


def dtifit(data, bvecs, bvals, mask, out, wls=False, save_tensor=False,
           fslconfig=DEFAULT_FSL_PATH):
    """ Fit a diffusion tensor model at each voxel in the mask.

    Binding around the FSL's 'dtifit' command.

    The basic usage is:
        dtifit --data <filename>
        dtifit --verbose

    Parameters
    ----------
    data : str (mandatory)
        Diffusion weighted image data file.
        A 4D series of data volumes.
    mask: str (mandatory)
        Brain binary mask file (i.e. from BET).
        A single binarized volume in diffusion space containing ones inside the
        brain and zeros outside the brain.
    out : str (mandatory)
        User specifies a basename that will be used to name the outputs of
        dtifit.
    bvecs : str (mandatory)
        b vectors file.
        Gradient directions.
        An ASCII text file containing a list of gradient directions applied
        during diffusion weighted volumes. The order of entries in this file
        must match the order of volumes in the input data series.
    bvals : str (mandatory)
        v values file.
        An ASCII text file containing a list of b values applied during each
        volume acquisition. The order of entries in this file must match the
        order of volumes in the input data and entries in the gradient
        directions text file.
    wls : bool (optional, default False)
        Fit the tensor with weighted least squares.
    save_tensor: bool (optional, default False)
        Save the elements of the tensor.

    Returns
    -------
    v1_file: str
        path/name of file with the 1st eigenvector.
    v2_file: str
        path/name of file with the 2nd eigenvector.
    v3_file: str
        path/name of file with the 3rd eigenvector.
    l1_file: str
        path/name of file with the 1st eigenvalue.
    l2_file: str
       path/name of file with the  2nd eigenvalue.
    l3_file: str
        path/name of file with the 3rd eigenvalue.
    md_file: str
        path/name of file with the mean diffusivity.
    fa_file: str
        path/name of file with the Fractional anisotropy.
    s0_file: str
        path/name of file with the Raw T2 signal with no diffusion weighting.
    tensor_file: str
        path/name of file with the 4D tensor volume.
    m0_file: str
        path/name of file with the mode of the anisotropy.
    """
    # Check input parameters
    for filename in (data, bvals, bvecs, mask):
        if not os.path.isfile(filename):
            raise ValueError("'{0}' is not a valid input file.".format(
                filename))

    # Check that the output directory exists
    if not os.path.isdir(out):
        os.mkdir(out)
    out = os.path.join(out, "dtifit")

    # Define the FSL command
    cmd = ["dtifit",
           "-k", data,
           "-r", bvecs,
           "-b", bvals,
           "-m", mask,
           "-o", out]

    # Add optional arguments
    if wls:
        cmd += ["--wls"]
    if save_tensor:
        cmd += ["--save_tensor"]

    # Execute the FSL command
    fslprocess = FSLWrapper(cmd, shfile=fslconfig)
    fslprocess()

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not decalred in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Build the output names
    image_ext = FSLWrapper.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]
    v1_file = out + "_V1" + image_ext
    v2_file = out + "_V2" + image_ext
    v3_file = out + "_V3" + image_ext
    l1_file = out + "_L1" + image_ext
    l2_file = out + "_L2" + image_ext
    l3_file = out + "_L3" + image_ext
    md_file = out + "_MD" + image_ext
    fa_file = out + "_FA" + image_ext
    s0_file = out + "_S0" + image_ext
    tensor_file = out + "_tensor" + image_ext
    m0_file = out + "_M0" + image_ext

    return (v1_file, v2_file, v3_file, l1_file, l2_file, l3_file, md_file,
            fa_file, s0_file, tensor_file, m0_file)
