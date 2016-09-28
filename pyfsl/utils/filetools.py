##########################################################################
# NSAP - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wrappers for the FSL's file utilities.
"""

# System import
import os
import nibabel

# Pyfsl import
from pyfsl import DEFAULT_FSL_PATH
from pyfsl.wrapper import FSLWrapper


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


def fslreorient2std(input_image, output_image, fslconfig=DEFAULT_FSL_PATH):
    """ Reorient an image to match the approximate orientation of the standard
    template images (MNI152).

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
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.
    """
    # check the input parameter
    if not os.path.isfile(input_image):
        raise ValueError("'{0}' is not a valid input file.".format(
                         input_image))

    # Define the FSL command
    cmd = ["fslreorient2std", input_image, output_image]

    # Call fslreorient2std
    fslprocess = FSLWrapper(cmd, shfile=fslconfig)
    fslprocess()


def apply_mask(input_image, output_image, mask_image,
               fslconfig=DEFAULT_FSL_PATH):
    """ Apply a mask to an image.

    Parameters
    ----------
    input_image: str (mandatory)
        The image to mask.
    output_image: str (mandatory)
        The computed masked image.
    mask_image: str (mandatory)
        The mask image.
    fslconfig: str (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.
    """
    # check the input parameter
    for filename in (input_image, mask_image):
        if not os.path.isfile(filename):
            raise ValueError("'{0}' is not a valid input "
                             "file.".format(filename))

    # Define the FSL command
    # "-mas": use (following image>0) to mask current image.
    cmd = ["fslmaths", input_image, "-mas", mask_image, output_image]

    # Call fslmaths
    fslprocess = FSLWrapper(cmd, shfile=fslconfig)
    fslprocess()
