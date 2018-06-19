##########################################################################
# NSAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
dMRI preprocessing tools.
"""

# System import
import os
import glob
import json
import shutil

# Third party
import numpy
import nibabel

# Package import
from pyconnectome.wrapper import FSLWrapper
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectomist.utils.dwitools import read_bvals_bvecs


def epi_reg(
        epi_file, structural_file, brain_structural_file, output_fileroot,
        fieldmap_file=None, effective_echo_spacing=None, magnitude_file=None,
        brain_magnitude_file=None, phase_encode_dir=None, wmseg_file=None,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Register EPI images (typically functional or diffusion) to structural
    (e.g. T1-weighted) images. The pre-requisites to use this method are:

    1) a structural image that can be segmented to give a good white matter
    boundary.
    2) an EPI that contains some intensity contrast between white matter and
    grey matter (though it does not have to be enough to get a segmentation).

    It is also capable of using fieldmaps to perform simultaneous registration
    and EPI distortion-correction. The fieldmap must be in rad/s format.

    Parameters
    ----------
    epi_file: str
        The EPI images.
    structural_file: str
        The structural image.
    brain_structural_file
        The brain extracted structural image.
    output_fileroot: str
        The corrected EPI file root.
    fieldmap_file: str, default None
        The fieldmap image (in rad/s)
    effective_echo_spacing: float, default None
        If parallel acceleration is used in the EPI acquisition then the
        effective echo spacing is the actual echo spacing between acquired
        lines in k-space divided by the acceleration factor.
    magnitude_file: str, default None
        The magnitude image.
    brain_magnitude_file: str
        The brain extracted magnitude image: should only contains brain
        tissues.
    phase_encode_dir: str, default None
         The phase encoding direction x/y/z/-x/-y/-z.
    wmseg_file: str, default None
        The white matter segmentatiion of structural image. If provided do not
        execute FAST.
    fsl_sh: str, default DEFAULT_FSL_PATH
        The FSL configuration batch.

    Returns
    -------
    corrected_epi_file: str
        The corrected EPI image.
    warp_file: str
        The deformation field (in mm).
    distortion_file: str
        The distortion correction only field (in voxels).
    """
    # Check the input parameter
    for path in (epi_file, structural_file, brain_structural_file,
                 fieldmap_file, magnitude_file, brain_magnitude_file,
                 wmseg_file):
        if path is not None and not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid input file.".format(path))

    # Define the FSL command
    cmd = [
        "epi_reg",
        "--epi={0}".format(epi_file),
        "--t1={0}".format(structural_file),
        "--t1brain={0}".format(brain_structural_file),
        "--out={0}".format(output_fileroot),
        "-v"]
    if fieldmap_file is not None:
        cmd.extend([
            "--fmap={0}".format(fieldmap_file),
            "--echospacing={0}".format(effective_echo_spacing),
            "--fmapmag={0}".format(magnitude_file),
            "--fmapmagbrain={0}".format(brain_magnitude_file),
            "--pedir={0}".format(phase_encode_dir)])
    if wmseg_file is not None:
        cmd.append("--wmseg={0}".format(wmseg_file))

    # Call epi_reg
    fslprocess = FSLWrapper(cmd, shfile=fsl_sh)
    fslprocess()

    # Get outputs
    corrected_epi_file = glob.glob(output_fileroot + ".*")[0]
    if fieldmap_file is not None:
        warp_file = glob.glob(output_fileroot + "_warp.*")[0]
        distortion_file = glob.glob(
            output_fileroot + "_fieldmaprads2epi_shift.*")[0]
    else:
        warp_file = None
        distortion_file = None

    return corrected_epi_file, warp_file, distortion_file


def fsl_prepare_fieldmap(
        manufacturer, phase_file, brain_magnitude_file, output_file,
        delta_te, fsl_sh=DEFAULT_FSL_PATH):
    """ Prepares a fieldmap suitable for FEAT from SIEMENS data.

    Saves output in rad/s format.

    Parameters
    ----------
    manufacturer: str
        The manufacturer name.
    phase_file: str
        The phase image.
    brain_magnitude_file: str
        The magnitude brain image: should only contains brain tissues.
    output_file: str
        The generated fieldmap image.
    delta_te: float
        The echo time difference of the fieldmap.
    fsl_sh: str, default DEFAULT_FSL_PATH
        The FSL configuration batch.

    Returns
    -------
    output_file: str
        The generated fieldmap image.
    """
    # Check the input parameter
    for path in (phase_file, brain_magnitude_file):
        if not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid input file.".format(path))

    # Define the FSL command
    cmd = ["fsl_prepare_fieldmap", manufacturer, phase_file,
           brain_magnitude_file, output_file, delta_te]

    # Call fsl_prepare_fieldmap
    fslprocess = FSLWrapper(cmd, shfile=fsl_sh)
    fslprocess()

    return output_file


def eddy(
        dwi,
        dwi_brain_mask,
        acqp,
        index,
        bvecs,
        bvals,
        deformation_field,
        outroot,
        strategy="openmp",
        fsl_sh=DEFAULT_FSL_PATH):
    """ Wraps FSL eddy tool to correct eddy currents and movements in
    diffusion data:

    * 'eddy_cuda' runs on multiple GPUs. For more information, refer to:
      https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--mask. You may
      need to install nvidia-cuda-toolkit'.
    * 'eddy_openmp' runs on multiple CPUs. The outlier replacement step is
      not available with this precessing strategy.

    Note that this code is working with FSL >= 5.0.11.

    Parameters
    ----------
    dwi: str
        path to dwi volume.
    dwi_brain_mask: str
        path to dwi brain mask segmentation.
    acqp: str
        path to the required eddy acqp file. Refer to:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/Faq#
        How_do_I_know_what_to_put_into_my_--acqp_file
    index: str
        path to the required eddy index file. Refer to:
        https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide#A--imain
    bvecs: str
        path to the bvecs file.
    bvals: str
        path to the bvals file.
    deformation_field: str
        path to the deformation field.
    outroot: str
        fileroot name for output.
    strategy: str, default 'openmp'
        the execution strategy: 'openmp' or 'cuda'.
    fsl_sh: str, optional default 'DEFAULT_FSL_PATH'
        path to fsl setup sh file.

    Returns
    -------
    corrected_dwi: str
        path to the corrected DWI.
    corrected_bvec: str
        path to the rotated b-vectors.
    """
    # The Eddy command
    cmd = [
        "eddy_{0}".format(strategy),
        "--imain={0}".format(dwi),
        "--mask={0}".format(dwi_brain_mask),
        "--acqp={0}".format(acqp),
        "--index={0}".format(index),
        "--bvecs={0}".format(bvecs),
        "--bvals={0}".format(bvals),
        "--field={0}".format(deformation_field),
        "--repol",
        "--out={0}".format(outroot),
        "-v"]

    # Run the Eddy correction
    fslprocess = FSLWrapper(cmd, shfile=fsl_sh)
    fslprocess()

    # Get the outputs
    corrected_dwi = "{0}.nii.gz".format(outroot)
    corrected_bvec = "{0}.eddy_rotated_bvecs".format(outroot)

    return corrected_dwi, corrected_bvec


def concatenate_volumes(nii_files, bvals_files, bvecs_files, outdir, axis=-1):
    """ Concatenate volumes of different nifti files.

    Parameters
    ----------
    nii_files: array of str
        array containing the different nii files to concatenate.
    bvals_files: list of str
        path to the diffusion b-values files.
    bvecs_files: list of str
        path to the diffusion b-vectors files.
    outdir: str
        subject output directory.
    axis: int, default -1
        the concatenation axis.

    Returns
    -------
    dwi_file: str
        path to the concatenated nii files.
    bval_file: str
        path to the concatenated bval files.
    bvec_file: str
        path to the concatenated bvec files.
    """
    # Concatenate volumes
    data = []
    affines = []
    for path in nii_files:
        im = nibabel.load(path)
        data.append(im.get_data())
        affines.append(im.affine)
    concatenated_volumes = numpy.concatenate(data, axis=axis)

    # Check that affine are the same between volumes
    ref_affine = affines[0]
    for aff in affines:
        if not numpy.allclose(ref_affine, aff):
            raise ValueError("Different affines between DWI volumes: {0}"
                             "...".format(nii_files))

    # Read the bvals and bvecs
    bvals, bvecs, nb_shells, nb_nodiff = read_bvals_bvecs(
        bvals_files, bvecs_files, min_bval=200)

    if nb_nodiff > 1:
        nodiff_indexes = numpy.where(bvals == 0)[0].tolist()
        b0_array = concatenated_volumes[..., nodiff_indexes[0]]
        b0_array.shape += (1, )
        cpt_delete = 0
        for i in nodiff_indexes:
            concatenated_volumes = numpy.delete(
                concatenated_volumes, i - cpt_delete, axis=3)
            bvals = numpy.delete(bvals, i - cpt_delete, axis=0)
            bvecs = numpy.delete(bvecs, i - cpt_delete, axis=0)
            cpt_delete += 1
        concatenated_volumes = numpy.concatenate(
            (b0_array, concatenated_volumes), axis=3)
        bvals = numpy.concatenate((numpy.array([0]), bvals), axis=0)
        bvecs = numpy.concatenate((numpy.array([[0, 0, 0]]), bvecs), axis=0)

    # Save the results
    dwi_file = os.path.join(outdir, "dwi.nii.gz")
    bval_file = os.path.join(outdir, "dwi.bval")
    bvec_file = os.path.join(outdir, "dwi.bvec")
    concatenated_nii = nibabel.Nifti1Image(concatenated_volumes, ref_affine)
    nibabel.save(concatenated_nii, dwi_file)
    bvals.shape += (1, )
    numpy.savetxt(bval_file, bvals.T, fmt="%f")
    numpy.savetxt(bvec_file, bvecs.T, fmt="%f")

    return dwi_file, bval_file, bvec_file


def get_dcm_info(dicom_dir, outdir, dicom_img=None):
    """ Get the sequence parameters, especiallt the phase encoded direction.

    Parameters
    ----------
    dicom_dir: str
        path to the dicoms directory.
    outdir: str
        path to the subject output directory.
    dicom_img: dicom.dataset.FileDataset object, default None
        one of the dicom image loaded by pydicom. If not specified load one
        DICOM file available in the 'dicom_dir' folder.

    Returns
    -------
    dcm_info: dict
        Dictionnary with scanner characteristics.  The phase encode direction
        is encoded as (i, -i, j, -j).
    """
    # Dicom phase encoding direction tag
    p_enc_tag = [24, 4882]

    # Dicom manufacturer tag (0008, 0070)
    manufacturer_tag = [8, 112]

    # Magnetic field strength (0018,0087)
    field_tag = [24, 135]

    # Load the image if necessary
    if dicom_img is None:
        dicom_img = dicom.read_file(
            glob.glob(os.path.join(dicom_dir, "*.*"))[0])

    # Get the manufacturer
    manufacturer = dicom_img[manufacturer_tag[0], manufacturer_tag[1]].value

    # Use DICOM files
    if manufacturer == "Philips Medical Systems":

        phase_enc_dir = dicom_img[p_enc_tag[0], p_enc_tag[1]].value
        if phase_enc_dir == "COL":
            phase_enc_dir = "j"
        elif phase_enc_dir == "ROW":
            phase_enc_dir = "i"
        else:
            raise ValueError("Unknown phase encode direction: "
                             "{0}".format(phase_enc_dir))
        dcm_info = {"PhaseEncodingDirection": phase_enc_dir}

    # Use dcm2niix
    elif manufacturer in ["SIEMENS", "GE", "GE MEDICAL SYSTEMS"]:
        dcm_info_dir = os.path.join(outdir, "DCM_INFO")
        if os.path.isdir(dcm_info_dir):
            shutil.rmtree(dcm_info_dir)
        os.mkdir(dcm_info_dir)
        cmd = ["dcm2niix", "-b", "o", "-v", "n", "-o", dcm_info_dir, dicom_dir]
        cmd = " ".join(cmd)
        os.system(cmd)
        dcm_info_json = glob.glob(os.path.join(dcm_info_dir, "*.json"))[0]
        with open(dcm_info_json, "rb") as open_file:
            dcm_info = json.load(open_file)
        if manufacturer == "SIEMENS":
            phase_enc_dir = dcm_info["PhaseEncodingDirection"]
        if manufacturer in ["GE", "GE MEDICAL SYSTEMS"]:
            phase_enc_dir = dcm_info["InPlanePhaseEncodingDirectionDICOM"]
            if phase_enc_dir == "COL":
                phase_enc_dir = "j"
            elif phase_enc_dir == "ROW":
                phase_enc_dir = "i"
            else:
                raise ValueError("Unknown phase encode direction: "
                                 "{0}".format(phase_enc_dir))
            dcm_info["PhaseEncodingDirection"] = phase_enc_dir
    else:
        raise ValueError("Unknown scanner: {0}...".format(manufacturer))

    # Add some information
    dcm_info["Manufacturer"] = manufacturer
    dcm_info["MagneticFieldStrength"] = float(dicom_img[
        field_tag[0], field_tag[1]].value)

    return dcm_info


def get_readout_time(dicom_img, dcm_info):
    """ Get read out time from a dicom image.

    Parameters
    ----------
    dicom_img: dicom.dataset.FileDataset object
        one of the dicom image loaded by pydicom.
    dcm_info: dict
        array containing dicom data.

    Returns
    -------
    readout_time: float
        read-out time in seconds.

    For philips scanner
    ~~~~~~~~~~~~~~~~~~~
    Formula to compute read out time:
    echo spacing (seconds) * (epi - 1)

    Formula to compute echo spacing:
    (water-fat shift (per pixel)/(water-fat shift (in Hz) * echo train length))

    Formula to compute water shift in Hz:
    fieldstrength (T) * water-fat difference (ppm) * resonance frequency(MHz/T)

    References :
    http://support.brainvoyager.com/functional-analysis-preparation/
    27-pre-processing/459-epi-distortion-correction-echo-spacing.html
    and
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/
    Faq#How_do_I_know_what_to_put_into_my_--acqp_file

    For Siemens scanner
    ~~~~~~~~~~~~~~~~~~~
    Use dcm2niix to generate a json summary of the parameters.
    """
    # Compute readout time
    manufacturer = dcm_info["Manufacturer"]
    b0 = dcm_info["MagneticFieldStrength"]
    if manufacturer == "Philips Medical Systems":

        # Compute water-fat shift (per pixel)
        # gyromagnetic_proton_gamma_ratio = gamma /2pi = 42.576 MHz/T.
        gyromagnetic_proton_gamma_ratio = 42.576  # MHz/T
        delta_b0 = b0 * gyromagnetic_proton_gamma_ratio * pow(10, 6)  # Hz

        # Number of lines in k-spaces per slice
        # Generally nb of voxels in the phase encode direction multiplied by
        # Fourier partial ratio and divided by acceleration factor SENSE or
        # GRAPPA (iPAT)
        fourier_partial_ratio = dicom_img[24, 147].value  # Percent sampling
        acceleration_factor = dicom_img[int("2005", 16),
                                        int("140f", 16)][0][24, 36969].value
        nb_phase_encoding_steps = dicom_img[24, 137].value
        Ny = (
            float(nb_phase_encoding_steps) *
            float(fourier_partial_ratio) /
            100)
        Ny = Ny / acceleration_factor

        # Pixel bandwith (BW/Nx) Hz/pixel
        BW_Nx = float(dicom_img[24, 149].value)
        water_shift_pixel = delta_b0 * Ny / BW_Nx  # pixel

        # Water shift (Hz)
        # Haacke et al: 3.35ppm. Bernstein et al (pg. 960): Chemical shifts
        # (ppm, using protons in tetramethyl silane Si(CH3)4 as a reference).
        # Protons in lipids ~1.3, protons in water 4.7, difference:
        # 4.7 - 1.3 = 3.4.
        water_fat_difference = 3.35  # ppm

        # Resonance frequency (Hz/T)
        resonance_frequency = 42.576 * pow(10, 6)  # Haacke et al.
        # water_shift_hertz (Hz)
        water_shift_hertz = b0 * water_fat_difference * resonance_frequency

        # Compute echo spacing
        etl = float(dicom_img[24, 145].value)  # echo train length
        echo_spacing = water_shift_pixel / (water_shift_hertz * etl)  # s

        # Compute readout time
        # Compute number of phase encoding steps epi
        epi = float(dicom_img[int("0018", 16), int("0089", 16)].value)
        readout_time = echo_spacing * (epi - 1)

    elif manufacturer == "SIEMENS" or "GE":
        readout_time = dcm_info["TotalReadoutTime"]

    else:
        raise ValueError("Unknown manufacturer : {0}".format(manufacturer))

    return readout_time
