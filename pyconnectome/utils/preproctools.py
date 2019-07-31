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
from pydcmio.dcmconverter.converter import dcm2niix


def topup(
        b0s,
        phase_enc_dirs,
        readout_time,
        outroot,
        apply_to=None,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Wraps FSL topup tool to estimate the susceptibility induced
    off-resonance field.

    Parameters
    ----------
    b0s: list of str
        path to b0 file acquired in opposite phase enc. directions.
    phase_enc_dirs: list of str
        the phase enc. directions.
    readout_time: float
        the readout time.
    outroot: str
        fileroot name for output.
    apply_to: 2-uplet, default None
        apply the topup correction to the volumes acquired in the blip up and
        blip down acquisition settings respectivelly. Will take the first
        and last indices of the acquisiton parameters file during the
        correction.
    fsl_sh: str, optional default 'DEFAULT_FSL_PATH'
        path to fsl setup sh file.

    Returns
    -------
    fieldmap: str
        path to the fieldmap in Hz.
    corrected_b0s: str
        path to the unwarped b0 images.
    mean_corrected_b0s: str
        path to the mean unwarped b0 images.
    """

    # Write topup acqp
    # https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide
    if len(b0s) != len(phase_enc_dirs):
        raise ValueError("Please specify properly the topup input data.")
    acqp_file = os.path.join(outroot, "acqp.txt")
    data = []
    affine = None
    tot_nvol = 0
    with open(acqp_file, "wt") as open_file:
        for enc_dir, path in zip(phase_enc_dirs, b0s):
            im = nibabel.load(path)
            if affine is None:
                affine = im.affine
            else:
                assert numpy.allclose(affine, im.affine)
            arr = im.get_data()
            for cnt, size in enumerate(arr.shape[:3]):
                if size % 2 == 1:
                    print("[warn] reducing TOPUP B0 image size.")
                    arr = numpy.delete(arr, -1, axis=cnt)
            if arr.ndim == 3:
                arr.shape += (1, )
            data.append(arr)
            nvol = arr.shape[-1]
            tot_nvol += nvol
            if enc_dir == "i":
                row = "1 0 0 {0}".format(readout_time)
            elif enc_dir == "i-":
                row = "-1 0 0 {0}".format(readout_time)
            elif enc_dir == "j":
                row = "0 1 0 {0}".format(readout_time)
            elif enc_dir == "j-":
                row = "0 -1 0 {0}".format(readout_time)
            else:
                raise ValueError("Unknown encode phase direction : "
                                 "{0}...".format(enc_dir))
            for indx in range(nvol):
                open_file.write(row + "\n")
    concatenated_b0s_file = os.path.join(outroot, "concatenated_b0s.nii.gz")
    concatenated_b0s = numpy.concatenate(data, axis=-1)
    concatenated_b0s_im = nibabel.Nifti1Image(concatenated_b0s, affine)
    nibabel.save(concatenated_b0s_im, concatenated_b0s_file)

    # The topup command
    fieldmap = os.path.join(outroot, "fieldmap.nii.gz")
    corrected_b0s = os.path.join(outroot, "unwarped_b0s.nii.gz")
    cmd = [
        "topup",
        "--imain={0}".format(concatenated_b0s_file),
        "--datain={0}".format(acqp_file),
        "--config=b02b0.cnf",
        "--out={0}".format(os.path.join(outroot, "topup")),
        "--fout={0}".format(fieldmap),
        "--iout={0}".format(corrected_b0s),
        "-v"]
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

    # Apply topup correction
    if apply_to is not None:
        if len(apply_to) != 2:
            raise ValueError(
                "Need two volumes acquired in the blip up and blip down "
                "acquisition settings to apply topup.")
        cmd = [
            "applytopup",
            "--imain={0}".format(",".join(apply_to)),
            "--inindex=1,{0}".format(tot_nvol),
            "--datain={0}".format(acqp_file),
            "--topup={0}".format(os.path.join(outroot, "topup")),
            "--out={0}".format(os.path.join(outroot, "applytopup"))]
        fslprocess(cmd=cmd)

    # Average b0s
    mean_corrected_b0s = os.path.join(outroot, "mean_unwarped_b0s.nii.gz")
    cmd = [
        "fslmaths",
        corrected_b0s,
        "-Tmean", mean_corrected_b0s]
    fslprocess(cmd=cmd)

    return fieldmap, corrected_b0s, mean_corrected_b0s


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
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

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
        The generated fieldmap image in rad/s.
    output_hz_file: str
        The generated fieldmap image in Hz.
    """
    # Check the input parameter
    for path in (phase_file, brain_magnitude_file):
        if not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid input file.".format(path))
    if not output_file.endswith(".nii.gz"):
        output_file += ".nii.gz"

    # Define the FSL command
    cmd = ["fsl_prepare_fieldmap", manufacturer, phase_file,
           brain_magnitude_file, output_file, delta_te]

    # Call fsl_prepare_fieldmap
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

    # Convert the fieldmap in rad/s to Hz
    output_hz_file = output_file.replace(".nii.gz", "_hz.nii.gz")
    cmd = ["fslmaths", output_file, "-div", "6.28", output_hz_file]
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

    return output_file, output_hz_file


def eddy(
        dwi,
        dwi_brain_mask,
        acqp,
        index,
        bvecs,
        bvals,
        outroot,
        field=None,
        no_qspace_interpolation=False,
        no_slice_correction=True,
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
    outroot: str
        fileroot name for output.
    field: str, default None
        path to the field map in Hz.
    no_qspace_interpolation: bool, default False
        if set do not remove any slices deemed as outliers and replace them
        with predictions made by the Gaussian Process.
    no_slice_correction: bool, True
        if set do not perform the slice to volume correction.
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
        "--out={0}".format(outroot),
        "-v"]
    if field is not None:
        cmd += ["--field={0}".format(field)]
    if not no_qspace_interpolation:
        cmd += ["--repol"]
    else:
        cmd += ["--data_is_shelled"]
    if not no_slice_correction:
        cmd += [
            "--mporder=6",
            "--s2v_niter=5",
            "--s2v_lambda=1",
            "--s2v_interp=trilinear"]

    # Run the Eddy correction
    print(" ".join(cmd))
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

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
        nodiff_indexes = numpy.where(bvals <= 50)[0].tolist()
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
    """ Get the sequence parameters, especially the phase encoded direction.
    Uses Christopher Rorden tool dcm2niix.

    Parameters
    ----------
    dicom_dir: str
        path to the dicoms directory.
    outdir: str
        path to the subject output directory.

    Returns
    -------
    dcm_info: dict
        Dictionnary with scanner characteristics.  The phase encode direction
        is encoded as (i, -i, j, -j).
        /!\ Warnings : For Philips and GE scanners only phase encode direction
            axis is available and is named PhaseEncodingAxis.
            Phase encode direction axis + orientation is only available for
            Siemens scanners and is named PhaseEncodingDirection.
        /!\
    """

    # Use Christopher Rorden tool dcm2niix to extract other dicom info.
    dcm_info_dir = os.path.join(outdir, "DCM_INFO")
    if os.path.isdir(dcm_info_dir):
        shutil.rmtree(dcm_info_dir)
    os.mkdir(dcm_info_dir)
    _, _, _, bids = dcm2niix(
        input=dicom_dir,
        o=dcm_info_dir,
        f="%p",
        z="n",
        b="o")
    if len(bids) == 0:
        raise ValueError(
            "Dcm2niix could not extract information from '{0}'".format(
                dicom_dir))
    with open(bids[0], "rb") as open_file:
        dcm_info = json.load(open_file)

    return dcm_info


def get_readout_time(dicom_img, dcm_info, dwell_time):
    """ Get read out time from a dicom image.

    Parameters
    ----------
    dicom_img: dicom.dataset.FileDataset object
        one of the dicom image loaded by pydicom.
    dcm_info: dict
        array containing dicom data.
    dwell_time: float
        Effective echo spacing in s.

    Returns
    -------
    readout_time: float
        read-out time in seconds.

    For philips scanner
    ~~~~~~~~~~~~~~~~~~~
    Formula to compute read out time:
    echo spacing (seconds) * (epi - 1) where
    epi = nb of phase encoding steps/ acceleration factor.

    For Siemens and GE scanners
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Readout time is written by dcm2niix in json summary dcm_info.
    """

    manufacturer = dcm_info["Manufacturer"]
    if manufacturer.upper() in ["SIEMENS", "GE MEDICAL SYSTEMS", "GE"]:
        readout_time = dcm_info["TotalReadoutTime"]

    elif manufacturer.upper() in ["PHILIPS MEDICAL SYSTEMS", "PHILIPS"]:
        acceleration_factor = dicom_img[int("2005", 16),
                                        int("140f", 16)][0][24, 36969].value
        etl = float(dicom_img[0x0018, 0x0089].value)
        readout_time = dwell_time * (etl - 1)
    else:
        raise ValueError("Unknown manufacturer : {0}".format(manufacturer))

    return readout_time


def get_dwell_time(dicom_img, dcm_info):
    """ Get the dwell time or effective echo spacing.
        Returns effective echo spacing written in dcm_info for Siemens and GE
        scanners and compute it for Philips scanner as described in:
        https://www.spinozacentre.nl/wiki/index.php/NeuroWiki:Current_\\
        developments

    For further references see:
    http://support.brainvoyager.com/functional-analysis-preparation/
    27-pre-processing/459-epi-distortion-correction-echo-spacing.html
    and
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/
    Faq#How_do_I_know_what_to_put_into_my_--acqp_file

    Parameters
    ----------
    dicom_img: dicom.dataset.FileDataset object
        one of the dicom image loaded by pydicom.
    dcm_info: dict
        dictionary containing dicom metadata.

    Returns
    -------
    dwell_time: float
        effective echo spacing in s.
    """
    manufacturer = dcm_info["Manufacturer"]

    if manufacturer.upper() in ["SIEMENS", "GE MEDICAL SYSTEMS", "GE"]:
        dwell_time = dcm_info["EffectiveEchoSpacing"]

    elif manufacturer.upper() in ["PHILIPS MEDICAL SYSTEMS", "PHILIPS"]:

        # Compute pixel water fat shift
        gyromagnetic_proton_gamma_ratio = 42.576 * pow(10, 6)  # Hz/T
        b0 = float(dicom_img[0x0018, 0x0087].value)
        water_fat_difference = 3.35 * pow(10, -6)  # ppm
        delta_b0 = water_fat_difference * b0

        # Ny : Number of lines in k-spaces per slice
        # Generally nb of voxels in the phase encode direction multiplied by
        # Fourier partial ratio and divided by acceleration factor SENSE or
        # GRAPPA (iPAT)
        fourier_partial_ratio = dicom_img[24, 147].value  # Percent sampling
        acceleration_factor = dicom_img[int("2005", 16),
                                        int("140f", 16)][0][24, 36969].value
        nb_phase_encoding_steps = float(dicom_img[24, 137].value)
        Ny = (
            float(nb_phase_encoding_steps) *
            (float(fourier_partial_ratio) /
             100))
        Ny = Ny / acceleration_factor
        BW_Nx = float(dicom_img[24, 149].value)
        water_shift_pixel = (gyromagnetic_proton_gamma_ratio * delta_b0 * Ny /
                             BW_Nx)  # pixel

        # Calculate Water fat shift in hertz
        # Resonance frequency (Hz/T)
        resonance_frequency = 42.576 * pow(10, 6)  # Haacke et al.

        # Water_shift_hertz (Hz)
        water_shift_hertz = b0 * water_fat_difference * resonance_frequency

        # Number of phase encoding steps
        etl = float(dicom_img[0x0018, 0x0089].value)
        dwell_time = (water_shift_pixel / (water_shift_hertz * etl /
                      acceleration_factor))
    else:
        raise ValueError("Unknown manufacturer : {0}...".format(manufacturer))

    return dwell_time


def pixel_shift_to_fieldmap(pixel_shift_file, dwell_time, output_file,
                            fsl_sh=DEFAULT_FSL_PATH):
    """ Convert a pixel shift map to a FSL field map.

    Parameters
    ----------
    pixel_shift_file: str
        the pixel shift map.
    dwell_time: float
        the dwell time in s.
    output_file: str
        The generated fieldmap image.
    fsl_sh: str, optional default 'DEFAULT_FSL_PATH'
        path to fsl setup sh file.

    Returns
    -------
    fieldmap_file: str
        the FSL fieldmap in rad/s.
    fieldmap_hz_file: str
        the FSL fieldmap in Hz.
    """
    # Check the input parameter
    if not os.path.isfile(pixel_shift_file):
        raise ValueError("'{0}' is not a valid input file.".format(
            pixel_shift_file))
    if not output_file.endswith(".nii.gz"):
        output_file += ".nii.gz"

    # Convert the fieldmap
    cmd = [
        "fugue",
        "--dwell={0}".format(dwell_time),
        "--loadshift={0}".format(pixel_shift_file),
        "--savefmap={0}".format(output_file)]
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

    # Convert the fieldmap in rad/s to Hz
    output_hz_file = output_file.replace(".nii.gz", "_hz.nii.gz")
    cmd = ["fslmaths", output_file, "-div", "6.28", output_hz_file]
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)

    return output_file, output_hz_file


def smooth_fieldmap(fieldmap, dwell_time, output_file, sigma=2,
                    fsl_sh=DEFAULT_FSL_PATH):
    """ Smooth a field map using FSL using a Gaussian 3D kernel.

    Parameters
    ----------
    fieldmap_file: str
        the FSL fieldmap.
    dwell_time: float
        the dwell time in s.
    output_file: str
        The generated smoothed fieldmap image.
    sigma: float, default 2
        The Gaussian smoothing kernel sigma.
    fsl_sh: str, optional default 'DEFAULT_FSL_PATH'
        path to fsl setup sh file.
    """
    cmd = [
        "fugue",
        "--dwell={0}".format(dwell_time),
        "--loadfmap={0}".format(fieldmap),
        "--savefmap={0}".format(output_file),
        "-s", str(sigma)]
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)


def fieldmap_reflect(fieldmap, phase_enc_dir, output_file):
    """ Replace the zero values in the fieldmap by the last non null value in
    the phase encoding direction.

    Parameters
    ----------
    fieldmap_file: str
        the FSL fieldmap.
    phase_enc_dir: str
        the phase encoding direction.
    output_file: str
        The generated reflected fieldmap.

    Returns
    -------
    reflect_fieldmap: str
        the filled fieldmap.
    """
    # Check input parameters
    if phase_enc_dir not in ("i", "i-"):
        axis = 0
    elif phase_enc_dir not in ("j", "j-"):
        axis = 1
    else:
        raise ValueError("Unsupported phase encoded direction '{0}'.".format(
            phase_enc_dir))

    # Load field map
    im = nibabel.load(fieldmap)
    arr = im.get_data()
    if arr.ndim != 3:
        raise ValueError("A 3d array is expected!")

    # Reflection
    transpose = [0, 1, 2]
    transpose.pop(axis)
    transpose.insert(0, axis)
    arr = arr.transpose(transpose)
    for indx_slice in range(arr.shape[2]):
        for indx_pedir in range(arr.shape[0]):
            indices = sorted(numpy.where(
                arr[indx_pedir, :, indx_slice] != 0)[0])
            if len(indices) == 0:
                continue
            min_index = indices[0]
            min_value = arr[indx_pedir, min_index, indx_slice]
            arr[indx_pedir, :min_index, indx_slice] = min_value
            max_index = indices[-1]
            max_value = arr[indx_pedir, max_index, indx_slice]
            arr[indx_pedir, max_index:, indx_slice] = max_value
    arr = arr.transpose(transpose)

    # Save result
    im_reflect = nibabel.Nifti1Image(arr, im.affine)
    nibabel.save(im_reflect, output_file)
