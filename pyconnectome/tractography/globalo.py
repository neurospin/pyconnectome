# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Compute global tractography using MRtrix or MITK Gibbs Tracking.
"""

# Standard
import os
import subprocess
import shutil
import tempfile
from xml.etree import ElementTree

# Package
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper
from pyconnectome.utils.filetools import mrtrix_extract_b0s_and_mean_b0
from pyconnectome.utils.regtools import freesurfer_bbregister_t1todif

# Third-party
from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper
from pyfreesurfer.utils.filetools import get_or_check_freesurfer_subjects_dir


def mitk_gibbs_tractogram(
        outdir,
        subject_id,
        dwi,
        bvals,
        bvecs,
        nodif_brain=None,
        nodif_brain_mask=None,
        subjects_dir=None,
        sh_order=4,
        reg_factor=0.006,
        nb_iterations=5e8,
        particle_length=0,
        particle_width=0,
        particle_weight=0,
        start_temperature=0.1,
        end_temperature=0.001,
        inex_energy_balance=0,
        min_fiber_length=20,
        curvature_threshold=45,
        tempdir=None,
        fs_sh=DEFAULT_FREESURFER_PATH,
        fsl_sh=DEFAULT_FSL_PATH):
    """
    Wrapper to the MITK global tractography tool (Gibbs Tracking).

    Parameters
    ----------
    outdir: str
        Directory where to output.
    subject_id: str
        Subject id used with FreeSurfer 'recon-all' command.
    dwi: str
        Path to the diffusion-weighted images (Nifti required).
    bvals: str
        Path to the bvalue list.
    bvecs: str
        Path to the list of diffusion-sensitized directions.
    nodif_brain: str, default None
        Diffusion brain-only Nifti volume with bvalue ~ 0. If not passed, it is
        generated automatically by averaging all the b0 volumes of the DWI.
    nodif_brain_mask: str, default None
        Path to the Nifti brain binary mask in diffusion. If not passed, it is
        created with MRtrix 'dwi2mask'.
    subjects_dir: str or None, default None
        Path to the FreeSurfer subjects directory. Required if the FreeSurfer
        environment variable (i.e. $SUBJECTS_DIR) is not set.
    sh_order: int, default 4
        Qball reconstruction spherical harmonics order.
    reg_factor: float, default
        Qball reconstruction regularization factor..
    nb_iterations: int, default 5E8
        Gibbs tracking number of iterations.
    particle_length: float, default 0
         Gibbs tracking particle length, selected automatically if 0.
    particle_width: float, default 0
        Gibbs tracking particle width, selected automatically if 0.
    particle_weight: float, default 0
        Gibbs tracking particle weight, selected automatically if 0.
    start_temperature: float, default 0.1
        Gibbs tracking start temperature.
    end_temperature: float, default 0.001
        Gibbs tracking end temperature.
    inex_energy_balance: float, default 0
        Gibbs tracking weighting between in/ext energies.
    min_fiber_length: int, default 20
        Minimum fiber length in mm. Fibers that are shorter are discarded.
    curvature_threshold: int, default 45
        Maximum fiber curvature in degrees.
    tempdir: str
        Path to the directory where temporary directories should be written.
        It should be a partition with 5+ GB available.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.

    Returns
    -------
    mitk_tractogram: str
        The computed global tractogram in VTK format.
    """
    # -------------------------------------------------------------------------
    # STEP 0 - Check arguments

    # FreeSurfer subjects_dir
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    # Check input paths
    paths_to_check = [dwi, bvals, bvecs, nodif_brain_mask, fs_sh, fsl_sh]
    for p in [nodif_brain, nodif_brain_mask]:
        if p is not None:
            paths_to_check.append(p)
    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Create <outdir> and/or <tempdir> if not existing
    for directory in [outdir, tempdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # -------------------------------------------------------------------------
    # STEP 1 - Compute DWI to T1 transformation and project the T1
    #          to the diffusion space without resampling.

    # If user has not provided a 'nodif_brain_mask', compute one with
    # MRtrix 'dwi2mask'
    if nodif_brain_mask is None:
        nodif_brain_mask = os.path.join(outdir, "nodif_brain_mask.nii.gz")
        cmd_1a = ["dwi2mask", dwi, nodif_brain_mask, "-fslgrad", bvecs, bvals]
        subprocess.check_call(cmd_1a)

    # If user has not provided a 'nodif_brain', apply 'nodif_brain_mask' to
    # mean b=0 volume
    if nodif_brain is None:
        # Extract b=0 volumes and compute mean b=0 volume
        b0s = os.path.join(outdir, "b0s.nii.gz")
        mean_b0 = os.path.join(outdir, "mean_b0.nii.gz")
        mrtrix_extract_b0s_and_mean_b0(dwi=dwi, b0s=b0s, mean_b0=mean_b0,
                                       bvals=bvals, bvecs=bvecs, nb_threads=1)
        # Apply nodif_brain_mask to dwi
        nodif_brain = os.path.join(outdir, "nodif_brain.nii.gz")
        cmd_1b = ["mri_mask", mean_b0, nodif_brain_mask, nodif_brain]
        FSWrapper(cmd_1b, shfile=fs_sh)()

    # Register nodif_brain to FreeSurfer T1
    t1_brain_to_dif, dif2anat_dat, _ = freesurfer_bbregister_t1todif(
        outdir=outdir,
        subject_id=subject_id,
        nodif_brain=nodif_brain,
        subjects_dir=subjects_dir,
        fs_sh=fs_sh,
        fsl_sh=fsl_sh)

    # -------------------------------------------------------------------------
    # STEP 2 - Apply brain mask to DWI before Qball reconstruction
    dwi_brain = os.path.join(outdir, "dwi_brain.nii.gz")
    cmd_4 = ["fslmaths", dwi, "-mas", nodif_brain_mask, dwi_brain]
    process = FSLWrapper(shfile=fsl_sh)
    process(cmd=cmd_4)

    # MITK requires the Nifti to have an .fslgz extension and the bvals/bvecs
    # to have the same name with .bvals/.bvecs extension
    dwi_brain_fslgz = os.path.join(outdir, "dwi_brain.fslgz")
    shutil.copyfile(dwi_brain, dwi_brain_fslgz)
    shutil.copyfile(bvals, "%s.bvals" % dwi_brain_fslgz)
    shutil.copyfile(bvecs, "%s.bvecs" % dwi_brain_fslgz)

    # -------------------------------------------------------------------------
    # STEP 3 - Qball reconstruction
    qball_coefs = os.path.join(outdir, "sphericalHarmonics_CSA_Qball.qbi")
    cmd_5 = ["MitkQballReconstruction.sh",
             "-i",  dwi_brain_fslgz,
             "-o",  qball_coefs,
             "-sh", "%i" % sh_order,
             "-r",  "%f" % reg_factor,
             "-csa",
             "--mrtrix"]
    # TODO: create MITK wrapper with LD_LIBRARY_PATH and QT_PLUGIN_PATH
    subprocess.check_call(cmd_5)

    # -------------------------------------------------------------------------
    # STEP 4 - Create white matter probability map with FSL Fast

    # Create directory for temporary files
    fast_tempdir = tempfile.mkdtemp(prefix="FSL_fast_", dir=tempdir)
    base_outpath = os.path.join(fast_tempdir, "brain")
    cmd_6 = ["fast", "-o", base_outpath, t1_brain_to_dif]
    process = FSLWrapper(cmd_6, shfile=fsl_sh)
    process(cmd=cmd_6)

    # Save the white matter probability map
    wm_prob_map = os.path.join(outdir, "wm_prob_map.nii.gz")
    shutil.copyfile(base_outpath + "_pve_2.nii.gz", wm_prob_map)

    # Clean temporary directory
    shutil.rmtree(fast_tempdir)

    # -------------------------------------------------------------------------
    # STEP 5 - Gibbs tracking (global tractography)

    # Create XML parameter file
    root = ElementTree.Element("global_tracking_parameter_file")
    root.set("version", "1.0")
    attributes = {"iterations":          "%i" % nb_iterations,
                  "particle_length":     "%f" % particle_length,
                  "particle_width":      "%f" % particle_width,
                  "particle_weight":     "%f" % particle_weight,
                  "temp_start":          "%f" % start_temperature,
                  "temp_end":            "%f" % end_temperature,
                  "inexbalance":         "%f" % inex_energy_balance,
                  "fiber_length":        "%i" % min_fiber_length,
                  "curvature_threshold": "%i" % curvature_threshold}
    ElementTree.SubElement(root, "parameter_set", attrib=attributes)
    tree = ElementTree.ElementTree(element=root)
    path_xml = os.path.join(outdir, "parameters.gtp")
    tree.write(path_xml)

    # Run tractography
    mitk_tractogram = os.path.join(outdir, "fibers.fib")
    cmd_7 = ["MitkGibbsTracking.sh",
             "-i", qball_coefs,
             "-p", path_xml,
             "-m", wm_prob_map,
             "-o", mitk_tractogram,
             "-s", "MRtrix"]
    subprocess.check_call(cmd_7)

    return mitk_tractogram
