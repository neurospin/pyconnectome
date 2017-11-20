##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Compute the connectome of a given tesellation, like the FreeSurfer, using
FSL Probtrackx2.
"""

# Standard import
import os
import subprocess
import numpy
import nibabel

# PyFreeSurfer import
from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper
from pyfreesurfer.utils.filetools import get_or_check_freesurfer_subjects_dir
from pyfreesurfer.conversions.volconvs import mri_binarize
from pyfreesurfer.conversions.volconvs import mri_convert

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper
from pyconnectome.tractography.probabilist import probtrackx2


def probtrackx2_connectome_complete(outdir,
                                    subject_id,
                                    lh_surf,
                                    rh_surf,
                                    nodif_brain,
                                    nodif_brain_mask,
                                    bedpostx_dir,
                                    nsamples,
                                    nsteps,
                                    steplength,
                                    subjects_dir=None,
                                    loopcheck=True,
                                    cthr=0.2,
                                    fibthresh=0.01,
                                    distthresh=0.0,
                                    sampvox=0.0,
                                    fs_sh=DEFAULT_FREESURFER_PATH,
                                    fsl_sh=DEFAULT_FSL_PATH):
    """ Compute the connectome of a given tesellation, like the FreeSurfer,
    using ProbTrackx2.

    Requirements:
        - brain masks for the preprocessed DWI: nodif_brain and
          nodif_brain_mask.
        - FreeSurfer: result of recon-all on the T1.
        - FSL Bedpostx: computed for the preprocessed DWI.

    Connectome construction strategy:
        - Pathways are constructed from 'constitutive points' and not from
          endpoints. A pathway is the result of 2 samples propagating in
          opposite directions from a seed point. It is done using the
          --omatrix3 option of Probtrackx2.
        - The seed mask is the mask of WM voxels that are neighbors
          (12-connexity) of nodes.
        - The stop mask is the inverse of white matter, i.e. a sample stops
          propagating as soon as it leaves the white matter.

    Note:

    --randfib refers to initialization of streamlines only (i.e. the very first
    step) and only affects voxels with more than one fiber reconstructed:
    randfib==0, only sample from the strongest fiber
    randfib==1, randomly sample from all fibers regardless of strength that are
    above --fibthresh
    randfib==2, sample fibers stronger than --fibthresh in proportion to their
    strength (in my opinion, this is the best choice)
    randfib==3, sample all fibers randomly regardless of whether or not they
    are above --fibthresh.

    Parameters
    ----------
    outdir: str
        Directory where to output.
    subject_id: str
        Subject id used with FreeSurfer 'recon-all' command.
    lh_surf: str
        The left hemisphere surface.
    rh_surf: str
        The left hemisphere surface.
    nodif_brain: str
        Path to the preprocessed brain-only DWI volume.
    nodif_brain_mask: str
        Path to the brain binary mask.
    bedpostx_dir: str
        Bedpostx output directory.
    nsamples: int
        Number of samples per voxel to initiate in the seed mask.
    nsteps: int
        Maximum number of steps for a given sample.
    steplength: int
        Step size in mm.
    subjects_dir: str or None, default None
        Path to the FreeSurfer subjects directory. Required if the FreeSurfer
        environment variable (i.e. $SUBJECTS_DIR) is not set.
    cthr: float, optional
        Probtrackx2 option.
    fibthresh, distthresh, sampvox: float, optional
        Probtrackx2 options.
    loopcheck: bool, optional
        Probtrackx2 option.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.

    Returns
    ------
    coords: str
        The connectome coordinates.
    weights: str
        The connectome weights.
    """
    # -------------------------------------------------------------------------
    # STEP 0 - Check arguments

    # FreeSurfer subjects_dir
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    # Check input paths
    paths_to_check = [nodif_brain, nodif_brain_mask, bedpostx_dir,
                      fs_sh, fsl_sh]
    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Create <outdir> if not existing
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # -------------------------------------------------------------------------
    # STEP 1 - Compute T1 <-> DWI rigid transformation

    # FreeSurfer T1 to Nifti
    fs_t1_brain = os.path.join(subjects_dir, subject_id, "mri", "brain.mgz")
    t1_brain = os.path.join(outdir, "t1_brain.nii.gz")
    cmd_1a = ["mri_convert", fs_t1_brain, t1_brain]
    FSWrapper(cmd_1a, shfile=fs_sh)()

    # Register diffusion to T1
    dif2anat_dat = os.path.join(outdir, "dif2anat.dat")
    dif2anat_mat = os.path.join(outdir, "dif2anat.mat")
    nodif_brain_reg = os.path.join(outdir, "nodif_brain_to_t1.nii.gz")
    cmd_1b = ["bbregister",
              "--s",      subject_id,
              "--mov",    nodif_brain,
              "--reg",    dif2anat_dat,
              "--fslmat", dif2anat_mat,
              "--dti",
              "--init-fsl",
              "--o",      nodif_brain_reg]
    FSWrapper(cmd_1b, subjects_dir=subjects_dir, shfile=fs_sh,
              add_fsl_env=True, fsl_sh=fsl_sh)()

    # Invert dif2anat transform
    m = numpy.loadtxt(dif2anat_mat)
    m_inv = numpy.linalg.inv(m)
    anat2dif_mat = os.path.join(outdir, "anat2dif.mat")
    numpy.savetxt(anat2dif_mat, m_inv)

    # -------------------------------------------------------------------------
    # STEP 2 - Create the masks for Probtrackx2

    # White matter mask
    aparc_aseg = os.path.join(subjects_dir, subject_id, "mri",
                              "aparc+aseg.mgz")
    wm_mask = os.path.join(outdir, "wm_mask.nii.gz")
    mri_binarize(
        inputfile=aparc_aseg,
        outputfile=wm_mask,
        match=None,
        wm=True,
        inv=False,
        fsconfig=fs_sh)

    # Stop mask is inverse of white matter mask
    stop_mask = os.path.join(outdir, "inv_wm_mask.nii.gz")
    mri_binarize(
        inputfile=aparc_aseg,
        outputfile=stop_mask,
        match=None,
        wm=True,
        inv=True,
        fsconfig=fs_sh)

    # Create seed mask
    seed_mask = wm_mask

    # Create target masks: the white surface
    white_surf = os.path.join(outdir, "white.asc")
    cmd_2a = ["mris_convert",
              "--combinesurfs",
              lh_surf,
              rh_surf,
              white_surf]
    FSWrapper(cmd_2a, subjects_dir=subjects_dir, shfile=fs_sh)()

    # -------------------------------------------------------------------------
    # STEP 7 - Run Probtrackx2
    probtrackx2(dir=outdir,
                forcedir=True,
                seedref=t1_brain,
                xfm=anat2dif_mat,
                invxfm=dif2anat_mat,
                samples=os.path.join(bedpostx_dir, "merged"),
                mask=nodif_brain_mask,
                seed=seed_mask,
                omatrix3=True,
                target3=white_surf,
                stop=stop_mask,
                nsamples=nsamples,
                nsteps=nsteps,
                steplength=steplength,
                loopcheck=loopcheck,
                cthr=cthr,
                fibthresh=fibthresh,
                distthresh=distthresh,
                sampvox=sampvox,
                pd=True,
                randfib=2,
                shfile=fsl_sh)
    coords = os.path.join(outdir, "coords_for_fdt_matrix3")
    weights = os.path.join(outdir, "fdt_matrix3.dot")

    return coords, weights
