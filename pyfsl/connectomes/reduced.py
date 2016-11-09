# -*- coding: utf-8 -*-

##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Compute the connectome of a given parcellation, like the FreeSurfer aparc+aseg
segmentation, using MRtrix or Probtrackx2.
"""

import os
import subprocess

import numpy
import nibabel

from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper
from pyfreesurfer.utils.filetools import (get_or_check_path_of_freesurfer_lut,
                                          get_or_check_freesurfer_subjects_dir,
                                          load_look_up_table)

from pyfsl import DEFAULT_FSL_PATH
from pyfsl.wrapper import FSLWrapper
from pyfsl.tractography.probabilist import probtrackx2


def get_region_names_of_lausanne_2008_atlas():
    """
    Return ordered region names of the Lausanne 2008 atlas.

    It corresponds to the Desikan atlas in the cortex, without the corpus
    callosum along with 7 subcortical regions.
    """

    # All left cortical regions of the Desikan atlas except the corpus callosum
    lh_ctx_rois = [
        'ctx-lh-lateralorbitofrontal',
        'ctx-lh-parsorbitalis',
        'ctx-lh-frontalpole',
        'ctx-lh-medialorbitofrontal',
        'ctx-lh-parstriangularis',
        'ctx-lh-parsopercularis',
        'ctx-lh-rostralmiddlefrontal',
        'ctx-lh-superiorfrontal',
        'ctx-lh-caudalmiddlefrontal',
        'ctx-lh-precentral',
        'ctx-lh-paracentral',
        'ctx-lh-rostralanteriorcingulate',
        'ctx-lh-caudalanteriorcingulate',
        'ctx-lh-posteriorcingulate',
        'ctx-lh-isthmuscingulate',
        'ctx-lh-postcentral',
        'ctx-lh-supramarginal',
        'ctx-lh-superiorparietal',
        'ctx-lh-inferiorparietal',
        'ctx-lh-precuneus',
        'ctx-lh-cuneus',
        'ctx-lh-pericalcarine',
        'ctx-lh-lateraloccipital',
        'ctx-lh-lingual',
        'ctx-lh-fusiform',
        'ctx-lh-parahippocampal',
        'ctx-lh-entorhinal',
        'ctx-lh-temporalpole',
        'ctx-lh-inferiortemporal',
        'ctx-lh-middletemporal',
        'ctx-lh-bankssts',
        'ctx-lh-superiortemporal',
        'ctx-lh-transversetemporal',
        'ctx-lh-insula'
    ]

    # Same for right hemisphere
    rh_ctx_rois = [x.replace("ctx-lh-", "ctx-rh-") for x in lh_ctx_rois]

    # Ordered left subcortical regions of Lausanne 2008 scale 33 atlas
    lh_subctx_rois = [
        'Left-Thalamus-Proper',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Accumbens-area',
        'Left-Hippocampus',
        'Left-Amygdala',
    ]

    # Ordered right subcortical regions
    rh_subctx_rois = [x.replace("Left-", "Right-") for x in lh_subctx_rois]

    # Non-hemispheric subcortical region
    axial_subctx_rois = ['Brain-Stem']

    atlas_names = (lh_ctx_rois + lh_subctx_rois + rh_ctx_rois +
                   rh_subctx_rois + axial_subctx_rois)

    return atlas_names


def create_lausanne2008_lut(outdir, freesurfer_lut=None):
    """
    Create a Look Up Table for the Lausanne2008 atlas. It has the same format
    as the FreeSurfer LUT (FreeSurferColorLUT.txt), but it lists only the
    regions of the Lausanne2008 atlas and the integer labels are the row/col
    positions of the regions in the connectome.

    Parameters
    ----------
    outdir: str
        Path to directory where to write "Lausanne2008LUT.txt"
    freesurfer_lut: str, default None
        Path to the FreeSurfer Look Up Table. If not passed, try to use
        $FREESURFER_HOME/FreeSurferColorLUT.txt. If not found raise Exception.
    """

    # Ordered ROIs (i.e. nodes of the connectome) of the Lausanne 2008 atlas
    roi_names = get_region_names_of_lausanne_2008_atlas()

    # Path to the FreeSurfer LUT
    freesurfer_lut = get_or_check_path_of_freesurfer_lut(freesurfer_lut)

    # Load table
    table = numpy.loadtxt(freesurfer_lut, dtype=str)

    # Keep rows that corresponds to regions of the atlas
    table = numpy.array([r for r in table if r[1] in set(roi_names)])

    # Order rows (i.e. regions) of the LUT like Lausanne2008 atlas
    table = numpy.array(sorted(table, key=lambda r: roi_names.index(r[1])))

    # Replace FreeSurfer label by row/col position in connectome
    table[:, 0] = numpy.arange(1, table.shape[0] + 1)

    # Header lines
    header_1 = "# Look up Table for Lausanne 2008 atlas\n"
    header_2 = "#<Label> <Label Name> <R> <G> <B> <A>\n"

    # Save as .txt file
    lausanne2008_lut = os.path.join(outdir, "Lausanne2008LUT.txt")
    with open(lausanne2008_lut, "w") as f:
        f.write(header_1)
        f.write(header_2)
        # Maintain the indentation
        line_format = "{0: <8} {1: <50} {2: <4} {3: <4} {4: <4} {5: <4}\n"
        for i, row in enumerate(table, start=1):
            f.write(line_format.format(*row))

    return lausanne2008_lut


def connectome_snapshot(connectome, snapshot, labels=None, transform=None,
                        colorbar_title="", dpi=200, labels_size=4):
    """
    Create a PNG snapshot of the connectogram (i.e. connectivity matrix).

    Parameters
    ----------
    connectome: str
        Path to txt file storing the connectivity matrix.
    snapshot: str
        Path to the output snapshot.
    labels: str, default None
        Path to txt file listing the label names. By default no labels.
        Should be ordered like the rows/cols of the connectivity matrix.
    transform: callable, default None
        A Callable function to apply on the matrix (e.g. numpy.log1p).
        By default no transformation is applied.
    colorbar_title: str, default ""
        How to interpret the values of the connectivity,
        e.g. "Log(# of tracks)" or "% of tracks"
    dpi: int, default 200
        "Dot Per Inch", set higher for better resolution.

    Returns
    -------
    path_out: str
        Path to the output connectogram snapshot.
    """

    # Import in function, so that the rest of the module can be used even
    # if matplotlib is not available
    import matplotlib.pyplot as plt

    # Load the connectivity matrix
    matrix = numpy.loadtxt(connectome)
    nrows, ncols = matrix.shape

    # Check connectivity matrix dimensions
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connectivity matrix should be a square matrix."
                         "Shape of matrix: {}".format(matrix.shape))

    # Apply transformation if requested
    if transform is not None:
        matrix = transform(matrix)

    # -------------------------------------------------------------------------
    # Create the figure with matplotlib

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Reds)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticks(numpy.arange(0.5, matrix.shape[0]))
    ax.set_yticks(numpy.arange(0.5, matrix.shape[1]))

    ax.tick_params(which="both", axis="both", width=0, length=0)

    # Add the labels if passed
    if labels is not None:
        labels = numpy.loadtxt(labels, dtype=str)

        if len(labels) != matrix.shape[0]:
            raise ValueError("Wrong number of labels: {}. Should be {}."
                             .format(len(labels), matrix.shape[0]))

        ax.set_xticklabels(labels, size=labels_size, rotation=90)
        ax.set_yticklabels(labels, size=labels_size)

    ax.set_aspect("equal")
    plt.tight_layout(rect=[0, 0, 1, 1])

    colorbar = fig.colorbar(heatmap)
    colorbar.set_label(colorbar_title, rotation=270, labelpad=20)

    fig.tight_layout()
    # -------------------------------------------------------------------------

    # Save to PNG file
    if not snapshot.endswith(".png"):
        snapshot += ".png"
    fig.savefig(snapshot, dpi=dpi)

    # Release memory
    fig.clear()
    plt.close()

    return snapshot


def mrtrix_extract_b0s_and_mean_b0(dwi, b0s, mean_b0, nb_threads=1):
    """
    Extract b=0 (bvalue=0) volumes from DWI and compute mean b=0 volume.
    """
    cmd_1 = ["dwiextract", "-bzero", dwi, b0s,
             "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_1)

    cmd_2 = ["mrmath", b0s, "mean", mean_b0, "-axis", "3",
             "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_2)

    return b0s, mean_b0


def fix_freesurfer_subcortical_parcellation(parc, t1_brain, lut, output,
                                            tempdir=None, nb_threads=None,
                                            fsl_sh=DEFAULT_FSL_PATH):
    """
    Use the MRtrix labelsgmfix command to correct the FreeSurfer parcellation.
    It uses FSL First to recompute 5 subcortical structures.

    Parameters
    ----------
    parc: str
        Path to the FreeSurfer parcellation, generally aparc+aseg or
        aparc.a2009s+aseg.
    t1_brain: str
        Path to the T1 brain-only image on which recompute the segmentation.
    lut: str
        Path to the Look Up Table. If you haven't change the labels it should
        be FreeSurfer LUT (FreeSurferColorLUT.txt in $FREESURFER_HOME dir).
    output: str
        Path to output fixed parcellation.
    tempdir: str, default None
        Directory that MRtrix will use as temporary directory.
    nb_threads: int, default None
        Number of threads that MRtrix is allowed to use.
    """
    cmd = ["labelsgmfix", parc, t1_brain, lut, output, "-premasked"]
    if tempdir is not None:
        cmd += ["-tempdir", tempdir]
    if nb_threads is not None:
        cmd += ["-nthreads", "%i" % nb_threads]
    fsl_process = FSLWrapper(cmd, env=os.environ, shfile=fsl_sh)
    fsl_process()

    return output


def mrtrix_connectome_pipeline(outdir,
                               tempdir,
                               subject_id,
                               dwi,
                               bvals,
                               bvecs,
                               t1_parc,
                               t1_parc_lut,
                               connectome_lut,
                               nb_threads,
                               global_tractography=False,
                               mtracks=None,
                               maxlength=None,
                               cutoff=None,
                               seed_gmwmi=False,
                               sift_mtracks=None,
                               sift2=False,
                               nodif_brain=None,
                               nodif_brain_mask=None,
                               fix_freesurfer_subcortical=False,
                               subjects_dir=None,
                               mif_gz=True,
                               delete_raw_tracks=False,
                               delete_dwi_mif=True,
                               snapshots=True,
                               fs_sh=DEFAULT_FREESURFER_PATH,
                               fsl_sh=DEFAULT_FSL_PATH):
    """
    Compute the connectome using MRtrix.

    Parameters
    ----------
    outdir: str
        Path to directory where to output.
    tempdir: str
        Path to the directory where temporary directories should be written.
        If you be on a partition with 5+ GB available.
    subject_id: str
        Subject identifier.
    dwi: str
        Path to the diffusion-weighted images (Nifti required).
    bvals: str
        Path to the bvalue list.
    bvecs: str
        Path to the list of diffusion-sensitized directions.
    t1_parc: str
        Path to the parcellation that defines the nodes of the connectome, e.g.
        aparc+aseg.mgz from FreeSurfer. It has to be in the FreeSurfer space
        (i.e. aligned with FreeSurfer "brain.mgz").
    t1_parc_lut: str
        Path to the Look Up Table for the passed parcellation in the
        FreeSurfer LUT format. If your T1 parcellation is from FreeSurfer,
        this will most likely be <$FREESURFER_HOME>/FreeSurferColorLUT.txt.
    connectome_lut: str
        Path to a Look Up Table in the FreeSurfer LUT format, listing the
        regions from the parcellation to use as nodes in the connectome. The
        region names should match the ones used in the <t1_parc_lut> LUT and
        the integer labels should be the row/col positions in the connectome.
        Alternatively it can be set to 'Lausanne2008' to use the predefined
        LUT for the Lausanne 2008 atlas, which is based on the FreeSurfer
        aparc+aseg parcellation.
    nb_threads: int
        Number of threads.
    global_tractography: bool, default False
        If True run global tractography (tckglobal) instead of local (tckgen).
    mtracks: int, default None
        For non-global tractography only. Number of millions of tracks of the
        raw tractogram.
    maxlength: int, default None
        For non-global tractography only. Max fiber length in mm.
    cutoff: float, default None
        For non-global tractography only.
        FOD amplitude cutoff, stopping criteria.
    seed_gmwmi: bool, default False
        For non-global tractography only.
        Set this option if you want to activate the '-seed_gmwmi' option of
        MRtrix 'tckgen', to seed from the GM/WM interface. Otherwise, and by
        default, the seeding is done in white matter ('-seed_dynamic' option).
    sift_mtracks: int, default None
        For non-global tractography only.
        Number of millions of tracks to keep with SIFT.
        If not set, SIFT is not applied.
    sift2: bool, default False
        For non-global tractography only.
        To activate SIFT2.
    nodif_brain: str, default None
        Diffusion brain-only Nifti volume with bvalue ~ 0. If not passed, it is
        generated automatically by averaging all the b0 volumes of the DWI.
    nodif_brain_mask: str, default None
        Path to the Nifti brain binary mask in diffusion. If not passed, it is
        created with MRtrix 'dwi2mask'.
    fix_freesurfer_subcortical: bool, default False
        If the <t1_parc> is aparc+aseg or aparc.a2009s+aseg from FreeSurfer,
        set this option to True, to recompute the subcortical segmentations
        of the 5 structures that are uncorrectly segmented by FreeSurfer,
        using FSL FIRST.
    subjects_dir: str, default None
        Path to the FreeSurfer subjects directory. Required if FreeSurfer is
        required and the environment variable (i.e. $SUBJECTS_DIR) is not set.
    mif_gz: bool, default True
        Use compressed MIF files (.mif.gz) instead of .mif to save space.
    delete_raw_tracks: bool, default False
        Delete the raw tracks (<outdir>/<mtracks>M.tck) at the end of
        processing, to save space.
    delete_dwi_mif: bool, default True
        Delete <outdir>/DWI.mif(.gz) at the end of processing, which is a copy
        of the input <dwi> in the MIF format, to save space.
    snapshots: bool, default True
        If True, create PNG snapshots for QC.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.
    """

    # STEP 0 - Check arguments

    # FreeSurfer $SUBJECTS_DIR has to be passed or set as an env variable
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    # Use compressed MIF files or not
    MIF_EXT = ".mif.gz" if mif_gz else ".mif"

    # Is SIFT to be applied
    sift = (sift_mtracks is not None)

    if connectome_lut.lower() == "lausanne2008":
        module_dir = os.path.dirname(os.path.abspath(__file__))
        connectome_lut = os.path.join(module_dir, "Lausanne2008LUT.txt")

    # Check input paths
    paths_to_check = [dwi, bvals, bvecs, t1_parc, t1_parc_lut, connectome_lut,
                      fsl_sh]

    # Optional paths that could be set
    for p in [nodif_brain, nodif_brain_mask]:
        if p is not None:
            paths_to_check.append(p)

    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Identify whether the DWI acquisition is single or multi-shell
    bvalues = numpy.loadtxt(bvals, dtype=int)
    is_multi_shell = len(set(bvalues)) - 1 > 1

    # Check compatibility of arguments
    if global_tractography:
        if not is_multi_shell:
            raise ValueError("MRtrix global tractography is only applicable "
                             "to multi shell data.")
        if seed_gmwmi:
            raise ValueError("'seed_gmwmi' cannot be applied when requesting "
                             "global tractography.")
        if sift or sift2:
            raise ValueError("SIFT or SIFT2 are not meant to be used with "
                             "global tractography.")
    else:
        value_of_required_arg = dict(mtracks=mtracks, maxlength=maxlength,
                                     cutoff=cutoff)
        for required_arg, value in value_of_required_arg.items():
            if value is None:
                raise ValueError("When 'global_tractography' is set to False "
                                 "%s is required." % required_arg)

    # Create <outdir> and/or <tempdir> if not existing
    for directory in [outdir, tempdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # -------------------------------------------------------------------------
    # STEP 1 - Format DWI and compute nodif brain and nodif brain mask if
    # not provided

    # Convert DWI to MRtrix format
    dwi_mif = os.path.join(outdir, "DWI.mif")
    cmd_1a = ["mrconvert", dwi, dwi_mif, "-fslgrad", bvecs, bvals,
              "-datatype", "float32", "-stride", "0,0,0,1",
              "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_1a)

    # If user has not provided a 'nodif_brain_mask', compute one with
    # MRtrix 'dwi2mask'
    if nodif_brain_mask is None:
        nodif_brain_mask = os.path.join(outdir, "nodif_brain_mask.nii.gz")
        cmd_1b = ["dwi2mask", dwi_mif, nodif_brain_mask]
        subprocess.check_call(cmd_1b)

    # If user has not provided a 'nodif_brain', apply 'nodif_brain_mask' to
    # mean b=0 volume
    if nodif_brain is None:
        # Extract b=0 volumes and compute mean b=0 volume
        b0s = os.path.join(outdir, "b0s.nii.gz")
        mean_b0 = os.path.join(outdir, "mean_b0.nii.gz")
        mrtrix_extract_b0s_and_mean_b0(dwi=dwi_mif, b0s=b0s, mean_b0=mean_b0,
                                       nb_threads=nb_threads)
        # Apply nodif_brain_mask to dwi
        nodif_brain = os.path.join(outdir, "nodif_brain.nii.gz")
        cmd_1b = ["mri_mask", mean_b0, nodif_brain_mask, nodif_brain]
        FSWrapper(cmd_1b, shfile=fs_sh)()

    # -------------------------------------------------------------------------
    # STEP 2 - Register DWI to T1 using FreeSurfer bbregister
    # - compute the rigid transformation
    # - apply transformation to align T1 and T1 parcellation with
    #   diffusion without downsampling

    # Register diffusion to T1
    dif2anat_dat = os.path.join(outdir, "dif2anat.dat")
    cmd_2a = ["bbregister",
              "--s",   subject_id,
              "--mov", nodif_brain,
              "--reg", dif2anat_dat,
              "--dti",
              "--init-fsl"]
    FSWrapper(cmd_2a, subjects_dir=subjects_dir, shfile=fs_sh,
              add_fsl_env=True, fsl_sh=fsl_sh)()

    # Align FreeSurfer T1 brain to diffusion without downsampling
    fs_t1_brain = os.path.join(subjects_dir, subject_id, "mri", "brain.mgz")
    t1_brain_to_dif = os.path.join(outdir, "t1_brain_to_dif.nii.gz")
    cmd_2b = ["mri_vol2vol",
              "--mov",  nodif_brain,
              "--targ", fs_t1_brain,
              "--inv",
              "--no-resample",
              "--o",    t1_brain_to_dif,
              "--reg",  dif2anat_dat,
              "--no-save-reg"]
    FSWrapper(cmd_2b, shfile=fs_sh)()

    # Align T1 parcellation to diffusion without downsampling
    parc_name = os.path.basename(t1_parc).split(".nii")[0].split(".mgz")[0]
    t1_parc_to_dif = os.path.join(outdir, parc_name + "_to_dif.nii.gz")
    cmd_2c = ["mri_vol2vol",
              "--mov",    nodif_brain,
              "--targ",   t1_parc,
              "--inv",
              "--no-resample",
              "--interp", "nearest",
              "--o",      t1_parc_to_dif,
              "--reg",    dif2anat_dat,
              "--no-save-reg"]
    FSWrapper(cmd_2c, shfile=fs_sh)()

    # -------------------------------------------------------------------------
    # STEP 3 - "5 tissue types" segmentation
    # Generate the 5TT image based on a FSL FAST
    five_tissues = os.path.join(outdir, "5TT%s" % MIF_EXT)
    cmd_3 = ["5ttgen", "fsl", t1_brain_to_dif, five_tissues, "-premasked",
             "-tempdir", tempdir, "-nthreads", "%i" % nb_threads]
    FSLWrapper(cmd_3, env=os.environ, shfile=fsl_sh)()

    # -------------------------------------------------------------------------
    # STEP 4 - Convert LUT
    # Change integer labels in the LUT so that the each label corresponds
    # to the row/col position in the connectome FreeSurfer
    nodes = os.path.join(outdir, "nodes%s" % MIF_EXT)
    cmd_4 = ["labelconvert", t1_parc_to_dif, t1_parc_lut, connectome_lut,
             nodes, "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_4)

    # -------------------------------------------------------------------------
    # STEP 5 - If the T1 parcellation is aparc+aseg or aparc.a2009s+aseg
    # from FreeSurfer, this option allows the recompute the subcortical
    # segmentations of 5 structures that are uncorrectly segmented by
    # FreeSurfer, using FSL FIRST
    if fix_freesurfer_subcortical:
        fixed_nodes = os.path.join(outdir, "nodes_fixSGM%s" % MIF_EXT)
        nodes = fix_freesurfer_subcortical_parcellation(
            parc=nodes,
            t1_brain=t1_brain_to_dif,
            lut=connectome_lut,
            output=fixed_nodes,
            tempdir=tempdir,
            nb_threads=nb_threads,
            fsl_sh=fsl_sh)

    # -------------------------------------------------------------------------
    # STEP 6 - Estimation of the response function of fibers in each voxel
    if is_multi_shell:
        rf_wm = os.path.join(outdir, "RF_WM.txt")
        rf_gm = os.path.join(outdir, "RF_GM.txt")
        rf_csf = os.path.join(outdir, "RF_CSF.txt")
        cmd_6 = ["dwi2response", "msmt_5tt", dwi_mif, five_tissues,
                 rf_wm, rf_gm, rf_csf]
    else:
        rf = os.path.join(outdir, "RF.txt")
        cmd_6 = ["dwi2response", "tournier", dwi_mif, rf]
    rf_voxels = os.path.join(outdir, "RF_voxels%s" % MIF_EXT)
    cmd_6 += ["-voxels", rf_voxels, "-tempdir", tempdir,
              "-nthreads", "%i" % nb_threads]
    subprocess.check_call(cmd_6)

    # -------------------------------------------------------------------------
    # STEP 7 - Compute FODs
    wm_fods = os.path.join(outdir, "WM_FODs%s" % MIF_EXT)
    if is_multi_shell:
        gm_mif = os.path.join(outdir, "GM%s" % MIF_EXT)
        csf_mif = os.path.join(outdir, "CSF%s" % MIF_EXT)
        cmd_7 = ["dwi2fod", "msmt_csd", dwi_mif, rf_wm, wm_fods,
                 rf_gm, gm_mif, rf_csf, csf_mif]
    else:
        cmd_7 = ["dwi2fod", "csd", dwi_mif, rf, wm_fods]
    cmd_7 += ["-mask", nodif_brain_mask, "-nthreads", "%i" % nb_threads,
              "-failonwarn"]
    subprocess.check_call(cmd_7)

    # -------------------------------------------------------------------------
    # STEP 8 - Image to visualize for multi-shell
    if is_multi_shell:
        wm_fods_vol0 = os.path.join(outdir, "WM_FODs_vol0%s" % MIF_EXT)
        cmd_8a = ["mrconvert", wm_fods, wm_fods_vol0, "-coord", "3", "0",
                  "-nthreads", "%i" % nb_threads]
        subprocess.check_call(cmd_8a)

        tissueRGB_mif = os.path.join(outdir, "tissueRGB%s" % MIF_EXT)
        cmd_8b = ["mrcat", csf_mif, gm_mif, wm_fods_vol0, tissueRGB_mif,
                  "-axis", "3", "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_8b)

    # -------------------------------------------------------------------------
    # STEP 9 - Tractography: tckglobal or tckgen

    if global_tractography:
        tracks = os.path.join(outdir, "global.tck")
        global_fod = os.path.join(outdir, "fod%s" % MIF_EXT)
        fiso_mif = os.path.join(outdir, "fiso%s" % MIF_EXT)
        cmd_9 = ["tckglobal", dwi_mif, rf_wm, "-riso", rf_csf, "-riso", rf_gm,
                 "-mask", nodif_brain_mask, "-niter", "1e8",
                 "-fod", global_fod, "-fiso", fiso_mif, tracks]
    else:
        # Anatomically Constrained Tractography:
        # iFOD2 algorithm with backtracking and crop fibers at GM/WM interface
        tracks = os.path.join(outdir, "%iM.tck" % mtracks)
        cmd_9 = ["tckgen", wm_fods, tracks, "-act", five_tissues, "-backtrack",
                 "-crop_at_gmwmi", "-maxlength", "%i" % maxlength,
                 "-number", "%dM" % mtracks, "-cutoff", "%f" % cutoff]

        # Requested seeding strategy: -seed_gmwmi or -seed_dynamic
        if seed_gmwmi:
            gmwmi_mask = os.path.join(outdir, "gmwmi_mask%s" % MIF_EXT)
            cmd_9b = ["5tt2gmwmi", five_tissues, gmwmi_mask]
            subprocess.check_call(cmd_9b)
            cmd_9 += ["-seed_gmwmi", gmwmi_mask]
        else:
            cmd_9 += ["-seed_dynamic", wm_fods]
    cmd_9 += ["-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_9)

    # -------------------------------------------------------------------------
    # STEP 10 - Filter tracts with SIFT if requested
    if sift:
        sift_tracks = os.path.join(outdir, "%iM_SIFT.tck" % sift_mtracks)
        cmd_10 = ["tcksift", tracks, wm_fods, sift_tracks,
                  "-act", five_tissues, "-term_number", "%iM" % sift_mtracks,
                  "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_10)

    # -------------------------------------------------------------------------
    # STEP 11 - run SIFT2 if requested (compute weights of fibers)
    if sift2:
        sift2_weights = os.path.join(outdir, "sift2_weights.txt")
        cmd_11 = ["tcksift2", tracks, wm_fods, sift2_weights,
                  "-act", five_tissues,
                  "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_11)

    # -------------------------------------------------------------------------
    # STEP 12 - Create the connectome(s) of the raw tractogram,
    # and for SIFT and SIFT2 if used.
    if global_tractography:
        raw_connectome = os.path.join(outdir, "tckglobal_connectome.csv")
    else:
        raw_connectome = os.path.join(outdir, "tckgen_connectome.csv")
    cmd_12a = ["tck2connectome", tracks, nodes, raw_connectome,
               "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_12a)

    if sift:
        sift_connectome = os.path.join(outdir, "sift_connectome.csv")
        cmd_12b = ["tck2connectome", sift_tracks, nodes, sift_connectome,
                   "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_12b)

    if sift2:
        sift2_connectome = os.path.join(outdir, "sift2_connectome.csv")
        cmd_12c = ["tck2connectome", tracks, nodes, sift2_connectome,
                   "-tck_weights_in", sift2_weights,
                   "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_12c)

    # -------------------------------------------------------------------------
    # STEP 13 - create snapshots of the connectomes

    # Read labels from LUT and create a list of labels: labels.txt
    labels = numpy.loadtxt(connectome_lut, dtype=str, usecols=[1])
    path_labels = os.path.join(outdir, "labels.txt")
    numpy.savetxt(path_labels, labels, fmt="%s")

    # Create plots with matplotlib if requested
    if snapshots:
        if global_tractography:
            raw_snapshot = os.path.join(outdir, "tckglobal_connectome.png")
        else:
            raw_snapshot = os.path.join(outdir, "tckgen_connectome.png")
        connectome_snapshot(raw_connectome, raw_snapshot, labels=path_labels,
                            transform=numpy.log1p,
                            colorbar_title="log(# tracks)")

        if sift:
            sift_snapshot = os.path.join(outdir, "sift_connectome.png")
            connectome_snapshot(sift_connectome, sift_snapshot,
                                labels=path_labels, transform=numpy.log1p,
                                colorbar_title="log(# tracks)")

        if sift2:
            sift2_snapshot = os.path.join(outdir, "sift2_connectome.png")
            connectome_snapshot(sift2_connectome, sift2_snapshot,
                                labels=path_labels, transform=numpy.log1p,
                                colorbar_title="log(# tracks)")

    # -------------------------------------------------------------------------
    # STEP 14 - clean if requested
    if delete_raw_tracks:
        os.remove(tracks)
        tracks = None

    if delete_dwi_mif:
        os.remove(dwi_mif)
        dwi_mif = None
    else:
        if mif_gz:
            subprocess.check_call(["gzip", dwi_mif])
            dwi_mif += ".gz"

    return outdir


def voxel_to_node_connectivity(probtrackx2_dir, nodes, connectome_lut, outdir,
                               basename="connectome"):
    """
    When using the --omatrix3 option in Probtrackx2, the result is a
    VOXELxVOXEL connectivity matrix. This function creates the NODExNODE
    (i.e. ROIxROI) connectivity matrix for the given parcellation.

    Parameters
    ----------
    probtrackx2_dir: str
        Path to dir where to find the files created by probtrackx2 when using
        --omatrix3 option, i.e "fdt_matrix3.dot" and "coords_for_fdt_matrix3"
    nodes: str
        Path to parcellation defining the nodes of the connectome,
        e.g. FreeSurfer aparc+aseg parcellation with only the regions
        (i.e. labels) to keep in the connectome.
    connectome_lut: str
        Path to the Look Up Table of the given parcellation in the
        FreeSurfer LUT format.
    outdir: str
        Path to directory where output.
    basename: str, default "connectome"
        Basename of output files (<outdir>/<basename>.[mat|labels]).
    """

    # Check input and output dirs
    if not os.path.isdir(probtrackx2_dir):
        raise ValueError("Directory does not exist: %s" % probtrackx2_dir)

    # coords: map x,y,z coordinates to voxel index
    # <x> <y> <z> <voxel index>
    path_coords = os.path.join(probtrackx2_dir, "coords_for_fdt_matrix3")
    coords = numpy.loadtxt(path_coords, dtype=int, usecols=[0, 1, 2, 4])

    # Load parcellation volume with node labels
    nodes_vol = nibabel.load(nodes).get_data().astype(dtype=int)

    # Load LUT to get the node names and set of int labels
    node_labels, node_names, _ = load_look_up_table(connectome_lut)
    set_labels = set(node_labels)

    # Connectivity matriw
    nb_nodes = len(node_names)
    connectome = numpy.zeros((nb_nodes, nb_nodes), dtype=int)

    # fdt_matrix3.dot: voxel to voxel connectivity
    # <index voxel 1> <index voxel 2> <nb connections>
    path_fdt_matrix3 = os.path.join(probtrackx2_dir, "fdt_matrix3.dot")

    # Since the fdt_matrix3.dot file can be very large, we parse it line by
    # line without loading it completely in memory
    with open(path_fdt_matrix3) as f:
        for line in f:

            # Get indexes of connected voxels and nb of connections
            v1_idx, v2_idx, nb_connections = map(int, line.strip().split())

            if nb_connections == 0:
                continue

            # Volume coordinates of connected voxels
            x1, y1, z1, _v1_idx = coords[v1_idx - 1, :]
            x2, y2, z2, _v2_idx = coords[v2_idx - 1, :]

            # Checking assumption that a voxel's info is stored at row=index-1
            assert v1_idx == _v1_idx, "%i %i" % (v1_idx, _v1_idx)
            assert v2_idx == _v2_idx, "%i %i" % (v2_idx, _v2_idx)

            # Labels of the 2 connected voxels
            label1, label2 = nodes_vol[x1, y1, z1], nodes_vol[x2, y2, z2]

            # Ignore pairs of voxels which labels are not in the connectome
            if not {label1, label2}.issubset(set_labels):
                continue

            # Update counts
            i, j = label1 - 1, label2 - 1  # 0-indexed in python
            connectome[i, j] += nb_connections
            connectome[j, i] += nb_connections

    # Write output connectome
    out_connectome = os.path.join(outdir, basename + ".mat")
    numpy.savetxt(out_connectome, connectome, fmt="%i")

    # Write nodes names
    out_labels = os.path.join(outdir, basename + ".labels")
    numpy.savetxt(out_labels, node_names, fmt="%s")

    return out_connectome, out_labels


def probtrackx2_connectome_pipeline(outdir,
                                    tempdir,
                                    subject_id,
                                    t1_parc,
                                    t1_parc_lut,
                                    connectome_lut,
                                    nodif_brain,
                                    nodif_brain_mask,
                                    bedpostx_dir,
                                    nsamples,
                                    nsteps,
                                    steplength,
                                    fix_freesurfer_subcortical=False,
                                    subjects_dir=None,
                                    loopcheck=True,
                                    cthr=0.2,
                                    fibthresh=0.01,
                                    distthresh=0.0,
                                    sampvox=0.0,
                                    snapshots=True,
                                    fs_sh=DEFAULT_FREESURFER_PATH,
                                    fsl_sh=DEFAULT_FSL_PATH):
    """
    Compute the connectome of a given parcellation, like the FreeSurfer
    aparc+aseg segmentation, using MRtrix.

    Requirements:
        - preprocessed DWI with bvals and bvecs: if distortions from
          acquisition have been properly corrected it should be possible
          to register diffusion to T1 with rigid transformation.
        - brain masks for the preprocessed DWI: nodif_brain and
          nodif_brain_mask
        - FreeSurfer: result of recon-all on the T1
        - FSL Bedpostx: computed for the preprocessed DWI
        - a T1 parcellation that defines the nodes of the connectome, it has
          to be in the FreeSurfer space (i.e. aligned with
          <subjects dir>/<subject>/mri/brain.mgz), e.g. aparc+aseg from
          FreeSurfer.

    Connectome construction strategy:
        - Pathways are constructed from 'constitutive points' and not from
          endpoints. A pathway is the result of 2 samples propagating in
          opposite directions from a seed point. It is done using the
          --omatrix3 option of Probtrackx2.
        - The seed mask is the mask of WM voxels that are neighbors
          (12-connexity) of nodes.
        - The stop mask is the inverse of white matter, i.e. a sample stops
          propagating as soon as it leaves the white matter.

    Parameters
    ----------
    outdir: str
        Directory where to output.
    tempdir: str
        Path to the directory where temporary directories should be written.
        If you be on a partition with 5+ GB available.
    subject_id: str
        Subject id used with FreeSurfer 'recon-all' command.
    t1_parc: str
        Path to the parcellation that defines the nodes of the connectome, e.g.
        aparc+aseg.mgz from FreeSurfer. Should be in the same space as the T1.
    t1_parc_lut: str
        Path to the Look Up Table for the passed parcellation in the
        FreeSurfer LUT format. If you T1 parcellation is from FreeSurfer, this
        will most likely be <$FREESURFER_HOME>/FreeSurferColorLUT.txt.
    connectome_lut: str
        Path to a Look Up Table in the FreeSurfer LUT format, listing the
        regions from the parcellation to use as nodes in the connectome. The
        region names should match the ones used in the <t1_parc_lut> LUT and
        the integer labels should be the row/col positions in the connectome.
        Alternatively it can be set to 'Lausanne2008' to use the predefined
        LUT for the Lausanne 2008 atlas, which is based on the FreeSurfer
        aparc+aseg parcellation.
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
    fix_freesurfer_subcortical: bool, default False
        If the <t1_parc> is aparc+aseg or aparc.a2009s+aseg from FreeSurfer,
        set this option to True, to recompute the subcortical segmentations
        of the 5 structures that are uncorrectly segmented by FreeSurfer,
        using FSL FIRST.
    subjects_dir: str or None, default None
        Path to the FreeSurfer subjects directory. Required if the FreeSurfer
        environment variable (i.e. $SUBJECTS_DIR) is not set.
    cthr: int, optional
        Probtrackx2 option.
    fibthresh, distthresh, sampvox: int, optional
        Probtrackx2 options.
    loopcheck: bool, optional
        Probtrackx2 option.
    snapshots: bool, default True
        If True, create PNG snapshots for QC.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.
    """

    # STEP 0 - Check arguments

    # FreeSurfer subjects_dir
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    if connectome_lut.lower() == "lausanne2008":
        module_dir = os.path.dirname(os.path.abspath(__file__))
        connectome_lut = os.path.join(module_dir, "Lausanne2008LUT.txt")

    # Check input paths
    paths_to_check = [t1_parc, t1_parc_lut, connectome_lut, nodif_brain,
                      nodif_brain_mask, bedpostx_dir, fsl_sh]
    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Create <outdir> and/or <tempdir> if not existing
    for directory in [outdir, tempdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

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
    cmd_1b = ["bbregister",
              "--s",      subject_id,
              "--mov",    nodif_brain,
              "--reg",    dif2anat_dat,
              "--fslmat", dif2anat_mat,
              "--dti",
              "--init-fsl"]
    FSWrapper(cmd_1b, subjects_dir=subjects_dir, shfile=fs_sh,
              add_fsl_env=True, fsl_sh=fsl_sh)()

    # anat2dif: invert dif2anat transform
    m = numpy.loadtxt(dif2anat_mat)
    m_inv = numpy.linalg.inv(m)
    anat2dif_mat = os.path.join(outdir, "anat2dif.mat")
    numpy.savetxt(anat2dif_mat, m_inv)

    # -------------------------------------------------------------------------
    # STEP 2 - Convert LUT
    # Change integer labels in the LUT so that the each label corresponds
    # to the row/col position in the connectome

    nodes = os.path.join(outdir, "nodes.nii.gz")
    cmd_2 = ["labelconvert", t1_parc, t1_parc_lut, connectome_lut, nodes,
             "-nthreads", "0", "-failonwarn"]
    subprocess.check_call(cmd_2)

    # -------------------------------------------------------------------------
    # STEP 3 - If the T1 parcellation is aparc+aseg or aparc.a2009s+aseg
    # from FreeSurfer, this option allows the recompute the subcortical
    # segmentations of 5 structures that are uncorrectly segmented by
    # FreeSurfer, using FSL FIRST

    if fix_freesurfer_subcortical:
        fixed_nodes = os.path.join(outdir, "nodes_fixSGM.nii.gz")
        nodes = fix_freesurfer_subcortical_parcellation(parc=nodes,
                                                        t1_brain=t1_brain,
                                                        lut=connectome_lut,
                                                        output=fixed_nodes,
                                                        tempdir=tempdir,
                                                        nb_threads=0,
                                                        fsl_sh=fsl_sh)

    # -------------------------------------------------------------------------
    # STEP 4 - Create the masks for Probtrackx2

    # White matter mask
    aparc_aseg = os.path.join(subjects_dir, subject_id, "mri/aparc+aseg.mgz")
    wm_mask = os.path.join(outdir, "wm_mask.nii.gz")
    cmd_4a = ["mri_binarize",
              "--i", aparc_aseg,
              "--o", wm_mask,
              "--wm"]
    FSWrapper(cmd_4a, shfile=fs_sh)()

    # Stop mask is inverse of white matter mask
    stop_mask = os.path.join(outdir, "inv_wm_mask.nii.gz")
    cmd_4b = ["mri_binarize",
              "--i", aparc_aseg,
              "--o", stop_mask,
              "--wm", "--inv"]
    FSWrapper(cmd_4b, shfile=fs_sh)()

    # Create seed mask: white matter voxels near nodes (target regions)

    # Create target mask: a mask of all nodes
    target_mask = os.path.join(outdir, "target_mask.nii.gz")
    cmd_4c = ["mri_binarize",
              "--i",   nodes,
              "--o",   target_mask,
              "--min", "1"]
    FSWrapper(cmd_4c, shfile=fs_sh)()

    # Dilate target mask by one voxel (12-connexity)
    target_mask_dil = os.path.join(outdir, "target_mask_dilated.nii.gz")
    cmd_4d = ["mri_morphology", target_mask, "dilate", "1", target_mask_dil]
    FSWrapper(cmd_4d, shfile=fs_sh)()

    # Intersect dilated target mask and white matter mask
    # -> white matter voxels neighbor (12-connectivity) to node voxels
    seed_mask = os.path.join(outdir, "wm_nodes_interface_mask.nii.gz")
    cmd_4e = ["mri_and", wm_mask, target_mask_dil, seed_mask]
    FSWrapper(cmd_4e, shfile=fs_sh)()

    # -------------------------------------------------------------------------
    # STEP 7 - Run Probtrackx2

    probtrackx2_dir = os.path.join(outdir, "probtrackx2")
    probtrackx2(dir=probtrackx2_dir,
                forcedir=True,
                seedref=t1_brain,
                xfm=anat2dif_mat,
                invxfm=dif2anat_mat,
                samples=os.path.join(bedpostx_dir, "merged"),
                mask=nodif_brain_mask,
                seed=seed_mask,
                omatrix3=True,
                target3=nodes,
                stop=stop_mask,
                nsamples=nsamples,
                nsteps=nsteps,
                steplength=steplength,
                loopcheck=loopcheck,
                cthr=cthr,
                fibthresh=fibthresh,
                distthresh=distthresh,
                sampvox=sampvox)

    # ------------------------------------------------------------------------
    # STEP 8 - Create NODExNODE connectivity matrix for nodes from <t1_parc>

    connectome, labels = voxel_to_node_connectivity(
        probtrackx2_dir=probtrackx2_dir,
        nodes=nodes,
        connectome_lut=connectome_lut,
        outdir=outdir)

    # ------------------------------------------------------------------------
    # STEP 9 - Create a connectome snapshot if requested

    if snapshots:
        snapshot = os.path.join(outdir, "connectome.png")
        connectome_snapshot(connectome, snapshot, labels=labels,
                            transform=numpy.log1p, dpi=300, labels_size=4,
                            colorbar_title="log(# of tracks)")
    return outdir
