# -*- coding: utf-8 -*-

##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Compute the connectome of a given parcellation, like the Freesurfer aparc+aseg
segmentation, using MRtrix.

Requirements:
    - preprocessed DWI with bval and bvec: if distortions from acquisition
      have been properly corrected it should be alignable to the T1 with a
      rigid transformation.
    - diffusion brain mask: nodif_brain_mask
    - parcellation: image of labeled regions, e.g. Freesurfer aparc+aseg

Connectogram strategy:
    <TO DO>
"""

import os
import subprocess

import numpy

from pyfsl.wrapper import FSLWrapper
from pyfreesurfer.wrapper import FSWrapper

import matplotlib.pyplot as plt


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


def get_or_check_freesurfer_subjects_dir(subjects_dir=None):
    """
    If 'subjects_dir' is passed, check whether the directory exists, otherwise
    look for the $SUBJECTS_DIR environment variable. If 'subjects_dir' is not
    passed and $SUBJECTS_DIR not in the environment, raise an Exception.
    """
    if subjects_dir is not None:
        if not os.path.isdir(subjects_dir):
            raise ValueError("Non existing directory: {}".format(subjects_dir))
    elif "SUBJECTS_DIR" in os.environ:
        subjects_dir = os.environ["SUBJECTS_DIR"]
    else:
        raise ValueError("Missing 'subjects_dir': set the $SUBJECTS_DIR "
                         "environment variable for Freesurfer or pass it "
                         "as an argument.")
    return subjects_dir


def get_path_of_freesurfer_lut():
    """
    """
    if "FREESURFER_HOME" in os.environ:
        freesurfer_home = os.environ["FREESURFER_HOME"]
        path_lut = os.path.join(freesurfer_home, "FreeSurferColorLUT.txt")
    else:
        raise Exception("Environment variable 'FREESURFER_HOME' is not set.")

    return path_lut


def create_lausanne2008_lut(path_out):
    """
    Create a Look Up Table for the Lausanne2008 atlas. It has the same format
    as the Freesurfer LUT ($FREESURFER_HOME/FreeSurferColorLUT.txt), but it
    lists only the regions of the Lausanne2008 atlas and the integer labels
    are the row/col positions of the regions in the connectome.
    """

    # Ordered ROIs (i.e. nodes of the connectome) of the Lausanne 2008 atlas
    roi_names = get_region_names_of_lausanne_2008_atlas()

    # Path to the Freesurfer LUT
    freesurfer_lut = get_path_of_freesurfer_lut()

    # Load table
    table = numpy.loadtxt(freesurfer_lut, dtype=str)

    # Keep rows that corresponds to regions of the atlas
    table = numpy.array([r for r in table if r[1] in set(roi_names)])

    # Order rows (i.e. regions) of the LUT like Lausanne2008 atlas
    table = numpy.array(sorted(table, key=lambda r: roi_names.index(r[1])))

    # Replace Freesurfer label by row/col position in connectome
    table[:, 0] = numpy.arange(1, table.shape[0] + 1)

    # Header lines
    header_1 = "# Look up Table for Lausanne 2008 atlas\n"
    header_2 = "#<Label> <Label Name> <R> <G> <B> <A>\n"

    # Save as .txt file
    with open(path_out, "w") as f:
        f.write(header_1)
        f.write(header_2)
        # Maintain the indentation
        line_format = "{0: <8} {1: <50} {2: <4} {3: <4} {4: <4} {5: <4}\n"
        for i, row in enumerate(table, start=1):
            f.write(line_format.format(*row))

    return path_out


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


# TODO: remove this function, replace by pyfreesurfer process
from pyfreesurfer.exceptions import FreeSurferRuntimeError
from pyfreesurfer.configuration import concat_environment


def run_freesurfer_cmd(cmd, subjects_dir=None, fsl_init="/etc/fsl/5.0/fsl.sh"):
    """
    To avoid repeating the code to run Freesurfer and check exitcode
    everywhere.
    Step:
        - add $SUBJECTS_DIR to the environment if requested
        - add FSL's environment if requested (some Freesurfer commands require
          FSL)
        - run the Freesurfer cmd
        - check exit code

    Parameters
    ----------
    cmd: list of str
        the command to run (subprocess like).
    subjects_dir: str, default None.
        To set the $SUBJECTS_DIR environment variable.
    add_fsl_env:  bool, default False
        To activate the FSL environment, required for commands like bbregister.
    fsl_init: str
        Path to the Bash script setting the FSL environment, if needed.
    """
    fs_process = FSWrapper(cmd)

    # Add FSL and current env to Freesurfer environment
    fsl_env = FSLWrapper([], env=os.environ, shfile=fsl_init).environment
    complete_env = concat_environment(fsl_env, fs_process.environment)
    fs_process.environment = complete_env

    if subjects_dir is not None:
        fs_process.environment["SUBJECTS_DIR"] = subjects_dir

    fs_process()  # Run
    if fs_process.exitcode != 0:
        raise FreeSurferRuntimeError(cmd[0], " ".join(cmd[1:]))

    return fs_process


def mrtrix_connectome_pipeline(outdir,
                               tempdir,
                               subject_id,
                               dwi,
                               bvals,
                               bvecs,
                               t1_brain_to_dif,
                               t1_parc,
                               t1_parc_lut,
                               connectome_lut,
                               mtracks,
                               maxlength,
                               cutoff,
                               nb_threads,
                               seed_gmwmi=False,
                               sift_mtracks=None,
                               nodif_brain=None,
                               nodif_brain_mask=None,
                               labelsgmfix=False,
                               subjects_dir=None,
                               mif_gz=True,
                               delete_raw_tracks=False,
                               delete_dwi_mif=True,
                               fsl_init="/etc/fsl/5.0/fsl.sh"):
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
    t1_brain_to_dif: str
        2 possibilities:
        - if you set this argument to 'freesurfer', the Freesurfer T1 brain
          (i.e. <subjects_dir>/<subject_id>/mri/brain.mgz) will be used and
          registered to diffusion with Freesurfer 'bbregister'. It implies
          that you either set 'subjects_dir' or have set $SUBJECTS_DIR
          environment variable.
        - if you don't want to provide/use Freesurfer, set to path of a T1
          brain-only volume registered to diffusion. It is not required that
          the T1 has the same spacing as the diffusion as long as they are
          registered (you can keep the T1 brain in its native resolution).
    t1_parc: str
        Path to the parcellation that defines the nodes of the connectome, e.g.
        aparc+aseg.mgz from Freesurfer. Should be in the same space as the T1.
    t1_parc_lut: str
        Path to the Look Up Table for the passed parcellation in the
        Freesurfer LUT format. If you T1 parcellation is from Freesurfer, this
        will most likely be <$FREESURFER_HOME>/FreeSurferColorLUT.txt.
    connectome_lut: str
        2 possibilities:
        - set to 'Lausanne2008', a predefined LUT for Freesurfer aparc+aseg
          parcellation (Lausanne et al. 2008 atlas).
        - set to the path to a Look Up Table in the Freesurfer LUT format,
          listing the regions from the parcellation to use as nodes in the
          connectome. The region names should match the ones used in the
          't1_parc_lut' and the integer labels should be the row/col positions
          in the connectome.
    mtracks: int
        Number of millions of tracks of the raw tractography.
    maxlength: int
        Max fiber length in mm.
    cutoff: float
        FOD amplitude cutoff, stopping criteria.
    nb_threads: int
        Number of threads.
    seed_gmwmi: bool, default False
        Set this option if you want to activate the '-seed_gmwmi' option of
        MRtrix 'tckgen', to seed from the GM/WM interface. Otherwise, and by
        default, the seeding is done in white matter ('-seed_dynamic' option).
    sift_mtracks: int, default None
        Number of millions of tracks to keep with SIFT.
        If not set, SIFT is not applied
    nodif_brain: str, default None
        Diffusion brain-only volume with bvalue ~ 0. If not passed, it is
        generated automatically by averaging all the b0 volumes of the DWI.
        It is only used if a registration between diffusion and T1 is needed
        (i.e. if you have set 't1_brain_to_dif' to 'freesurfer').
    nodif_brain_mask: str, default None
        Path to the brain binary mask in diffusion. If not passed, it is
        created with MRtrix 'dwi2mask'.
    labelsgmfix: bool, default False
        If the <t1_parc> is aparc+aseg or aparc.a2009s+aseg from Freesurfer,
        set this option to True, to recompute the subcortical segmentations
        of the 5 structures that are uncorrectly segmented by Freesurfer,
        using FSL FIRST.
    subjects_dir: str, default None
        Path to the Freesurfer subjects directory. Required if Freesurfer is
        required and the environment variable (i.e. $SUBJECTS_DIR) is not set.
    mif_gz: bool, default True
        Use compressed MIF files (.mif.gz) instead of .mif to save space.
    delete_raw_tracks: bool, default False
        Delete the raw tracks (<outdir>/<mtracks>M.tck) at the end of
        processing, to save space.
    delete_dwi_mif: bool, default True
        Delete <outdir>/DWI.mif(.gz) at the end of processing, which is a copy
        of the input <dwi> in the MIF format, to save space.
    fsl_init: str, optional.
        Path to the Bash script setting the FSL environment.
    """

    # Create <outdir> and <tempdir> if they don't exist
    for directory in [outdir, tempdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # Use compressed MIF files or not
    MIF_EXT = ".mif.gz" if mif_gz else ".mif"

    # Will Freesurfer be used
    use_freesurfer = (t1_brain_to_dif.lower() == "freesurfer")

    # Is SIFT to be applied
    apply_sift = (sift_mtracks is not None)

    if connectome_lut.lower() == "lausanne2008":
        lausanne2008_lut = os.path.join(outdir, "Lausanne2008LUT.txt")
        connectome_lut = create_lausanne2008_lut(lausanne2008_lut)

    # -------------------------------------------------------------------------
    # Check input paths

    paths_to_check = [outdir, tempdir, dwi, bvals, bvecs, t1_parc, t1_parc_lut,
                      connectome_lut]

    # Optional paths that could be set
    for p in [nodif_brain, nodif_brain_mask]:
        if p is not None:
            paths_to_check.append(p)

    if use_freesurfer:
        # Freesurfer $SUBJECTS_DIR has to be passed or set as an env variable
        subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)
        paths_to_check.append(subjects_dir)
    else:
        paths_to_check.append(t1_brain_to_dif)

    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # -------------------------------------------------------------------------

    # STEP 0

    # Identify whether the DWI acquisition is single or multi-shell
    bvalues = numpy.loadtxt(bvals, dtype=int)
    is_multi_shell = len(set(bvalues)) - 1 > 1

    # convert DWI to MRtrix desired format
    dwi_mif = os.path.join(outdir, "DWI.mif")
    cmd_0a = ["mrconvert", dwi, dwi_mif, "-fslgrad", bvecs, bvals,
              "-datatype", "float32", "-stride", "0,0,0,1",
              "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_0a)

    # Extract mean B0 (bvalue = 0) volume
    mean_b0 = os.path.join(outdir, "meanb0%s" % MIF_EXT)
    b0s = os.path.join(outdir, "b0s%s" % MIF_EXT)
    cmd_0b = ["dwiextract", "-bzero", dwi_mif, b0s,
              "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_0b)
    cmd_0c = ["mrmath", b0s, "mean", mean_b0, "-axis", "3",
              "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_0c)

    # If user has not provided a 'nodif_brain' use mean B0
    if nodif_brain is None:
        nodif_brain = mean_b0

    # If user has not provided a 'nodif_brain_mask', create one with 'dwi2mask'
    if nodif_brain_mask is None:
        nodif_brain_mask = os.path.join(outdir, "nodif_brain_mask%s" % MIF_EXT)
        cmd_0d = ["dwi2mask", dwi_mif, nodif_brain_mask]
        subprocess.check_call(cmd_0d)

    # -------------------------------------------------------------------------
    # STEP 1 - If T1 and DWI are not already registered, compute the rigid
    # transformation and align T1 and T1 parcellation with diffusion without
    # downsampling T1 and T1 parcellation to diffusion

    if use_freesurfer:

        # Register diffusion to T1
        dif2anat_dat = os.path.join(outdir, "dif2anat.dat")
        cmd_1a = ["bbregister",
                  "--s",   subject_id,
                  "--mov", nodif_brain,
                  "--reg", dif2anat_dat,
                  "--dti",
                  "--init-fsl"]
        run_freesurfer_cmd(cmd_1a, subjects_dir=subjects_dir)

        # Align Freesurfer T1 brain to diffusion without downsampling
        fs_t1 = os.path.join(subjects_dir, subject_id, "mri", "brain.mgz")
        t1_brain_to_dif = os.path.join(outdir, "t1_brain_to_dif.nii.gz")
        cmd_1b = ["mri_vol2vol",
                  "--mov",  nodif_brain,
                  "--targ", fs_t1,
                  "--inv",
                  "--no-resample",
                  "--o",    t1_brain_to_dif,
                  "--reg",  dif2anat_dat,
                  "--no-save-reg"]
        run_freesurfer_cmd(cmd_1b)

        # Align T1 parcellation to diffusion without downsampling
        parc_name = os.path.basename(t1_parc).split(".nii")[0].split(".mgz")[0]
        t1_parc_to_dif = os.path.join(outdir, parc_name + "_to_dif.nii.gz")
        cmd_1c = ["mri_vol2vol",
                  "--mov",    nodif_brain,
                  "--targ",   t1_parc,
                  "--inv",
                  "--no-resample",
                  "--interp", "nearest",
                  "--o",      t1_parc_to_dif,
                  "--reg",    dif2anat_dat,
                  "--no-save-reg"]
        run_freesurfer_cmd(cmd_1c)
    else:
        t1_parc_to_dif = t1_parc

    # -------------------------------------------------------------------------
    # STEP X - QC the T1/parcellation/DWI alignment by creating snapshots

#    path_snapshot_1 = os.path.join(qc_dir, "t1_dwi_alignment.pdf")
#    qc_vol2vol_alignment(path_ref=nodif_brain,
#                         path_edge=t1_brain_to_dif,
#                         path_out=path_snapshot_1,
#                         title="T1 - DWI alignment")
#
#    path_snapshot_2 = os.path.join(qc_dir, "t1_parc_dwi_alignment.pdf")
#    qc_vol2vol_alignment(path_ref=nodif_brain,
#                         path_edge=t1_parc_to_dif,
#                         path_out=path_snapshot_2,
#                         title="T1 parcellation - DWI alignment")

    # -------------------------------------------------------------------------

    # STEP 2 - "5 tissue types" segmentation
    # Generate the 5TT image based on a FSL FAST
    five_tissues = os.path.join(outdir, "5TT%s" % MIF_EXT)
    cmd_2 = ["5ttgen", "fsl", t1_brain_to_dif, five_tissues, "-premasked",
             "-tempdir", tempdir, "-nthreads", "%i" % nb_threads]
    fsl_process = FSLWrapper(cmd_2, env=os.environ, shfile=fsl_init)
    fsl_process()

    # STEP 3 - Convert LUT
    # Change integer labels in the LUT so that the each label corresponds
    # to the row/col position in the connectome freesurfer
    nodes = os.path.join(outdir, "nodes%s" % MIF_EXT)
    cmd_3 = ["labelconvert", t1_parc_to_dif, t1_parc_lut, connectome_lut,
             nodes, "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_3)

    # STEP 4 - If the T1 parcellation is aparc+aseg or aparc.a2009s+aseg
    # from Freesurfer, this option allows the recompute the subcortical
    # segmentations of 5 structures that are uncorrectly segmented by
    # Freesurfer, using FSL FIRST
    if labelsgmfix:
        fixed_nodes = os.path.join(outdir, "nodes_fixSGM%s" % MIF_EXT)
        cmd_4 = ["labelsgmfix", nodes, t1_brain_to_dif, connectome_lut,
                 fixed_nodes, "-premasked", "-tempdir", tempdir,
                 "-nthreads", "%i" % nb_threads]
        fsl_process = FSLWrapper(cmd_4, env=os.environ, shfile=fsl_init)
        fsl_process()
        nodes = fixed_nodes

    # STEP 7 - Estimation of the response function of fibers in each voxel
    if is_multi_shell:
        rf_wm = os.path.join(outdir, "RF_WM.txt")
        rf_gm = os.path.join(outdir, "RF_GM.txt")
        rf_csf = os.path.join(outdir, "RF_CSF.txt")
        cmd_7 = ["dwi2response", "msmt_5tt", dwi_mif, five_tissues,
                 rf_wm, rf_gm, rf_csf]
    else:
        rf = os.path.join(outdir, "RF.txt")
        cmd_7 = ["dwi2response", "tournier", dwi_mif, rf]
    rf_voxels = os.path.join(outdir, "RF_voxels%s" % MIF_EXT)
    cmd_7 += ["-voxels", rf_voxels, "-tempdir", tempdir,
              "-nthreads", "%i" % nb_threads]
    subprocess.check_call(cmd_7)

    # STEP 8 - Compute FODs
    wm_fods = os.path.join(outdir, "WM_FODs%s" % MIF_EXT)
    if is_multi_shell:
        gm_mif = os.path.join(outdir, "GM%s" % MIF_EXT)
        csf_mif = os.path.join(outdir, "CSF%s" % MIF_EXT)
        cmd_8 = ["dwi2fod", "msmt_csd", dwi_mif, rf_wm, wm_fods,
                 rf_gm, gm_mif, rf_csf, csf_mif]
    else:
        cmd_8 = ["dwi2fod", "csd", dwi_mif, rf, wm_fods]
    cmd_8 += ["-mask", nodif_brain_mask, "-nthreads", "%i" % nb_threads,
              "-failonwarn"]
    subprocess.check_call(cmd_8)

    # STEP 9 - Image to visualize
    if is_multi_shell:
        wm_fods_vol0 = os.path.join(outdir, "WM_FODs_vol0%s" % MIF_EXT)
        cmd_9a = ["mrconvert", wm_fods, wm_fods_vol0, "-coord", "3", "0",
                  "-nthreads", "%i" % nb_threads]
        subprocess.check_call(cmd_9a)

        tissueRGB_mif = os.path.join(outdir, "tissueRGB%s" % MIF_EXT)
        cmd_9b = ["mrcat", csf_mif, gm_mif, wm_fods_vol0, tissueRGB_mif,
                  "-axis", "3", "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_9b)

    # STEP 10 - Tractography
    tracks = os.path.join(outdir, "%iM.tck" % mtracks)
    cmd_10 = ["tckgen", wm_fods, tracks, "-act", five_tissues, "-backtrack",
              "-crop_at_gmwmi", "-maxlength", "%i" % maxlength,
              "-number", "%dM" % mtracks, "-cutoff", "%f" % cutoff,
              "-nthreads", "%i" % nb_threads, "-failonwarn"]
    if seed_gmwmi:
        gmwmi_mask = os.path.join(outdir, "gmwmi_mask.%s" % MIF_EXT)
        cmd_10b = ["5tt2gmwmi", five_tissues, gmwmi_mask]
        subprocess.check_call(cmd_10b)
        cmd_10 += ["-seed_gmwmi", gmwmi_mask]
    else:
        cmd_10 += ["-seed_dynamic", wm_fods]
    subprocess.check_call(cmd_10)

    # STEP 11 - Filter tracts with SIFT if requested
    if apply_sift:
        sifted_tracks = os.path.join(outdir, "%iM_SIFT.tck" % sift_mtracks)
        cmd_11 = ["tcksift", tracks, wm_fods, sifted_tracks,
                  "-act", five_tissues, "-term_number", "%iM" % sift_mtracks,
                  "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_11)

    # STEP 12 - Create connectome(s) array(s) from tracts for raw tractogram
    # and for SIFT tractogram, if SIFT was applied
    raw_connectome = os.path.join(outdir, "raw_connectome.csv")
    cmd_12a = ["tck2connectome", tracks, nodes, raw_connectome,
               "-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_12a)

    if apply_sift:
        sifted_connectome = os.path.join(outdir, "sifted_connectome.csv")
        cmd_12b = ["tck2connectome", sifted_tracks, nodes, sifted_connectome,
                   "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_12b)

    # STEP 13 - create snapshots of the connectomes

    # Read labels from LUT and create a list of labels: labels.txt
    labels = numpy.loadtxt(connectome_lut, dtype=str, usecols=[1])
    path_labels = os.path.join(outdir, "labels.txt")
    numpy.savetxt(path_labels, labels, fmt="%s")

    # Create plots with matplotlib
    raw_snapshot = os.path.join(outdir, "raw_connectome.png")
    connectome_snapshot(raw_connectome, raw_snapshot, labels=path_labels,
                        transform=numpy.log1p, colorbar_title="log(# tracks)")

    if apply_sift:
        sifted_snapshot = os.path.join(outdir, "sifted_connectome.png")
        connectome_snapshot(sifted_connectome, sifted_snapshot,
                            labels=path_labels, transform=numpy.log1p,
                            colorbar_title="log(# tracks)")

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
