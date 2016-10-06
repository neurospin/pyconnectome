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


def mrtrix_connectome_pipeline(t1_brain, dwi, bvals, bvecs, nodif_brain_mask,
                               parc, parc_lut, connectome_lut, outdir, tempdir,
                               mtracks, sift_mtracks, maxlength, cutoff,
                               nthreads, fsl_init="/etc/fsl/5.0/fsl.sh"):
    """
    Temporary pipeline for HCP, to be generalized.

    Parameters
    ----------
    t1_brain: str
        Path to brain-only T1 image.
    dwi: str
        Path to the diffusion-weighted images (Nifti required).
    bvals: str
        Path to the bvalue list.
    bvecs: str
        Path to the list of diffusion-sensitized directions.
    nodif_brain_mask: str
        Path to the brain binary mask.
    parc: str
        Path to the parcellation that defines the nodes of the connectome,
        e.g. 'aparc+aseg' from Freesurfer.
    parc_lut: str
        Path to the Look Up Table for the passed parcellation in the Freesurfer
        LUT format.
    connectome_lut: str
        Path to a Look Up Table, in the Freesurfer LUT format, listing the
        regions from the parcellation to use as nodes in the connectome. The
        integer label should be the row/col position in the connectome.
    outdir: str
        Path to directory where to output.
    tempdir: str
        Path to the directory where temporary directories should be written.
    mtracks: int
        Number of millions of tracks to compute.
    sift_mtracks: int
        Number of millions of tracks to keep with SIFT.
    maxlength: int
        Max fiber length in mm.
    cutoff: float
        FOD amplitude cutoff, stopping criteria.
    nthreads: int
        Number of threads.
    fsl_init: str, default "/etc/fsl/5.0/fsl.sh".
        Path to the Bash script setting the FSL environment.
    """

    # Create <tempdir> and <outdir> if don't exist
    for directory in [tempdir, outdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # Use Lausanne2008 LUT if no custom LUT has been passed
    if connectome_lut is None:
        lausanne2008_lut = os.path.join(outdir, "Lausanne2008LUT.txt")
        connectome_lut = create_lausanne2008_lut(lausanne2008_lut)

    # STEP 1 - "5 tissue types" segmentation
    # Generate the 5TT image based on a FSL FAST
    five_tissues = os.path.join(outdir, "5TT.mif")
    cmd_1 = ["5ttgen", "fsl", t1_brain, five_tissues, "-premasked",
             "-tempdir", tempdir, "-nthreads", "%i" % nthreads]
    fsl_process = FSLWrapper(cmd_1, env=os.environ, shfile=fsl_init)
    fsl_process()

    # STEP 2 - Convert LUT
    # Change integer labels in the LUT so that the each label corresponds
    # to the row/col position in the connectome freesurfer
    nodes = os.path.join(outdir, "nodes.mif")
    cmd_2 = ["labelconvert", parc, parc_lut, connectome_lut, nodes,
             "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_2)

    # STEP 3 - Replace 5 subcortical segmentations from Freesurfer by
    # the ones from FSL FIRST (considered as better ones)
    fixed_nodes = os.path.join(outdir, "nodes_fixSGM.mif")
    cmd_3 = ["labelsgmfix", nodes, t1_brain, connectome_lut, fixed_nodes,
             "-premasked", "-tempdir", tempdir, "-nthreads", "%i" % nthreads]
    fsl_process = FSLWrapper(cmd_3, env=os.environ, shfile=fsl_init)
    fsl_process()

    # STEP 4 - convert DWI to MRtrix desired format
    dwi_mif = os.path.join(outdir, "DWI.mif")
    cmd_4 = ["mrconvert", dwi, dwi_mif, "-fslgrad", bvecs, bvals,
             "-datatype", "float32", "-stride", "0,0,0,1",
             "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_4)

    # STEP 5 - Extract mean B0 volume
    mean_b0 = os.path.join(outdir, "meanb0.mif")
    b0s = os.path.join(outdir, "b0s.mif")
    cmd_5a = ["dwiextract", "-bzero", dwi_mif, b0s,
              "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_5a)
    cmd_5b = ["mrmath", b0s, "mean", mean_b0, "-axis", "3",
              "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_5b)

    # STEP 6 -Estimation of response function ?
    rf_wm = os.path.join(outdir, "RF_WM.txt")
    rf_gm = os.path.join(outdir, "RF_GM.txt")
    rf_csf = os.path.join(outdir, "RF_CSF.txt")
    rf_voxels = os.path.join(outdir, "RF_voxels.mif")
    cmd_6 = ["dwi2response", "msmt_5tt", dwi_mif, five_tissues, rf_wm,
             rf_gm, rf_csf, "-voxels", rf_voxels,
             "-tempdir", tempdir, "-nthreads", "%i" % nthreads]
    subprocess.check_call(cmd_6)

    # STEP 7 - Compute FODs
    wm_fods = os.path.join(outdir, "WM_FODs.mif")
    gm_mif = os.path.join(outdir, "GM.mif")
    csf_mif = os.path.join(outdir, "CSF.mif")
    cmd_7 = ["dwi2fod", "msmt_csd", dwi_mif, rf_wm, wm_fods, rf_gm, gm_mif,
             rf_csf, csf_mif, "-mask", nodif_brain_mask,
             "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_7)

    # STEP 8 - Image to visualize
    wm_fods_vol0 = os.path.join(outdir, "WM_FODs_vol0.mif")
    cmd_8a = ["mrconvert", wm_fods, wm_fods_vol0, "-coord", "3", "0",
              "-nthreads", "%i" % nthreads]
    subprocess.check_call(cmd_8a)

    tissueRGB_mif = os.path.join(outdir, "tissueRGB.mif")
    cmd_8b = ["mrcat", csf_mif, gm_mif, wm_fods_vol0, tissueRGB_mif,
              "-axis", "3", "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_8b)

    # STEP 9 - Tractography
    tracks = os.path.join(outdir, "%iM.tck" % mtracks)
    cmd_9 = ["tckgen", wm_fods, tracks, "-act", five_tissues, "-backtrack",
             "-crop_at_gmwmi", "-seed_dynamic", wm_fods,
             "-maxlength", "%i" % maxlength, "-number", "%dM" % mtracks,
             "-cutoff", "%f" % cutoff, "-nthreads", "%i" % nthreads,
             "-failonwarn"]
    subprocess.check_call(cmd_9)

    # STEP 10 - Filter tracts
    sifted_tracks = os.path.join(outdir, "%iM_SIFT.tck" % sift_mtracks)
    cmd_10 = ["tcksift", tracks, wm_fods, sifted_tracks, "-act", five_tissues,
              "-term_number", "%iM" % sift_mtracks,
              "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_10)

    # STEP 11 - Create connectome array from tracts
    connectome = os.path.join(outdir, "connectome.csv")
    cmd_11 = ["tck2connectome", sifted_tracks, fixed_nodes, connectome,
              "-nthreads", "%i" % nthreads, "-failonwarn"]
    subprocess.check_call(cmd_11)

    return outdir
