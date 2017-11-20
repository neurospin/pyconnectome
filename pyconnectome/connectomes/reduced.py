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
segmentation, using MRtrix, FSL Probtrackx2 or MITK Gibbs Tracking.
"""

# Standard
import os
import subprocess

# Package
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.tractography.probabilist import probtrackx2
from pyconnectome.utils.segtools import fix_freesurfer_subcortical_parcellation
from pyconnectome.utils.filetools import convert_mitk_vtk_fibers_to_tck
from pyconnectome.utils.filetools import convert_trk_fibers_to_tck
from pyconnectome.utils.filetools import convert_probtrackx2_saved_paths_to_tck
from pyconnectome.utils.regtools import freesurfer_bbregister_t1todif

# Third-party
import numpy
import nibabel
from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper
from pyfreesurfer.utils.filetools import (get_or_check_freesurfer_subjects_dir,
                                          load_look_up_table)


def connectome_snapshot(connectome, snapshot, labels=None, transform=None,
                        colorbar_title="", dpi=200, labels_size=4,
                        vmin=None, vmax=None):
    """
    Create a PNG snapshot of the connectome (i.e. connectivity matrix).

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
    labels_size: int, default 4
        The label font size.
    vmin, vmax: float, default None
        The display range.

    Returns
    -------
    snapshot: str
        Path to the output connectome snapshot.
    """
    # Import in function, so that the rest of the module can be used even
    # if matplotlib is not available
    import matplotlib.pyplot as plt
    # Load the connectivity matrix
    matrix = numpy.loadtxt(connectome)

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
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticks(numpy.arange(0.5, matrix.shape[0]))
    ax.set_yticks(numpy.arange(0.5, matrix.shape[1]))

    ax.tick_params(which="both", axis="both", width=0, length=0)

    # Add the labels if passed
    if labels is not None:
        labels_array = numpy.loadtxt(labels, dtype=str)

        if len(labels_array) != matrix.shape[0]:
            raise ValueError(
                "Wrong number of labels: {}. Should be {}.".format(
                    len(labels_array), matrix.shape[0]))

        ax.set_xticklabels(labels_array, size=labels_size, rotation=90)
        ax.set_yticklabels(labels_array, size=labels_size)

    ax.set_aspect("equal")
    kwargs = {}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Reds, **kwargs)
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

    Returns
    -------
    out_connectome: str
        The generated connectome.
    out_labels: str
        The coonectome associated labels.
    """
    # Check input and output dirs
    for directory in (probtrackx2_dir, outdir):
        if not os.path.isdir(directory):
            raise ValueError("Directory does not exist: %s" % directory)

    # Coords: map x,y,z coordinates to voxel index
    # <x> <y> <z> <voxel index>
    path_coords = os.path.join(probtrackx2_dir, "coords_for_fdt_matrix3")
    coords = numpy.loadtxt(path_coords, dtype=int, usecols=[0, 1, 2, 4])

    # Load parcellation volume with node labels
    nodes_vol = nibabel.load(nodes).get_data().astype(dtype=int)

    # Load LUT to get the node names and set of int labels
    node_labels, node_names, _ = load_look_up_table(connectome_lut)
    set_labels = set(node_labels)

    # Connectivity matrix
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


def probtrackx2_connectome(
        outdir,
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
    aparc+aseg segmentation, using ProbTrackx2.

    Requirements:
        - brain masks for the preprocessed DWI: nodif_brain and
          nodif_brain_mask.
        - FreeSurfer: result of recon-all on the T1.
        - FSL Bedpostx: computed for the preprocessed DWI.
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
        It should be a partition with 5+ GB available.
    subject_id: str
        Subject id used with FreeSurfer 'recon-all' command.
    t1_parc: str
        Path to the parcellation that defines the nodes of the connectome, e.g.
        aparc+aseg.mgz from FreeSurfer.
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
    cthr: float, optional
        Probtrackx2 option.
    fibthresh, distthresh, sampvox: float, optional
        Probtrackx2 options.
    loopcheck: bool, optional
        Probtrackx2 option.
    snapshots: bool, default True
        If True, create PNG snapshots for QC.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.

    Returns
    -------
    connectome_file: str
        The generated connectome.
    labels_file: str
        The coonectome associated labels.
    connectome_snap_file: str
        A grphical representation of the connectome.
    """
    # -------------------------------------------------------------------------
    # STEP 0 - Check arguments

    # FreeSurfer subjects_dir
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    if connectome_lut.lower() == "lausanne2008":
        module_dir = os.path.dirname(os.path.abspath(__file__))
        connectome_lut = os.path.join(module_dir, "Lausanne2008LUT.txt")

    # Check input paths
    paths_to_check = [t1_parc, t1_parc_lut, connectome_lut, nodif_brain,
                      nodif_brain_mask, bedpostx_dir, fs_sh, fsl_sh]
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
    _, dif2anat_dat, dif2anat_mat = freesurfer_bbregister_t1todif(
            outdir=outdir,
            subject_id=subject_id,
            nodif_brain=nodif_brain,
            subjects_dir=subjects_dir,
            fs_sh=fs_sh,
            fsl_sh=fsl_sh)

    # Invert dif2anat transform
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
    aparc_aseg = os.path.join(subjects_dir, subject_id, "mri",
                              "aparc+aseg.mgz")
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

    # Create seed mask: white matter voxels near nodes (target regions)
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
                sampvox=sampvox,
                shfile=fsl_sh)

    # ------------------------------------------------------------------------
    # STEP 8 - Create NODExNODE connectivity matrix for nodes from <t1_parc>
    connectome_file, labels_file = voxel_to_node_connectivity(
        probtrackx2_dir=probtrackx2_dir,
        nodes=nodes,
        connectome_lut=connectome_lut,
        outdir=outdir)

    # ------------------------------------------------------------------------
    # STEP 9 - Create a connectome snapshot if requested
    if snapshots:
        connectome_snap_file = os.path.join(outdir, "connectome.png")
        connectome_snapshot(connectome_file, connectome_snap_file,
                            labels=labels_file, transform=numpy.log1p, dpi=300,
                            labels_size=4, colorbar_title="log(# of tracks)")

    return connectome_file, labels_file, connectome_snap_file


def mrtrix_connectomes(
        outdir,
        tempdir,
        tractogram,
        t1_brain,
        nodif_brain,
        t1_parc,
        t1_parc_lut,
        connectome_lut,
        tractogram_weights=None,
        tractogram_type="mrtrix",
        dif2anat_dat=None,
        dif2anat_mat=None,
        fix_freesurfer_subcortical=False,
        radial_search_dist=2.,
        forward_search_dist=5.,
        snapshots=True,
        fs_sh=DEFAULT_FREESURFER_PATH,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Compute the reduced connectome from a parcellation using MRtrix.

    Parameters
    ----------
    outdir: str
        Directory where to output.
    tempdir: str
        Path to the directory where temporary directories should be written.
        It should be a partition with 5+ GB available.
    tractogram: str or list of str
        The tractogram to be used in VTK, TRK, TXT or TRK format. It is
        possible to provide a list of tractograms only for Connectomist and
        Tracula.
    t1_brain: str
        The anatomical image.
    nodif_brain: str, default None
        Diffusion brain-only Nifti volume with bvalue ~ 0. If not passed, it is
        generated automatically by averaging all the b0 volumes of the DWI.
    t1_parc: str
        Path to the parcellation that defines the nodes of the connectome, e.g.
        aparc+aseg.mgz from FreeSurfer in the 't1_brain' space.
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
    tractogram_weights: str, default None
        The weight associated to each fiber: one weight per line.
    tractogram_type: str, default 'mrtrix'
        The software used to generate the tractogram. This parameter is used
        for format conversion purposes.
    dif2anat_dat: str, default None
        The diffusion to T1 FreeSurfer registration '.dat' file.
    dif2anat_mat: str
        The diffusion to T1 FSL registration '.mat' file.
    fix_freesurfer_subcortical: bool, default False
        If the <t1_parc> is aparc+aseg or aparc.a2009s+aseg from FreeSurfer,
        set this option to True, to recompute the subcortical segmentations
        of the 5 structures that are uncorrectly segmented by FreeSurfer,
        using FSL FIRST.
    radial_search_dist: float, default 2.0
        Multiple connectomes are generated depending on the streamline-node
        association strategy. The radial search assigns the nearest
        node from the streamline endpoint within this radius (in mm).
    forward_search_dist: float, default 5.0
        Multiple connectomes are generated depending on the streamline-node
        association strategy. The forward assignment projects the
        streamline forward from the endpoint to find a node, within this
        distance (in mm).
    snapshots: bool, default True
        If True, create PNG snapshots for QC.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.

    Returns
    -------
    connectome_endvox, connectome_radial, connectome_forward: str
        The generated reduced connectomes.
    """
    # -------------------------------------------------------------------------
    # STEP 0 - Check arguments

    # Get the default module LUT if parameter not provided
    if connectome_lut.lower() == "lausanne2008":
        module_dir = os.path.dirname(os.path.abspath(__file__))
        connectome_lut = os.path.join(module_dir, "Lausanne2008LUT.txt")

    # Check input paths
    paths_to_check = [t1_parc, t1_parc_lut, connectome_lut, t1_brain,
                      nodif_brain]
    paths_to_check.extend(tractogram)
    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Check supported tractogram
    if tractogram_type not in ["mrtrix", "mitk", "connectomist", "fsl",
                               "tracula"]:
        raise ValueError("Unsupported tractogram: {0}".format(tractogram_type))

    # Create <outdir> and/or <tempdir> if not existing
    for directory in [outdir, tempdir]:
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # -------------------------------------------------------------------------
    # STEP 1 - Align T1 parcellation to diffusion without downsampling
    parc_name = os.path.basename(t1_parc).split(".nii")[0].split(".mgz")[0]
    t1_parc_to_dif = os.path.join(outdir, parc_name + "_to_dif.nii.gz")
    if dif2anat_dat is not None:
        cmd_1c = ["mri_vol2vol",
                  "--mov",  nodif_brain,
                  "--targ", t1_parc,
                  "--inv",
                  "--no-resample",
                  "--interp", "nearest",
                  "--o",   t1_parc_to_dif,
                  "--reg", dif2anat_dat,
                  "--no-save-reg"]
        FSWrapper(cmd_1c, shfile=fs_sh)()
    elif dif2anat_mat is not None:
        raise NotImplementedError(
            "This code only supports 'dif2anat_dat' parameter.")
    else:
        raise ValueError("One transformation matrix is mandatory.")

    # -------------------------------------------------------------------------
    # STEP 2 - Convert LUT
    # Change integer labels in the LUT so that the each label corresponds
    # to the row/col position in the connectome
    nodes = os.path.join(outdir, "nodes.nii.gz")
    cmd_2 = ["labelconvert", t1_parc_to_dif, t1_parc_lut, connectome_lut,
             nodes, "-nthreads", "0", "-failonwarn"]
    subprocess.check_call(cmd_2)

    # -------------------------------------------------------------------------
    # STEP 3 - If the T1 parcellation is aparc+aseg or aparc.a2009s+aseg
    # from FreeSurfer, this option allows the recompute the subcortical
    # segmentations of 5 structures that are uncorrectly segmented by
    # FreeSurfer, using FSL FIRST
    if fix_freesurfer_subcortical:
        fixed_nodes = os.path.join(outdir, "nodes_fixSGM.nii.gz")
        nodes = fix_freesurfer_subcortical_parcellation(
            parc=nodes,
            t1_brain=t1_brain,
            lut=connectome_lut,
            output=fixed_nodes,
            tempdir=tempdir,
            nb_threads=0,
            fsl_sh=fsl_sh)

    # -------------------------------------------------------------------------
    # STEP 4 - Create connectomes with labels by combining fibers and nodes

    # Convert streamlines so that they can be used in MRtrix
    tck_tractogram = os.path.join(outdir, "fibers.tck")
    if tractogram_type == "mitk":
        if len(tractogram) != 1:
            raise ValueError("A one-file tractogram is expected.")
        convert_mitk_vtk_fibers_to_tck(tractogram[0], tck_tractogram)
    elif tractogram_type in ("connectomist", "tracula"):
        convert_trk_fibers_to_tck(
            nodif_brain, tractogram, tck_tractogram, tempdir)
    elif tractogram_type == "fsl":
        convert_probtrackx2_saved_paths_to_tck(
            nodif_brain, tractogram, tck_tractogram, tempdir)
    else:
        if len(tractogram) != 1:
            raise ValueError("A one-file tractogram is expected.")
        tck_tractogram = tractogram[0]

    # Read labels from LUT and create a list of labels: labels.txt
    labels_array = numpy.loadtxt(connectome_lut, dtype=str, usecols=[1])
    path_labels = os.path.join(outdir, "labels.txt")
    numpy.savetxt(path_labels, labels_array, fmt="%s")

    # Create connectome with end-voxel assignment
    connectome_endvox = os.path.join(outdir, "connectome_endvox.txt")
    cmd_8a = ["tck2connectome", tck_tractogram, nodes, connectome_endvox,
              "-keep_unassigned",  # Keep 'Unknown' label
              "-assignment_end_voxels"]
    if tractogram_weights is not None:
        cmd_8a += ["-tck_weights_in", tractogram_weights]
    subprocess.check_call(cmd_8a)

    # Create connectome with radial search assignment
    connectome_radial = os.path.join(
        outdir, "connectome_radial_{:.2f}mm.txt".format(radial_search_dist))
    cmd_8b = ["tck2connectome", tck_tractogram, nodes, connectome_radial,
              "-keep_unassigned",  # Keep 'Unknown' label
              "-assignment_radial_search", "{0}".format(radial_search_dist)]
    if tractogram_weights is not None:
        cmd_8b += ["-tck_weights_in", tractogram_weights]
    subprocess.check_call(cmd_8b)

    # Create connectome assigning the streamline to all nodes it intersects
    connectome_forward = os.path.join(
        outdir, "connectome_forward_{:.2f}mm.txt".format(forward_search_dist))
    cmd_8c = ["tck2connectome", tck_tractogram, nodes, connectome_forward,
              "-keep_unassigned",  # Keep 'Unknown' label
              "-assignment_forward_search", "{0}".format(forward_search_dist)]
    if tractogram_weights is not None:
        cmd_8c += ["-tck_weights_in", tractogram_weights]
    subprocess.check_call(cmd_8c)

    # If snapshots are requested
    if snapshots:
        snapshot_endvox = os.path.join(outdir, "connectome_endvox.png")
        connectome_snapshot(connectome_endvox, snapshot_endvox,
                            labels=path_labels, transform=numpy.log1p, dpi=300,
                            labels_size=4, colorbar_title="log(# of tracks)")

        snapshot_radial = os.path.join(
            outdir, "connectome_radial_{:.2f}mm.png".format(
                radial_search_dist))
        connectome_snapshot(connectome_radial, snapshot_radial,
                            labels=path_labels, transform=numpy.log1p, dpi=300,
                            labels_size=4, colorbar_title="log(# of tracks)")

        snapshot_forward = os.path.join(
            outdir, "connectome_forward_{:.2f}mm.png".format(
                forward_search_dist))
        connectome_snapshot(connectome_forward, snapshot_forward,
                            labels=path_labels, transform=numpy.log1p, dpi=300,
                            labels_size=4, colorbar_title="log(# of tracks)")

    return connectome_endvox, connectome_radial, connectome_forward
