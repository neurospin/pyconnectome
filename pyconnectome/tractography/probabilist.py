##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Probabilistic tractography using FSL or MRtrix.
"""

# System import
import os
import glob
import subprocess

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper
from pyconnectome.utils.filetools import mrtrix_extract_b0s_and_mean_b0

# Third-party
from pyfreesurfer import DEFAULT_FREESURFER_PATH
from pyfreesurfer.wrapper import FSWrapper
from pyfreesurfer.utils.filetools import get_or_check_freesurfer_subjects_dir
from pyconnectomist.utils.dwitools import read_bvals_bvecs
from pyconnectome.utils.regtools import freesurfer_bbregister_t1todif


def probtrackx2(
        samples,
        mask,
        seed,
        out="fdt_paths",
        dir="logdir",
        forcedir=False,
        simple=False,
        network=False,
        opd=False,
        pd=False,
        os2t=False,
        targetmasks=None,
        waypoints=None,
        onewaycondition=False,
        avoid=None,
        stop=None,
        wtstop=None,
        omatrix1=False,
        omatrix2=False,
        target2=None,
        omatrix3=False,
        target3=None,
        xfm=None,
        invxfm=None,
        seedref=None,
        nsamples=5000,
        nsteps=2000,
        steplength=0.5,
        distthresh=0.0,
        cthr=0.2,
        fibthresh=0.01,
        loopcheck=False,
        usef=None,
        sampvox=0.0,
        randfib=0,
        savepaths=False,
        shfile=DEFAULT_FSL_PATH):
    """ Wraps command probtrackx2.

    Single voxel
    ------------

    [1] Connectivity from a single seed point.

    probtrackx2(samples="/.../fsl.bedpostX/merged",
                mask="/.../fsl.bedpostX/nodif_brain_mask",
                seed="$PATH/tracto/seedvox_coordinates.txt",
                simple=True,
                loopcheck=True,
                dir="$PATH",
                out="SingleVoxel_paths")

    [2] Tracking in a standard / no-diffusion space.

    probtrackx2(samples="/.../fsl.bedpostX/merged",
                mask="/.../fsl.bedpostX/nodif_brain_mask",
                seed="$PATH/tracto/seedvox_coordinates.txt",
                seeref="/.../fsl.bedpostX/nodif_brain_mask",
                simple=True,
                loopcheck=True,
                dir="$PATH",
                out="SingleVoxel_paths")

    Single mask
    -----------

    probtrackx2(seed="/.../lh-precentral.nii.gz",
                loopcheck=True,
                onewaycondition=True,
                samples="/.../fsl.bedpostX/merged",
                mask="/.../fsl.bedpostX/nodif_brain_mask",
                dir="$PATH")

    Multiple masks
    --------------
    probtrackx2(network=True,
                seed="$PATH/masks.txt",
                loopcheck=True,
                onewaycondition=True,
                samples="/.../fsl.bedpostX/merged",
                mask="/.../fsl.bedpostX/nodif_brain_mask",
                dir="$PATH")

    Usage:
    probtrackx2 -s <basename> -m <maskname> -x <seedfile> -o <output>
                --targetmasks=<textfile>

    Parameters
    ----------
    samples: str (mandatory)
        Basename for samples files - e.g. 'merged'.
    mask: str (mandatory)
        Bet binary mask file in diffusion space.
    seed: str (mandatory)
        Seed volume or list (ascii text file) of volumes and/or surfaces.
    out: str (optional, default "fdt_paths")
        Output file.
    dir: str (optional, default 'logdir')
        Directory to put the final volumes in - code makes this directory.
    forcedir: bool (optional, default False)
        Use the actual directory name given - i.e. don't add + to make a new
        directory.
    simple: (optional, default False)
        Track from a list of voxels (seed must be a ASCII list of coordinates).
    network: bool (optional, default False)
        Activate network mode - only keep paths going through at least one of
        the other seed masks.
    opd: bool (optional, default False)
        Output path distribution
    pd: bool (optional, default False)
        Correct path distribution for the length of the pathways
    os2t: bool (optional, default False)
        Output seeds to targets
    targetmasks: (optional, default None)
        File containing a list of target masks - for seeds_to_targets
        classification.
    waypoints: (optional, default None)
        Waypoint mask or ascii list of waypoint masks - only keep paths going
        through ALL the masks.
    onewaycondition: (optional, default False)
        Apply waypoint conditions to each half tract separately.
    avoid: (optional, default None)
        Reject pathways passing through locations given by this mask.
    stop: (optional, default None)
        Stop tracking at locations given by this mask file.
    wtstop: (optional, default None)
        One mask or text file with mask names. Allow propagation within mask
        but terminate on exit. If multiple masks, non-overlapping volumes
        expected.
    omatrix1: bool (optional, default False)
        Output matrix1 - SeedToSeed Connectivity.
    omatrix2: bool (optional, default False)
        Output matrix2 - SeedToLowResMask.
    target2: (optional, default None)
        Low resolution binary brain mask for storing connectivity distribution
        in matrix2 mode.
    omatrix3: bool (optional, default False)
        Output matrix3 (NxN connectivity matrix).
    target3: (optional, default None)
        Mask used for NxN connectivity matrix (or Nxn if lrtarget3 is set).
    xfm: (optional, default None)
        Transform taking seed space to DTI space (either FLIRT matrix or FNIRT
        warpfield) - default is identity.
    invxfm: (optional, default None)
        Transform taking DTI space to seed space (compulsory when using a
        warpfield for seeds_to_dti).
    seedref: (optional, default None)
        Reference vol to define seed space in simple mode - diffusion space
        assumed if absent.
    nsamples: int (optional, default 5000)
        Number of samples.
    nsteps: int (optional, default 2000)
        Number of steps per sample.
    steplength: float (optional, default 0.5)
        Step length in mm.
    distthresh: float (optional, default 0.0)
        Discards samples shorter than this threshold (in mm)
    cthr: float (optional, default 0.2)
        Curvature threshold.
    fibthresh: float (optional, default 0.01)
        Volume fraction before subsidary fibre orientations are considered.
    loopcheck: (optional, default False)
        Perform loopchecks on paths - slower, but allows lower curvature
        threshold.
    usef: (optional, default None)
        Use anisotropy to constrain tracking.
    sampvox: float (optional, default 0.0)
        Sample random points within x mm sphere seed voxels (e.g. --sampvox=5).
    randfib: int (optional, default 0)
        Set to 1 to randomly sample initial fibres (with f > fibthresh).
        Set to 2 to sample in proportion fibres (with f > fibthresh) to f.
        Set to 3 to sample ALL populations at random (even if f < fibthresh)
    savepaths: bool, optional
        Probtrackx2 hidden option: output a ASCII text file with all the
        coordinates (can be quite large if you run many streamlines).
    shfile: str (optional, default NeuroSpin path)
        The FSL configuration batch.

    Returns
    -------
    proba_files: list of str
        A list of files containing probabilistic fiber maps.
    network_file: str
        A voxel-by-target connection matrix.
    """
    # Check the input parameters
    for path in (seed, mask):
        if not os.path.isfile(path):
            raise ValueError("'{0}' is not a valid input file.".format(path))

    # Define the FSL command
    cmd = ["probtrackx2",
           "-s", samples,
           "-m", mask,
           "-x", seed,
           "--out=%s" % out,
           "--dir=%s" % dir,
           "--nsamples=%i" % nsamples,
           "--nsteps=%i" % nsteps,
           "--steplength=%f" % steplength,
           "--distthresh=%f" % distthresh,
           "--cthr=%f" % cthr,
           "--fibthresh=%f" % fibthresh,
           "--sampvox=%f" % sampvox,
           "--randfib=%i" % randfib]

    # Add optional arguments
    if forcedir:
        cmd += ["--forcedir"]
    if opd:
        cmd += ["--opd"]
    if pd:
        cmd += ["--pd"]
    if os2t:
        cmd += ["--os2t"]
    if network:
        cmd += ["--network"]
    if loopcheck:
        cmd += ["--loopcheck"]
    if omatrix1:
        cmd += ["--omatrix1"]
    if omatrix2:
        cmd += ["--omatrix2"]
    if omatrix3:
        cmd += ["--omatrix3"]
    if onewaycondition:
        cmd += ["--onewaycondition"]
    if simple:
        cmd += ["--simple"]
    if avoid is not None:
        cmd += ["--avoid=%s" % avoid]
    if targetmasks is not None:
        cmd += ["--targetmasks=%s" % targetmasks]
    if waypoints is not None:
        cmd += ["--waypoints=%s" % waypoints]
    if stop is not None:
        cmd += ["--stop=%s" % stop]
    if wtstop is not None:
        cmd += ["--wtstop=%s" % wtstop]
    if usef is not None:
        cmd += ["--usef=%s" % usef]
    if seedref is not None:
        cmd += ["--seedref=%s" % seedref]
    if target2 is not None:
        cmd += ["--target2=%s" % target2]
    if target3 is not None:
        cmd += ["--target3=%s" % target3]
    if xfm is not None:
        cmd += ["--xfm=%s" % xfm]
    if invxfm is not None:
        cmd += ["--invxfm=%s" % invxfm]
    if savepaths:
        cmd += ["--savepaths"]

    # Call probtrackx
    fslprocess = FSLWrapper(shfile=shfile)
    fslprocess(cmd=cmd)

    # Get the outputs
    proba_files = glob.glob(os.path.join(dir, out + "*"))
    if network:
        network_file = os.path.join(dir, "fdt_network_matrix")
    else:
        network_file = None

    return proba_files, network_file


def mrtrix_tractogram(
        outdir,
        tempdir,
        subject_id,
        dwi,
        bvals,
        bvecs,
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
        fast_t1_brain=None,
        subjects_dir=None,
        mif_gz=True,
        delete_raw_tracks=False,
        delete_dwi_mif=True,
        fs_sh=DEFAULT_FREESURFER_PATH,
        fsl_sh=DEFAULT_FSL_PATH):
    """
    Compute the connectome using MRtrix.

    Requirements:
        - FreeSurfer: result of recon-all on the T1.
        - a T1 parcellation that defines the nodes of the connectome, it has
          to be in the FreeSurfer space (i.e. aligned with
          <subjects dir>/<subject>/mri/brain.mgz), e.g. aparc+aseg from
          FreeSurfer.

    Parameters
    ----------
    outdir: str
        Path to directory where to output.
    tempdir: str
        Path to the directory where temporary directories should be written.
        It should be a partition with 5+ GB available.
    subject_id: str
        Subject identifier.
    dwi: str
        Path to the diffusion-weighted images (Nifti required).
    bvals: str
        Path to the bvalue list.
    bvecs: str
        Path to the list of diffusion-sensitized directions.
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
    fast_t1_brain: str, default None
        By default FSL FAST is run on the FreeSurfer 'brain.mgz'. If you want
        the WM probability map to be computed from another T1, pass the T1
        brain-only volume. Note that it has to be aligned with diffusion.
        This argument is useful for HCP, where some FreeSurfer 'brain.mgz'
        cannot be processed by FSL FAST.
    subjects_dir: str, default None
        Path to the FreeSurfer subjects directory. Required if the environment
        variable $SUBJECTS_DIR is not set.
    mif_gz: bool, default True
        Use compressed MIF files (.mif.gz) instead of .mif to save space.
    delete_raw_tracks: bool, default False
        Delete the raw tracks (<outdir>/<mtracks>M.tck) at the end of
        processing, to save space.
    delete_dwi_mif: bool, default True
        Delete <outdir>/DWI.mif(.gz) at the end of processing, which is a copy
        of the input <dwi> in the MIF format, to save space.
    fs_sh: str, default NeuroSpin path
        Path to the Bash script setting the FreeSurfer environment
    fsl_sh: str, default NeuroSpin path
        Path to the Bash script setting the FSL environment.

    Returns
    -------
    tracks: str
        The generated tractogram.
    sift_tracks: str
        The SIFT filtered tractogram.
    sift2_weights: str
        The SIFT2 tractogram associated weights.
    """
    # -------------------------------------------------------------------------
    # STEP 0 - Check arguments

    # FreeSurfer $SUBJECTS_DIR has to be passed or set as an env variable
    subjects_dir = get_or_check_freesurfer_subjects_dir(subjects_dir)

    # Use compressed MIF files or not
    MIF_EXT = ".mif.gz" if mif_gz else ".mif"

    # Is SIFT to be applied
    sift = (sift_mtracks is not None)

    # Check input and optional paths
    paths_to_check = [dwi, bvals, bvecs, fsl_sh]
    for p in [nodif_brain, nodif_brain_mask, fast_t1_brain]:
        if p is not None:
            paths_to_check.append(p)
    for p in paths_to_check:
        if not os.path.exists(p):
            raise ValueError("File or directory does not exist: %s" % p)

    # Identify whether the DWI acquisition is single or multi-shell
    _, _, nb_shells, _ = read_bvals_bvecs(bvals, bvecs, min_bval=200.)
    is_multi_shell = nb_shells > 1

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
    # - apply transformation to align T1 without downsampling
    t1_brain_to_dif, dif2anat_dat, _ = freesurfer_bbregister_t1todif(
            outdir=outdir,
            subject_id=subject_id,
            nodif_brain=nodif_brain,
            subjects_dir=subjects_dir,
            fs_sh=fs_sh,
            fsl_sh=fsl_sh)

    # -------------------------------------------------------------------------
    # STEP 3 - "5 tissue types" segmentation
    # Generate the 5TT image based on a FSL FAST
    five_tissues = os.path.join(outdir, "5TT%s" % MIF_EXT)
    fast_t1_brain = t1_brain_to_dif if fast_t1_brain is None else fast_t1_brain
    cmd_3 = ["5ttgen", "fsl", fast_t1_brain, five_tissues, "-premasked",
             "-tempdir", tempdir, "-nthreads", "%i" % nb_threads]
    process = FSLWrapper(cmd_3, env=os.environ, shfile=fsl_sh)
    process(cmd=cmd_3)

    # -------------------------------------------------------------------------
    # STEP 4 - Estimation of the response function of fibers in each voxel
    if is_multi_shell:
        rf_wm = os.path.join(outdir, "RF_WM.txt")
        rf_gm = os.path.join(outdir, "RF_GM.txt")
        rf_csf = os.path.join(outdir, "RF_CSF.txt")
        cmd_4 = ["dwi2response", "msmt_5tt", dwi_mif, five_tissues,
                 rf_wm, rf_gm, rf_csf]
    else:
        rf = os.path.join(outdir, "RF.txt")
        cmd_4 = ["dwi2response", "tournier", dwi_mif, rf]
    rf_voxels = os.path.join(outdir, "RF_voxels%s" % MIF_EXT)
    cmd_4 += ["-voxels", rf_voxels, "-tempdir", tempdir,
              "-nthreads", "%i" % nb_threads]
    subprocess.check_call(cmd_4)

    # -------------------------------------------------------------------------
    # STEP 5 - Compute FODs
    wm_fods = os.path.join(outdir, "WM_FODs%s" % MIF_EXT)
    if is_multi_shell:
        gm_mif = os.path.join(outdir, "GM%s" % MIF_EXT)
        csf_mif = os.path.join(outdir, "CSF%s" % MIF_EXT)
        cmd_5 = ["dwi2fod", "msmt_csd", dwi_mif, rf_wm, wm_fods,
                 rf_gm, gm_mif, rf_csf, csf_mif]
    else:
        cmd_5 = ["dwi2fod", "csd", dwi_mif, rf, wm_fods]
    cmd_5 += ["-mask", nodif_brain_mask, "-nthreads", "%i" % nb_threads,
              "-failonwarn"]
    subprocess.check_call(cmd_5)

    # -------------------------------------------------------------------------
    # STEP 6 - Image to visualize for multi-shell
    if is_multi_shell:
        wm_fods_vol0 = os.path.join(outdir, "WM_FODs_vol0%s" % MIF_EXT)
        cmd_6a = ["mrconvert", wm_fods, wm_fods_vol0, "-coord", "3", "0",
                  "-nthreads", "%i" % nb_threads]
        subprocess.check_call(cmd_6a)

        tissueRGB_mif = os.path.join(outdir, "tissueRGB%s" % MIF_EXT)
        cmd_6b = ["mrcat", csf_mif, gm_mif, wm_fods_vol0, tissueRGB_mif,
                  "-axis", "3", "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_6b)

    # -------------------------------------------------------------------------
    # STEP 7 - Tractography: tckglobal or tckgen
    if global_tractography:
        tracks = os.path.join(outdir, "global.tck")
        global_fod = os.path.join(outdir, "fod%s" % MIF_EXT)
        fiso_mif = os.path.join(outdir, "fiso%s" % MIF_EXT)
        cmd_7 = ["tckglobal", dwi_mif, rf_wm, "-riso", rf_csf, "-riso", rf_gm,
                 "-mask", nodif_brain_mask, "-niter", "1e8",
                 "-fod", global_fod, "-fiso", fiso_mif, tracks]
    else:
        # Anatomically Constrained Tractography:
        # iFOD2 algorithm with backtracking and crop fibers at GM/WM interface
        tracks = os.path.join(outdir, "%iM.tck" % mtracks)
        cmd_7 = ["tckgen", wm_fods, tracks, "-act", five_tissues, "-backtrack",
                 "-crop_at_gmwmi", "-maxlength", "%i" % maxlength,
                 "-number", "%dM" % mtracks, "-cutoff", "%f" % cutoff]

        # Requested seeding strategy: -seed_gmwmi or -seed_dynamic
        if seed_gmwmi:
            gmwmi_mask = os.path.join(outdir, "gmwmi_mask%s" % MIF_EXT)
            cmd_7b = ["5tt2gmwmi", five_tissues, gmwmi_mask]
            subprocess.check_call(cmd_7b)
            cmd_7 += ["-seed_gmwmi", gmwmi_mask]
        else:
            cmd_7 += ["-seed_dynamic", wm_fods]
    cmd_7 += ["-nthreads", "%i" % nb_threads, "-failonwarn"]
    subprocess.check_call(cmd_7)

    # -------------------------------------------------------------------------
    # STEP 8 - Filter tracts with SIFT if requested
    sift_tracks = None
    if sift:
        sift_tracks = os.path.join(outdir, "%iM_SIFT.tck" % sift_mtracks)
        cmd_8 = ["tcksift", tracks, wm_fods, sift_tracks,
                 "-act", five_tissues, "-term_number", "%iM" % sift_mtracks,
                 "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_8)

    # -------------------------------------------------------------------------
    # STEP 9 - run SIFT2 if requested (compute weights of fibers)
    sift2_weights = None
    if sift2:
        sift2_weights = os.path.join(outdir, "sift2_weights.txt")
        cmd_9 = ["tcksift2", tracks, wm_fods, sift2_weights,
                 "-act", five_tissues,
                 "-nthreads", "%i" % nb_threads, "-failonwarn"]
        subprocess.check_call(cmd_9)

    # -------------------------------------------------------------------------
    # STEP 10 - clean if requested
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

    return tracks, sift_tracks, sift2_weights
