##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Wrappers for the FSL's probabilistic tractography.
"""

# System import
import os
import glob

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper


def probtrackx2(samples, mask, seed, out="fdt_paths",
                dir="logdir", forcedir=False, simple=False, network=False,
                opd=False, os2t=False, targetmasks=None, waypoints=None,
                onewaycondition=False, avoid=None, stop=None, wtstop=None,
                omatrix1=False, omatrix2=False, target2=None, omatrix3=False,
                target3=None, xfm=None, invxfm=None, seedref=None,
                nsamples=5000, nsteps=2000, steplength=0.5, distthresh=0.0,
                cthr=0.2, fibthresh=0.01, loopcheck=False, usef=None,
                sampvox=0.0, randfib=0, shfile=DEFAULT_FSL_PATH):
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
        Steplength in mm.
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

    # Call probtrackx
    fslprocess = FSLWrapper(cmd, shfile=shfile)
    fslprocess()

    # Get the outputs
    proba_files = glob.glob(os.path.join(dir, out + "*"))
    if network:
        network_file = os.path.join(dir, "fdt_network_matrix")
    else:
        network_file = None

    return proba_files, network_file
