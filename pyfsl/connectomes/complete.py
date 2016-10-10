##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Use FSL to generate a complete/dense vertices-based connectogram.
"""

# System import
import os
import glob
import numpy
import nibabel
from operator import itemgetter

# Pyfsl imports
from pyfsl import DEFAULT_FSL_PATH
from pyfsl.utils.regtools import flirt
from pyfsl.tractography.probabilist import probtrackx2
from pyfreesurfer.segmentation.freesurfer import mri_vol2surf


def get_profile(ico_order, nodif_file, nodifmask_file, seed_file,
                bedpostx_samples, outdir, t1_file, trf_file, dat_file, fsdir,
                sid, fslconfig=DEFAULT_FSL_PATH):
    """ Probabilistic profile.

    Computes the tractography using FSL probtrackx2 and projects the result
    on the cortical surface using FS mri_vol2surf.

    Parameters
    ----------
    ico_order: int (mandatory)
        Icosahedron order in [0, 7] that will be used to generate the cortical
        surface texture at a specific tessalation (the corresponding cortical
        surface can be resampled using the
        'clindmri.segmentation.freesurfer.resample_cortical_surface' function).
    nodif_file: str (mandatory)
        File for probtrackx2 containing the no diffusion volume and associated
        space information.
    nodifmask_file: str (mandatory)
        File for probtrackx2 containing the tractography mask (ie., a mask of
        the white matter).
    seed_file: str (mandatory)
        Text file for probtrackx2 containing seed coordinates.
    bedpostx_samples: str (mandatory)
        Path prefix for bedpostX model samples files injected in probtrackx2
        (eg., fsl.bedpostX/merged).
    outdir: str (mandatory)
        The output directory.
    t1_file : str (mandatory)
        T1 image file used to align the produced probabilitic tractography map
        to the T1 space.
    trf_file : str (mandatory)
        Diffusion to t1 space affine transformation matrix file.
    dat_file: str (mandatory)
        Structural to FreeSurfer space affine transformation matrix '.dat'
        file as computed by 'tkregister2'.
    fsdir: str (mandatory)
        FreeSurfer subjects directory 'SUBJECTS_DIR'.
    sid: str (mandatory)
        FreeSurfer subject identifier.
    fslconfig: str (mandatory)
        The FreeSurfer '.sh' config file.

    Returns
    -------
    proba_file: str
        The seed probabilistic tractography volume.
    textures: dict
        A dictionary containing the probabilist texture for each hemisphere.
    """
    # Generates the diffusion probability map
    proba_files, _ = probtrackx2(
        simple=True, seedref=nodif_file, out="fdt_paths", seed=seed_file,
        loopcheck=True, onewaycondition=True, samples=bedpostx_samples,
        mask=nodifmask_file, dir=outdir)

    # Check that only one 'fdt_paths' has been generated
    if len(proba_files) != 1:
        raise Exception("One probabilistic tractography file expected at this "
                        "point: {0}".format(proba_files))
    proba_file = proba_files[0]
    proba_fname = os.path.basename(proba_file).replace(".nii.gz", "")

    # Apply 'trf_file' affine transformation matrix using FSL flirt function:
    # probability map (diffusion space) -> probability map (T1 space).
    flirt_t1_file = os.path.join(outdir, proba_fname + "_t1_flirt.nii.gz")
    flirt(proba_file, t1_file, out=flirt_t1_file, applyxfm=True, init=trf_file)

    # Project the volumic probability map (T1 space) generated with FSL flirt
    # on the cortical surface (Freesurfer space) (both hemispheres) using
    # Freesurfer's mri_vol2surf and applying the 'dat_file' transformation.
    textures = {}
    for hemi in ["lh", "rh"]:
        prob_texture_file = os.path.join(
            outdir, "{0}.{1}_vol2surf.mgz".format(hemi, proba_fname))
        mri_vol2surf(hemi, flirt_t1_file, prob_texture_file, ico_order,
                     dat_file, fsdir, sid, surface_name="white",
                     flsconfig=fslconfig)
        textures[hemi] = prob_texture_file

    return proba_file, textures


def get_connectogram(profilesdir):
    """ Concatenate the conectivity profiles in a matrix.

    Parameters
    ----------
    profilesdir: str (mandatory)
        the directory with subfolders of the form '<hemi>_<seed_vertice>'
        containing the profiles of interest.

    Returns
    -------
    connectogram: array
        the connectogram array with sorted profiles as rows (according to the
        profile seeding vertice index). Profiles are formed with the right and
        left hemisphere probabilistic tractography volume projections in this
        order.
    seed_vertices: array
        the connectogram rows associated seed vertices indices.
    """
    # List and sort the profile subfolders
    rhtextures = glob.glob(os.path.join(profilesdir, "*", "rh.*.mgz"))
    rhtextures = [(int(texture.split(os.sep)[-2].split("_")[1]), texture)
                  for texture in rhtextures]
    rhtextures = sorted(rhtextures, key=itemgetter(0))

    # Construct the connectogram
    connectogram = []
    seed_vertices = []
    for seed_vertice, rhtexture in rhtextures:

        # Store the current seed vertice
        seed_vertices.append(seed_vertice)

        # Check if a left hemi texture has been computed
        lhtextures = glob.glob(os.path.join(
            profilesdir, "*_{0}".format(seed_vertice), "lh.*.mgz"))
        textures = [rhtexture]
        if len(lhtextures) == 1:
            textures.append(lhtextures[0])

        # Concatenate the right and left texture in a row
        profile = []
        for hemitexture in textures:

            # Load the hemi profile
            profile_array = nibabel.load(hemitexture).get_data()
            profile_dim = profile_array.ndim
            profile_shape = profile_array.shape
            if profile_dim != 3:
                raise ValueError(
                    "Expected profile texture array of dimension 3 not "
                    "'{0}'".format(profile_dim))
            if (profile_shape[1] != 1) or (profile_shape[2] != 1):
                raise ValueError(
                    "Expected profile texture array of shape (*, 1, 1) not "
                    "'{0}'.".format(profile_shape))
            texture = profile_array.ravel()
            profile.extend(texture.tolist())

        # Add this profile to the connectogram
        connectogram.append(profile)

    # Create a connectogram and rows indices arrays
    connectogram = numpy.asarray(connectogram)
    seed_vertices = numpy.asarray(seed_vertices)

    return connectogram, seed_vertices
