##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Utilities to check connectivity profiles.
"""

# System import
import os
import nibabel

# Pyfreesurfer import
from pyfreesurfer.utils.surftools import TriSurface

# Clindmri import
import clindmri.plot.pvtk as pvtk
from clindmri.plot.slicer import animate_image


def qc_profile(
        nodif_file,
        proba_file,
        proba_texture,
        ico_order,
        fsdir,
        sid,
        outdir,
        fsconfig,
        actor_ang=(0., 0., 0.)):
    """ Connectivity profile QC.

    Generates views of:
    - the superposition of the nodif image with tractography result volume.
    - the connected points on the cortical surface
    Resample cortical meshes if needed.
    Results output are available as gif and png.

    Parameters
    ----------
    nodif_file: str (mandatory)
        file for probtrackx2 containing the no diffusion volume and associated
        space information.
    proba_file: str (mandatory)
        the protrackx2 output seeding probabilistic path volume.
    proba_texture: dict (mandatory)
        the FreeSurfer mri_vol2surf '.mgz' 'lh' and 'rh' textrue that contains
        the cortiacal connection strength.
    ico_order: int (mandatory)
        icosahedron order in [0, 7] that will be used to generate the cortical
        surface texture at a specific tessalation (the corresponding cortical
        surface can be resampled using the
        'clindmri.segmentation.freesurfer.resample_cortical_surface' function).
    fsdir: str (mandatory)
        FreeSurfer subjects directory 'SUBJECTS_DIR'.
    sid: str (mandatory)
        FreeSurfer subject identifier.
    outdir: str (mandatory)
        The QC output directory.
    fsconfig: str (mandatory)
        the FreeSurfer '.sh' config file.
    actor_ang: 3-uplet (optinal, default (0, 0, 0))
        the actor x, y, z position (in degrees).

    Returns
    -------
    snaps: list of str
        two gifs images, one showing the connection profile as a texture on
        the cortical surface, the other a volumic representation of the
        deterministic tractography.
    """

    # Construct/check the subject directory
    subjectdir = os.path.join(fsdir, sid)
    if not os.path.isdir(subjectdir):
        raise ValueError(
            "'{0}' is not a FreeSurfer subject directory.".format(subjectdir))

    # Check that the output QC directory exists
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    # Superpose the nodif and probabilistic tractography volumes
    proba_shape = nibabel.load(proba_file).shape
    snaps = []
    snaps.append(
        animate_image(nodif_file, overlay_file=proba_file, clean=True,
                      overlay_cmap="Spectral", cut_coords=proba_shape[2],
                      outdir=outdir))

    # Define a renderer
    ren = pvtk.ren()

    # For each hemisphere
    for hemi in ["lh", "rh"]:

        # Get the white mesh on the desired icosphere
        meshfile = os.path.join(
            subjectdir, "convert", "{0}.white.{1}.native".format(
                hemi, ico_order))
        if not os.path.isfile(meshfile):
            raise ValueError(
                "'{0}' is not a valid white mesh. Generate it through the "
                "'clindmri.scripts.freesurfer_conversion' script.".format(
                    meshfile))

        # Check texture has the expected extension, size
        texture_file = proba_texture[hemi]
        if not texture_file.endswith(".mgz"):
            raise ValueError("'{0}' is not a '.mgz' file. Format not "
                             "supported.".format(texture_file))
        profile_array = nibabel.load(texture_file).get_data()
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

        # Flatten the profile texture array
        texture = profile_array.ravel()

        # Load the white mesh
        surface = TriSurface.load(meshfile)

        # Define a textured surface actor
        actor = pvtk.surface(surface.vertices, surface.triangles, texture)
        actor.RotateX(actor_ang[0])
        actor.RotateY(actor_ang[1])
        actor.RotateZ(actor_ang[2])
        pvtk.add(ren, actor)

    # Create a animaton with the generated surface
    qcname = "profile_as_texture"
    snaps.extend(
        pvtk.record(ren, outdir, qcname, n_frames=36, az_ang=10, animate=True,
                    delay=10))

    return snaps
