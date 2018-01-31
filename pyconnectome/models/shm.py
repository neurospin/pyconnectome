##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Q-Ball Constant Solid Angle using dipy.
"""

# System import
import os

# Third party import
import nibabel
from dipy.viz import fvtk
from dipy.data import get_sphere
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import peaks_from_model
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def qballfit(dwi_file, bvec_file, bval_file, mask_file, out, order=4):
    """ Model the diffusion signal using a Constant Solid Angle ODF (Q-Ball)
    model from Aganj et al. [Aganj2010].

    This code uses dipy.

    Parameters
    ----------
    dwi_file: str (mandatory)
        Diffusion weighted image data file (can be multi-shell).
        A 4D series of data volumes.
    bvec_file: str (mandatory)
        b vectors file.
        Gradient directions.
        An ASCII text file containing a list of gradient directions applied
        during diffusion weighted volumes. The order of entries in this file
        must match the order of volumes in the input data series.
    bval_file: str (mandatory)
        b values file.
        An ASCII text file containing a list of b values applied during each
        volume acquisition. The order of entries in this file must match the
        order of volumes in the input data and entries in the gradient
        directions text file.
    mask_file: str (mandatory)
        Brain binary mask file (i.e. from BET).
        A single binarized volume in diffusion space containing ones inside the
        brain and zeros outside the brain.
    out: str (mandatory)
        User specifies a basename that will be used to name the outputs.
    order: int, default 4
        the spherical harmonic order: must be even.

    Returns
    -------
    gfa_file: str
        the Generalized Fractional Anisotropy (GFA).
    qa_file: str
        the Quantitative Anisotropy (QA).
    shc_file: str
        the coefficients of the spherical harmonic basis for the ODF.
    odf_file: str
        the orientation distribution function on the sphere.
    """
    # Load the image
    dwi_image = nibabel.load(dwi_file)
    mask_image = nibabel.load(mask_file)

    # Load the bvalues and bvectors
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs)

    # Create the model
    model = CsaOdfModel(gtab, order)

    # Fit the model to data and computes peaks and metrics
    sphere = get_sphere("symmetric724")
    peaks = peaks_from_model(
        model=model,
        data=dwi_image.get_data(),
        sphere=sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=25,
        mask=mask_image.get_data(),
        return_sh=True,
        return_odf=True,
        normalize_peaks=True,
        sh_order=order,
        parallel=False)

    # Get the scalars
    gfa_image = nibabel.Nifti1Image(peaks.gfa, affine=dwi_image.affine)
    gfa_file = os.path.join(out, "qball_gfa.nii.gz")
    nibabel.save(gfa_image, gfa_file)
    qa_image = nibabel.Nifti1Image(peaks.qa, affine=dwi_image.affine)
    qa_file = os.path.join(out, "qball_qa.nii.gz")
    nibabel.save(qa_image, qa_file)

    # Get models
    # odf_image = nibabel.Nifti1Image(peaks.odf, affine=dwi_image.affine)
    # odf_file = os.path.join(out, "qball_odf.nii.gz")
    # nibabel.save(odf_image, odf_file)
    odf_file = None
    shc_image = nibabel.Nifti1Image(peaks.shm_coeff, affine=dwi_image.affine)
    shc_file = os.path.join(out, "qball_coeffs.nii.gz")
    nibabel.save(shc_image, shc_file)

    return gfa_file, qa_file, shc_file, odf_file
