##########################################################################
# NSAp - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Continuous and analytical diffusion signal modelling with 3D-SHORE using
dipy.
"""

# System import
import os
import numpy

# Third party import
import nibabel
from dipy.viz import fvtk
from dipy.reconst.shm import sh_to_sf
from dipy.data import get_sphere
from dipy.reconst.shore import ShoreModel
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def shorefit(dwi_file, bvec_file, bval_file, mask_file, out, radial_order=6,
             zeta=700, lambdan=1e-8, lambdal=1e-8):
    """ Model the diffusion signal as a linear combination of continuous
    functions from the SHORE basis [Merlet2013]. Compute  also the analytical
    Orientation Distribution Function (ODF).

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
    radial_order: int, default 6
        The radial order of the SHORE basis.
    zeta: int, default 700
        The scale factor of the SHORE basis.
    lambdan, lambdal: float, default 1e-8
        The radial and angular regularization constants, respectively.

    Returns
    -------
    coeffs_file: str
        the SHORE coefficients.
    rtop_signal_file: str
        the return to origin probability (rtop) on the signal.
    rtop_pdf_file: str
        the return to origin probability (rtop) on the propagator.
    msd_file: str
        the mean square displacement on the propagator.
    """
    # Load the image
    dwi_image = nibabel.load(dwi_file)
    mask_image = nibabel.load(mask_file)

    # Mask locations
    mask_indices = numpy.where(mask_image.get_data() == 0)

    # Load the bvalues and bvectors
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs)

    # Create/fit the model
    model = ShoreModel(gtab, radial_order=radial_order,
                       zeta=zeta, lambdaN=lambdan, lambdaL=lambdal)
    fit = model.fit(dwi_image.get_data())
    coeffs = fit.shore_coeff
    coeffs[mask_indices] = 0
    coeffs_image = nibabel.Nifti1Image(coeffs, affine=dwi_image.affine)
    coeffs_file = os.path.join(out, "shore_coeffs.nii.gz")
    nibabel.save(coeffs_image, coeffs_file)

    # Compute the ODFs
    # sphere = get_sphere("symmetric724")
    # odf = fit.odf(sphere)

    # Calculate the analytical rtop on the signal that corresponds to the
    # integral of the signal.
    rtop_signal = fit.rtop_signal()
    rtop_signal[mask_indices] = 0
    rtop_signal_image = nibabel.Nifti1Image(
        rtop_signal, affine=dwi_image.affine)
    rtop_signal_file = os.path.join(out, "shore_rtop_signal.nii.gz")
    nibabel.save(rtop_signal_image, rtop_signal_file)

    # Now we calculate the analytical rtop on the propagator, that corresponds
    # to its central value. In theory, these two measures must be equal.
    rtop_pdf = fit.rtop_pdf()
    rtop_pdf[mask_indices] = 0
    rtop_pdf_image = nibabel.Nifti1Image(rtop_pdf, affine=dwi_image.affine)
    rtop_pdf_file = os.path.join(out, "shore_rtop_pdf.nii.gz")
    nibabel.save(rtop_pdf_image, rtop_pdf_file)

    # Let's calculate the analytical mean square displacement on the
    # propagator.
    msd = fit.msd()
    msd[mask_indices] = 0
    msd_image = nibabel.Nifti1Image(msd, affine=dwi_image.affine)
    msd_file = os.path.join(out, "shore_msd.nii.gz")
    nibabel.save(msd_image, msd_file)

    return coeffs_file, rtop_signal_file, rtop_pdf_file, msd_file
