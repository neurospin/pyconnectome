##########################################################################
# NSAp - Copyright (C) CEA, 2016 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Use FSL or dipy to fit a diffusion tensor or diffusion kurtosis model at each
voxel.
"""

# System import
import os
import shutil
import numpy

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper

# Third party import
import nibabel
import dipy.reconst.dki as dki
import dipy.reconst.dki_micro as dki_micro
from scipy.ndimage.filters import gaussian_filter
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table


def dkifit(dwi_file, bvec_file, bval_file, mask_file, out, min_kurtosis=-0.42,
           max_kurtosis=10, micro=False):
    """ The diffusion kurtosis model is an expansion of the diffusion tensor
    model (see Reconstruction of the diffusion signal with the Tensor model).
    In addition to the diffusion tensor (DT), the diffusion kurtosis model
    quantifies the degree to which water diffusion in biological tissues is
    non-Gaussian using the kurtosis tensor (KT) [Jensen2005].

    Measurements of non-Gaussian diffusion from the diffusion kurtosis model
    are of interest because they can be used to charaterize tissue
    microstructural heterogeneity [Jensen2010] and to derive concrete
    biophysical parameters, such as the density of axonal fibres and
    diffusion tortuosity [Fieremans2011].

    Theoretically, computing classical scalar measures from DTI and DKI
    should be analogous. However, according to recent studies, the diffusion
    statistics from the kurtosis model are expected to have better accuracy
    [Veraar2011], [NetoHe2012].

    Kurtosis measures are susceptible to high amplitude outliers. The impact
    of high amplitude kurtosis outliers can be removed by introducing as an
    optional input the extremes of the typical values of kurtosis.
    Here these are assumed to be on the range between -0.42 and 10.

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
    min_kurtosis: float, default -0.42
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are smaller than `min_kurtosis` are replaced
        with `min_kurtosis`.
    max_kurtosis: float, default 10
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are larger than `max_kurtosis` are replaced
        with `max_kurtosis`.
    micro: bool, default False
        If set, estimate yhe DKI based microstructural model.

    Returns
    -------
    fa_file: str
        the fractional anisotropy (FA).
    md_file: str
        the mean diffusivity (MD).
    ad_file: str
        the axial diffusivity (AD).
    rd_file: str
        the radial diffusivity (RD).
    ci_file: str
        the lineraity, planarity and sphericity Westion shapes.
    mk_file: str
        the non-Gaussian measures of mean kurtosis (MK).
    ak_file: str
        the axial kurtosis (AK).
    rk_file: str
        the radial kurtosis (RK)
    dkimask_file: str
        well-aligned fiber mask.
    dkiawf_file: str
        the Axonal water fraction (AWF).
    dkitortuosity_file: str
        the tortuosity.
    """
    # Load the image
    dwi_image = nibabel.load(dwi_file)
    mask_image = nibabel.load(mask_file)

    # Smooth the data
    data = dwi_image.get_data()
    fwhm = 1.25
    gauss_std = fwhm / numpy.sqrt(8 * numpy.log(2))
    data_smooth = numpy.zeros(data.shape)
    for indx in range(data.shape[-1]):
        data_smooth[..., indx] = gaussian_filter(data[..., indx],
                                                 sigma=gauss_std)

    # Load the bvalues and bvectors
    bvals, bvecs = read_bvals_bvecs(bval_file, bvec_file)
    gtab = gradient_table(bvals, bvecs)

    # Create/fit the model
    model = dki.DiffusionKurtosisModel(gtab, fit_method="WLS")
    fit = model.fit(data_smooth, mask=mask_image.get_data())

    # Get the tensor part scalars
    kt_image = nibabel.Nifti1Image(fit.kt, affine=dwi_image.affine)
    dkikt_file = os.path.join(out, "dki_kt.nii.gz")
    nibabel.save(kt_image, dkikt_file)
    fa_image = nibabel.Nifti1Image(fit.fa, affine=dwi_image.affine)
    dkifa_file = os.path.join(out, "dki_fa.nii.gz")
    nibabel.save(fa_image, dkifa_file)
    md_image = nibabel.Nifti1Image(fit.md, affine=dwi_image.affine)
    dkimd_file = os.path.join(out, "dki_md.nii.gz")
    nibabel.save(md_image, dkimd_file)
    ad_image = nibabel.Nifti1Image(fit.ad, affine=dwi_image.affine)
    dkiad_file = os.path.join(out, "dki_ad.nii.gz")
    nibabel.save(ad_image, dkiad_file)
    rd_image = nibabel.Nifti1Image(fit.rd, affine=dwi_image.affine)
    dkird_file = os.path.join(out, "dki_rd.nii.gz")
    nibabel.save(rd_image, dkird_file)
    cl_image = nibabel.Nifti1Image(fit.linearity, affine=dwi_image.affine)
    dkicl_file = os.path.join(out, "dki_cl.nii.gz")
    nibabel.save(cl_image, dkicl_file)
    cp_image = nibabel.Nifti1Image(fit.planarity, affine=dwi_image.affine)
    dkicp_file = os.path.join(out, "dki_cp.nii.gz")
    nibabel.save(cp_image, dkicp_file)
    cs_image = nibabel.Nifti1Image(fit.sphericity, affine=dwi_image.affine)
    dkics_file = os.path.join(out, "dki_cs.nii.gz")
    nibabel.save(cs_image, dkics_file)

    # Get the kutosis part scalars
    mk_image = nibabel.Nifti1Image(
        fit.mk(min_kurtosis, max_kurtosis), affine=dwi_image.affine)
    dkimk_file = os.path.join(out, "dki_mk.nii.gz")
    nibabel.save(mk_image, dkimk_file)
    ak_image = nibabel.Nifti1Image(
        fit.ak(min_kurtosis, max_kurtosis), affine=dwi_image.affine)
    dkiak_file = os.path.join(out, "dki_ak.nii.gz")
    nibabel.save(ak_image, dkiak_file)
    rk_image = nibabel.Nifti1Image(
        fit.rk(min_kurtosis, max_kurtosis), affine=dwi_image.affine)
    dkirk_file = os.path.join(out, "dki_rk.nii.gz")
    nibabel.save(rk_image, dkirk_file)

    # Estimate the microstructural model if requested
    dkimask_file, dkiawf_file, dkitortuosity_file = None, None, None
    if micro:

        # Create a white matter mask based on the westin shapes
        # Fieremans et al. [Fieremans2011]
        well_aligned_mask = numpy.ones(dwi_image.shape[:-1], dtype="bool")
        cl = fit.linearity.copy()
        well_aligned_mask[cl < 0.4] = False
        cp = fit.planarity.copy()
        well_aligned_mask[cp > 0.2] = False
        cs = fit.sphericity.copy()
        well_aligned_mask[cs > 0.35] = False

        # Removing nan associated with background voxels
        well_aligned_mask[numpy.isnan(cl)] = False
        well_aligned_mask[numpy.isnan(cp)] = False
        well_aligned_mask[numpy.isnan(cs)] = False

        # Save mask
        mask_image = nibabel.Nifti1Image(
            well_aligned_mask.astype(int), affine=dwi_image.affine)
        dkimask_file = os.path.join(out, "dki_mask.nii.gz")
        nibabel.save(mask_image, dkimask_file)

        # Create/fit the model
        micro_model = dki_micro.KurtosisMicrostructureModel(gtab)
        micro_fit = micro_model.fit(data_smooth, mask=well_aligned_mask)

        # Get scalars
        awf_image = nibabel.Nifti1Image(micro_fit.awf, affine=dwi_image.affine)
        dkiawf_file = os.path.join(out, "dki_awf.nii.gz")
        nibabel.save(awf_image, dkiawf_file)
        tortuosity_image = nibabel.Nifti1Image(
            micro_fit.tortuosity, affine=dwi_image.affine)
        dkitortuosity_file = os.path.join(out, "dki_tortuosity.nii.gz")
        nibabel.save(tortuosity_image, dkitortuosity_file)

    return (dkikt_file, dkifa_file, dkimd_file, dkiad_file, dkird_file,
            dkicl_file, dkicp_file, dkics_file, dkimk_file, dkiak_file,
            dkirk_file, dkimask_file, dkiawf_file, dkitortuosity_file)


def dtifit(data, bvecs, bvals, mask, out, wls=False, save_tensor=False,
           fslconfig=DEFAULT_FSL_PATH):
    """ Fit a diffusion tensor model at each voxel in the mask.

    Binding around the FSL's 'dtifit' command.

    The basic usage is:
        dtifit --data <filename>
        dtifit --verbose

    Parameters
    ----------
    data: str (mandatory)
        Diffusion weighted image data file.
        A 4D series of data volumes.
    bvecs: str (mandatory)
        b vectors file.
        Gradient directions.
        An ASCII text file containing a list of gradient directions applied
        during diffusion weighted volumes. The order of entries in this file
        must match the order of volumes in the input data series.
    bvals: str (mandatory)
        b values file.
        An ASCII text file containing a list of b values applied during each
        volume acquisition. The order of entries in this file must match the
        order of volumes in the input data and entries in the gradient
        directions text file.
    mask: str (mandatory)
        Brain binary mask file (i.e. from BET).
        A single binarized volume in diffusion space containing ones inside the
        brain and zeros outside the brain.
    out: str (mandatory)
        User specifies a basename that will be used to name the outputs of
        dtifit.
    wls: bool (optional, default False)
        Fit the tensor with weighted least squares.
    save_tensor: bool (optional, default False)
        Save the elements of the tensor.

    Returns
    -------
    v1_file: str
        path/name of file with the 1st eigenvector.
    v2_file: str
        path/name of file with the 2nd eigenvector.
    v3_file: str
        path/name of file with the 3rd eigenvector.
    l1_file: str
        path/name of file with the 1st eigenvalue.
    l2_file: str
       path/name of file with the  2nd eigenvalue.
    l3_file: str
        path/name of file with the 3rd eigenvalue.
    md_file: str
        path/name of file with the mean diffusivity.
    fa_file: str
        path/name of file with the Fractional anisotropy.
    s0_file: str
        path/name of file with the Raw T2 signal with no diffusion weighting.
    tensor_file: str
        path/name of file with the 4D tensor volume.
    m0_file: str
        path/name of file with the mode of the anisotropy.
    """
    # Check input parameters
    for filename in (data, bvals, bvecs, mask):
        if not os.path.isfile(filename):
            raise ValueError("'{0}' is not a valid input file.".format(
                filename))

    # Check that the output directory exists
    if not os.path.isdir(out):
        os.mkdir(out)
    out = os.path.join(out, "dtifit")

    # Define the FSL command
    cmd = ["dtifit",
           "-k", data,
           "-r", bvecs,
           "-b", bvals,
           "-m", mask,
           "-o", out]

    # Add optional arguments
    if wls:
        cmd += ["--wls"]
    if save_tensor:
        cmd += ["--save_tensor"]

    # Execute the FSL command
    fslprocess = FSLWrapper(shfile=fslconfig)
    fslprocess(cmd=cmd)

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not decalred in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Build the output names
    image_ext = FSLWrapper.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]
    v1_file = out + "_V1" + image_ext
    v2_file = out + "_V2" + image_ext
    v3_file = out + "_V3" + image_ext
    l1_file = out + "_L1" + image_ext
    l2_file = out + "_L2" + image_ext
    l3_file = out + "_L3" + image_ext
    md_file = out + "_MD" + image_ext
    fa_file = out + "_FA" + image_ext
    s0_file = out + "_S0" + image_ext
    tensor_file = out + "_tensor" + image_ext
    m0_file = out + "_M0" + image_ext

    return (v1_file, v2_file, v3_file, l1_file, l2_file, l3_file, md_file,
            fa_file, s0_file, tensor_file, m0_file)
