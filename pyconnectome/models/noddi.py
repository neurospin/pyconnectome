##########################################################################
# NSAp - Copyright (C) CEA, 2016 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import shutil
import os


def noddifit(dwi_file, bvec_file, bval_file, mask_file, out):
    """ The NODDI model (neurite orientation dispersion and density imaging)
    estimates the microstructural complexity of dendrites and axons in
    vivo on clinical MRI scanners.

    Install:
    pip install git+https://github.com/samuelstjean/spams-python.git
    pip install --user git+https://github.com/daducci/AMICO.git

    This code uses amico.

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
        User specifies a output directory.

    Returns
    -------
    od_file: str
        the orientation dispersion index.
    icvf_file: str
        the intra cellular volume fraction.
    isovf_file: str
        the fisotropic (CSF) volume fraction.
    dir_file: str
        the fitted directions.
    config_file: str
        the configuration file.
    """
    import amico

    # Create the expected organization
    os.chdir(out)
    amico_root = os.path.join(out, "Study", "Subject")
    if not os.path.isdir(amico_root):
        os.makedirs(amico_root)
    os.symlink(dwi_file, os.path.join(amico_root, "dwi.nii.gz"))
    os.symlink(bvec_file, os.path.join(amico_root, "dwi.bvec"))
    os.symlink(bval_file, os.path.join(amico_root, "dwi.bval"))
    os.symlink(mask_file, os.path.join(amico_root, "mask.nii.gz"))

    # Load the data
    amico.core.setup()
    ae = amico.Evaluation("Study", "Subject")
    amico.util.fsl2scheme(os.path.join("Study", "Subject", "dwi.bval"),
                          os.path.join("Study", "Subject", "dwi.bvec"))
    ae.load_data(dwi_filename="dwi.nii.gz", scheme_filename="dwi.scheme",
                 mask_filename="mask.nii.gz", b0_thr=0)
    ae.set_model("NODDI")

    # Compute the response functions
    ae.generate_kernels()
    ae.load_kernels()

    # Model fit
    ae.fit()
    ae.save_results()

    # Export results
    amico_out = os.path.join(amico_root, "AMICO", "NODDI")
    outputs = []
    for basename in ("FIT_OD.nii.gz", "FIT_ICVF.nii.gz", "FIT_ISOVF.nii.gz",
                     "FIT_dir.nii.gz", "config.pickle"):
        src = os.path.join(amico_out, basename)
        dest = os.path.join(out, "noddi_{0}".format(
            basename.replace("FIT_", "").lower()))
        outputs.append(dest)
        shutil.copy2(src, dest)
    shutil.rmtree(os.path.join(out, "Study"))

    return outputs
