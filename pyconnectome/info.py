#! /usr/bin/env python
##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 1
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Define default FSL path for the package
DEFAULT_FSL_PATH = "/etc/fsl/5.0/fsl.sh"

# Define FreeSurfer supported version
FSL_RELEASE = "5.0.9"

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 5 - Production/Stable",
               "Environment :: Console",
               "Environment :: X11 Applications :: Qt",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering",
               "Topic :: Utilities"]

# Project descriptions
description = """
Diffusion MRI connectome.
"""
SUMMARY = """
.. container:: summary-carousel

    pyConnectome is a Python module that can be used to play with diffusion
    MRI connectomes. This package offers:

    * registration scrips using ANTS, FSL or SPM.
    * complete or reduced connectome computation using MrTrix, MITK or FSL.
    * connectome graph analysis and visulaization.
    * tractogram compression and filtergin tools.
"""
long_description = (
    "pyConnectome\n\n"
    "Python wrappers around different softwares to compute structural "
    "(complete or reduced) connectomes.\n")

# Main setup parameters
NAME = "pyConnectome"
ORGANISATION = "CEA"
MAINTAINER = "Antoine Grigis"
MAINTAINER_EMAIL = "antoine.grigis@cea.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/neurospin/pyconnectome"
DOWNLOAD_URL = "https://github.com/neurospin/pyconnectome"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "pyConnectome developers"
AUTHOR_EMAIL = "antoine.grigis@cea.fr"
PLATFORMS = "OS Independent"
ISRELEASE = True
VERSION = __version__
PROVIDES = ["pyconnectome"]
REQUIRES = [
    "numpy>=1.14.0",
    "scipy>=0.9.0",
    "nibabel>=1.1.0",
    "dipy>=0.13.0",
    "networkx>=1.11",
    "matplotlib>=1.5.1",
    "progressbar2>=3.34.3",
]
EXTRA_REQUIRES = {
    "obsolete": {
        "clindmri>=0.0.0"
    },
    "standalone": {
        "bctpy>=0.5.0",
        "nilearn>=0.3.1",
        "pydcmio>=2.0.1",
        "scikit-learn>=0.19.1"
        "pyconnectomist>=2.0.0",
        "pyfreesurfer>=1.2.0",
    }
}
SCRIPTS = [
    "pyconnectome_ants_register",
    "pyconnectome_ants_template",
    "pyconnectome_applyxfm",
    "pyconnectome_bedpostx",
    "pyconnectome_bet",
    "pyconnectome_blueprints",
    "pyconnectome_bundles",
    "pyconnectome_compress_tractogram",
    "pyconnectome_dmri_preproc",
    "pyconnectome_dtitk_create_templates",
    "pyconnectome_dtitk_import_tensors",
    "pyconnectome_dtitk_register",
    "pyconnectome_dtitk_tbss",
    "pyconnectome_folds_metrics",
    "pyconnectome_folds_metrics_summary",
    "pyconnectome_get_eddy_data",
    "pyconnectome_life",
    "pyconnectome_metrics",
    "pyconnectome_metrics_heatmap",
    "pyconnectome_mitk_tractogram",
    "pyconnectome_mrtrix_tractogram",
    "pyconnectome_probtrackx2_complete",
    "pyconnectome_probtrackx2_tractogram",
    "pyconnectome_project",
    "pyconnectome_reduced_connectome",
    "pyconnectome_reduced_probtrackx2",
    "pyconnectome_register",
    "pyconnectome_register_report",
    "pyconnectome_report_tractogram",
    "pyconnectome_scalars",
    "pyconnectome_segmentation_report",
    "pyconnectome_sulcal_pits_analysis",
    "pyconnectome_sulcal_pits_correction",
    "pyconnectome_sulcal_pits_parcellation",
    "pyconnectome_tbss",
    "pyconnectome_tbss_non_fa",
    "pyconnectome_tbss_outliers",
    "pyconnectome_tbss_psmd",
    "pyconnectome_tbss_report",
    "pyconnectome_tbss_stats",
    "pyconnectome_tissue_segmentation",
   " pyconnectome_tractseg"
]
