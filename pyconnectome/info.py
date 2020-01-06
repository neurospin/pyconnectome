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
    "pyconnectome/scripts/pyconnectome_ants_register",
    "pyconnectome/scripts/pyconnectome_ants_template",
    "pyconnectome/scripts/pyconnectome_applyxfm",
    "pyconnectome/scripts/pyconnectome_bedpostx",
    "pyconnectome/scripts/pyconnectome_bet",
    "pyconnectome/scripts/pyconnectome_blueprints",
    "pyconnectome/scripts/pyconnectome_bundles",
    "pyconnectome/scripts/pyconnectome_compress_tractogram",
    "pyconnectome/scripts/pyconnectome_dmri_preproc",
    "pyconnectome/scripts/pyconnectome_dtitk_create_templates",
    "pyconnectome/scripts/pyconnectome_dtitk_import_tensors",
    "pyconnectome/scripts/pyconnectome_dtitk_register",
    "pyconnectome/scripts/pyconnectome_dtitk_tbss",
    "pyconnectome/scripts/pyconnectome_folds_metrics",
    "pyconnectome/scripts/pyconnectome_folds_metrics_summary",
    "pyconnectome/scripts/pyconnectome_get_eddy_data",
    "pyconnectome/scripts/pyconnectome_life",
    "pyconnectome/scripts/pyconnectome_metrics",
    "pyconnectome/scripts/pyconnectome_metrics_heatmap",
    "pyconnectome/scripts/pyconnectome_mitk_tractogram",
    "pyconnectome/scripts/pyconnectome_mrtrix_tractogram",
    "pyconnectome/scripts/pyconnectome_probtrackx2_complete",
    "pyconnectome/scripts/pyconnectome_probtrackx2_tractogram",
    "pyconnectome/scripts/pyconnectome_project",
    "pyconnectome/scripts/pyconnectome_reduced_connectome",
    "pyconnectome/scripts/pyconnectome_reduced_probtrackx2",
    "pyconnectome/scripts/pyconnectome_register",
    "pyconnectome/scripts/pyconnectome_register_report",
    "pyconnectome/scripts/pyconnectome_report_tractogram",
    "pyconnectome/scripts/pyconnectome_scalars",
    "pyconnectome/scripts/pyconnectome_segmentation_report",
    "pyconnectome/scripts/pyconnectome_sulcal_pits_analysis",
    "pyconnectome/scripts/pyconnectome_sulcal_pits_correction",
    "pyconnectome/scripts/pyconnectome_sulcal_pits_parcellation",
    "pyconnectome/scripts/pyconnectome_tbss",
    "pyconnectome/scripts/pyconnectome_tbss_non_fa",
    "pyconnectome/scripts/pyconnectome_tbss_outliers",
    "pyconnectome/scripts/pyconnectome_tbss_psmd",
    "pyconnectome/scripts/pyconnectome_tbss_report",
    "pyconnectome/scripts/pyconnectome_tbss_stats",
    "pyconnectome/scripts/pyconnectome_tissue_segmentation",
    "pyconnectome/scripts/pyconnectome_tractseg"
]
