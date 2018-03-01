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
version_micro = 0

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
long_description = """
============
pyConnectome
============

Python wrappers around different softwares to compute structural (complete or
reduced) connectomes.
"""

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
    "dipy>=0.13.0"
]
EXTRA_REQUIRES = {}
SCRIPTS = [
    "pyconnectome/scripts/pyconnectome_ants_register",
    "pyconnectome/scripts/pyconnectome_ants_template",
    "pyconnectome/scripts/pyconnectome_bedpostx",
    "pyconnectome/scripts/pyconnectome_compress_tractogram",
    "pyconnectome/scripts/pyconnectome_life",
    "pyconnectome/scripts/pyconnectome_metrics",
    "pyconnectome/scripts/pyconnectome_metrics_heatmap",
    "pyconnectome/scripts/pyconnectome_mitk_tractogram",
    "pyconnectome/scripts/pyconnectome_mrtrix_tractogram",
    "pyconnectome/scripts/pyconnectome_probtrackx2_complete",
    "pyconnectome/scripts/pyconnectome_probtrackx2_tractogram",
    "pyconnectome/scripts/pyconnectome_reduced_connectome",
    "pyconnectome/scripts/pyconnectome_reduced_probtrackx2",
    "pyconnectome/scripts/pyconnectome_register",
    "pyconnectome/scripts/pyconnectome_report_tractogram",
    "pyconnectome/scripts/pyconnectome_tissue_segmentation"
]
