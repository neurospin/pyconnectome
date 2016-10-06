# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:51:37 2016

@author: tg244389
"""

import os
import argparse

from pyfsl.connectograms.reduced import (get_path_of_freesurfer_lut,
                                         mrtrix_connectome_pipeline)


DOC = """
Compute the connectome of a given parcellation, like the Freesurfer aparc+aseg
segmentation, using MRtrix.

Requirements:
    - preprocessed DWI with bval and bvec: if distortions from acquisition
      have been properly corrected it should be alignable to the T1 with a
      rigid transformation.
    - diffusion brain mask: nodif_brain_mask
    - parcellation: image of labeled regions, e.g. Freesurfer aparc+aseg

Connectogram strategy:
    <TO DO>
"""


def is_file(filepath):
    """ Check file's existence - argparse 'type' argument.
    """
    if not os.path.isfile(filepath):
        raise argparse.ArgumentError("File does not exist: %s" % filepath)
    return filepath


def get_cmd_line_args():
    """
    Create a command line argument parser and return a dict mapping
    <argument name> -> <argument value>.
    """

    usage = ("%(prog)s -a <t1 brain> -i <dwi> -b <bvals> -r <bvecs> "
             "-m <nodif_brain_mask> -p <parc> -o <outdir> -d <tempdir> "
             "-t <int> -z <int> -l <int> -s <float> -n <int> [options]")
    parser = argparse.ArgumentParser(prog="python mrtrix_connectome.py",
                                     usage=usage, description=DOC)

    # Required arguments

    parser.add_argument("-a", "--t1-brain", type=is_file, required=True,
                        metavar="<path>",
                        help="Path to the brain-only T1 (anatomy).")

    parser.add_argument("-i", "--dwi", type=is_file, required=True,
                        metavar="<path>", help="Path to the diffusion data.")

    parser.add_argument("-b", "--bvals", type=is_file, required=True,
                        metavar="<path>", help="Path to the bvalue list.")

    parser.add_argument("-r", "--bvecs", type=is_file, required=True,
                        metavar="<path>", help="Path to the list of diffusion-"
                                               "sensitized directions.")

    parser.add_argument("-m", "--nodif-brain-mask", type=is_file,
                        required=True, metavar="<path>",
                        help="Diffusion brain binary mask.")

    parser.add_argument("-p", "--parc", type=is_file, required=True,
                        metavar="<path>",
                        help="Parcellation that defines the nodes of the "
                             "connectome, e.g. 'aparc+aseg' from Freesurfer.")

    parser.add_argument("-o", "--outdir", required=True, metavar="<path>",
                        help="Directory where to output.")

    parser.add_argument("-d", "--tempdir", required=True, metavar="<path>",
                        help="Where to write temporary directories e.g. /tmp.")

    parser.add_argument("-t", "--mtracks", required=True, type=int,
                        metavar="<int>",
                        help="Number of millions of tracks to compute.")

    parser.add_argument("-z", "--sift-mtracks", required=True, type=int,
                        metavar="<int>",
                        help="Number of millions of tracks to keep with SIFT.")

    parser.add_argument("-l", "--maxlength", required=True, type=int,
                        metavar="<int>", help="Max fiber length in mm.")

    parser.add_argument("-s", "--cutoff", required=True, type=float,
                        metavar="<float>", help="Cutoff; stopping criteria.")

    parser.add_argument("-n", "--nthreads", required=True, type=int,
                        metavar="<int>", help="Number of threads.")

    # Optional arguments

    parser.add_argument("-f", "--parc-lut", type=is_file,
                        default=get_path_of_freesurfer_lut(), metavar="<path>",
                        help="Path to the Look Up Table for the passed parcel"
                             "lation in the Freesurfer LUT format. By default "
                             "$FREESURFER_HOME/FreeSurferColorLUT.txt.")

    chelp = ("Path to a Look Up Table, in the Freesurfer LUT format, listing "
             "the regions from the parcellation to use as nodes in the "
             "connectome. The integer label should be the row/col position in "
             "the connectome. By default a LUT is created for the Lausanne2008"
             " atlas, which implies that the passed parcellation is "
             "'aparc+aseg' from Freesurfer.")
    parser.add_argument("-c", "--connectome-lut", type=is_file, default=None,
                        metavar="<path>", help=chelp)

    parser.add_argument("-F", "--fsl-init", default="/etc/fsl/5.0/fsl.sh",
                        type=is_file, metavar="<path>",
                        help="Bash script initializing FSL's environment.")

    # Create a dict of arguments to pass to the 'main' function
    args = parser.parse_args()
    kwargs = vars(args)

    return kwargs


kwargs = get_cmd_line_args()
print kwargs
mrtrix_connectome_pipeline(**kwargs)
