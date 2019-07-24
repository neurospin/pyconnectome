##########################################################################
# NSAP - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Segmentation utilities.
"""

# System import
import os
import glob
import numpy
import nibabel
import subprocess

# Package import
from pyconnectome import DEFAULT_FSL_PATH
from pyconnectome.wrapper import FSLWrapper

# PyFreeSurfer import
from pyfreesurfer.utils.filetools import get_or_check_path_of_freesurfer_lut


def white_matter_interface(
        t1_brain_file,
        outdir,
        tempdir,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Generate a probabilist mask image appropriate for seeding streamlines
    on the grey matter-white matter interface using MRtrix.
    This mask deals with partial volume effects.

    Parameters
    ----------
    t1_brain_file: str
        the anatomical file used to generate the probabilist mask using MRtrix
        and FSL FAST.
    outdir: str
        the destination folder.
    tempdir: str
        a temporary folder for MRtrix with sufficient space.
    fsl_sh: str, default DEFAULT_FSL_PATH
        the FSL configuration script.

    Returns
    -------
    gmwmi_mask_file: str
        the generated probabilist seed mask.
    """
    # 5 tissue types segmentation
    # Generate the 5TT image based on a FSL FAST
    five_tissues_file = os.path.join(outdir, "5TT.nii.gz")
    cmd = ["5ttgen", "fsl", t1_brain_file, five_tissues_file, "-premasked",
           "-tempdir", tempdir, "-nocrop"]
    process = FSLWrapper(env=os.environ, shfile=fsl_sh)
    process(cmd=cmd)

    # Generate probabilist seed mask
    gmwmi_mask_file = os.path.join(outdir, "gmwmi_mask.nii.gz")
    cmd = ["5tt2gmwmi", five_tissues_file, gmwmi_mask_file]
    subprocess.check_call(cmd)

    return gmwmi_mask_file


def fix_freesurfer_subcortical_parcellation(
        parc,
        t1_brain,
        lut,
        output,
        tempdir=None,
        nb_threads=None,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Use the MRtrix labelsgmfix command to correct the FreeSurfer
    subcortical parcellation.
    It uses FSL First to recompute 5 subcortical structures.

    Parameters
    ----------
    parc: str
        Path to the FreeSurfer parcellation, generally aparc+aseg or
        aparc.a2009s+aseg.
    t1_brain: str
        Path to the T1 brain-only image on which recompute the segmentation.
    lut: str
        Path to the Look Up Table. If you haven't change the labels it should
        be FreeSurfer LUT (FreeSurferColorLUT.txt in $FREESURFER_HOME dir).
    output: str
        Path to output fixed parcellation.
    tempdir: str, default None
        Directory that MRtrix will use as temporary directory.
    nb_threads: int, default None
        Number of threads that MRtrix is allowed to use.
    fsl_sh: str, default DEFAULT_FSL_PATH
        The FSL configuration script.

    Returns
    -------
    output: str
        Path to output fixed parcellation.
    """
    cmd = ["labelsgmfix", parc, t1_brain, lut, output, "-premasked"]
    if tempdir is not None:
        cmd += ["-tempdir", tempdir]
    if nb_threads is not None:
        cmd += ["-nthreads", "%i" % nb_threads]
    fsl_process = FSLWrapper(env=os.environ, shfile=fsl_sh)
    fsl_process(cmd=cmd)

    return output


def roi_from_bbox(
        input_file,
        bbox,
        output_file):
    """ Create a ROI image from a bounding box.

    Parameters
    ----------
    input_file: str
        the reference image where the bbox is defined.
    bbox: 6-uplet
        the corner of the bbox in voxel coordinates: xmin, xmax, ymin, ymax,
        zmin, zmax.
    output_file: str
        the desired ROI image file.
    """
    # Load the reference image and generate a grid
    im = nibabel.load(input_file)
    xv, yv, zv = numpy.meshgrid(
        numpy.linspace(0, im.shape[0] - 1, im.shape[0]),
        numpy.linspace(0, im.shape[1] - 1, im.shape[1]),
        numpy.linspace(0, im.shape[2] - 1, im.shape[2]))
    xv = xv.astype(int)
    yv = yv.astype(int)
    zv = zv.astype(int)

    # Intersect the grid with the bbox
    xa = numpy.bitwise_and(xv >= bbox[0], xv <= bbox[1])
    ya = numpy.bitwise_and(yv >= bbox[2], yv <= bbox[3])
    za = numpy.bitwise_and(zv >= bbox[4], zv <= bbox[5])

    # Generate bbox indices
    indices = numpy.bitwise_and(numpy.bitwise_and(xa, ya), za)

    # Generate/save ROI
    roi = numpy.zeros(im.shape, dtype=int)
    roi[xv[indices].tolist(), yv[indices].tolist(), zv[indices].tolist()] = 1
    roi_im = nibabel.Nifti1Image(roi, affine=im.get_affine())
    nibabel.save(roi_im, output_file)


def robustfov(
        input_file,
        output_file,
        brain_size=170,
        matrix_file=None,
        fsl_sh=DEFAULT_FSL_PATH):
    """ Reduce FOV of image to remove lower head and neck.
    It is based on FSL robustfov command.

    Parameters
    ----------
    input_file: str
        the file to be cropped.
    output_file: str
        the cropped file name.
    brain_size: float (default 170)
        the size of brain in z-dimension (in mm).
    matrix_file: str (default None)
        if set, write the transformation matrix.
    fsl_sh: str, default DEFAULT_FSL_PATH
        The FSL configuration script.
    """
    # Check input parameters
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(input_file))

    # Define the FSL command
    cmd = ["robustfov", "-b", str(brain_size)]
    if matrix_file is not None:
        cmd += ["-m", matrix_file]
    cmd += ["-i", input_file, "-r", output_file]

    # Call FSL robustfov
    fslprocess = FSLWrapper(shfile=fsl_sh)
    fslprocess(cmd=cmd)


def fast(input_file, out_fileroot, klass=3, im_type=1, segments=False,
         bias_field=True, bias_corrected_im=True, probabilities=False,
         shfile=DEFAULT_FSL_PATH):
    """ FAST (FMRIB's Automated Segmentation Tool) segments a 3D image of
    the brain into different tissue types (Grey Matter, White Matter, CSF,
    etc.), whilst also correcting for spatial intensity variations (also
    known as bias field or RF inhomogeneities).
    The underlying method is based on a hidden Markov random field model and
    an associated Expectation-Maximization algorithm.

    Usage: fast [options] file(s)

    Parameters
    ----------
    input_file: str (mandatory)
        the image to be segmented.
    out: str (mandatory)
        output basename.
    klass: int (optional, default 3)
        number of tissue-type classes.
    im_type: int (optional, default 1)
        type of image 1=T1, 2=T2, 3=PD.
    segments: bool (optional, default False)
        outputs a separate binary image for each tissue type.
    bias_field: bool (optional, default True)
        output estimated bias field.
    bias_corrected_im: bool (optional, default True)
        output bias-corrected image.
    probabilities: bool (optional, default False)
        outputs individual probability maps.
    shfile: str (optional, default local path)
        the path to the FSL 'fsl.sh' configuration file.

    Returns
    -------
    tpm: list of str
        the generated tissue preobaility maps.
    tsm: list of str
        the generated tissue segmentation maps.
    segmentation_file: str
        the segmented tissues.
    bias_file: str
        the bias field.
    biascorrected_file: str
        the bias corrected input image.
    """
    # Check input parameters
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(input_file))

    # Define the FSL command
    bool_params = {
        "-g": segments,
        "-b": bias_field,
        "-B": bias_corrected_im,
        "-p": probabilities
    }
    cmd = ["fast", "-o", out_fileroot, "-n", str(klass), "-t", str(im_type)]
    for name, value in bool_params.items():
        if value:
            cmd.append(name)
    cmd.append(input_file)

    # Call FSL fast
    fslprocess = FSLWrapper(shfile=shfile)
    fslprocess(cmd=cmd)

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not declared in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Format outputs
    image_ext = fslprocess.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]
    segmentation_file = out_fileroot + "_seg" + image_ext
    bias_file = out_fileroot + "_bias" + image_ext
    if not os.path.isfile(bias_file):
        bias_file = None
    biascorrected_file = out_fileroot + "_restore" + image_ext
    if not os.path.isfile(biascorrected_file):
        biascorrected_file = None
    tpm = glob.glob(out_fileroot + "_prob_*")
    tsm = glob.glob(out_fileroot + "_pve_*")

    return tpm, tsm, segmentation_file, bias_file, biascorrected_file


def bet2(input_file, output_fileroot, outline=False, mask=False,
         skull=False, nooutput=False, f=0.5, g=0, radius=None, smooth=None,
         c=None, threshold=False, mesh=False, shfile=DEFAULT_FSL_PATH):
    """ Wraps command bet2.

    Deletes non-brain tissue from an image of the whole head. It can also
    estimate the inner and outer skull surfaces, and outer scalp surface,
    if you have good quality T1 and T2 input images.

    The basic usage is:
        bet2 <input_fileroot> <output_fileroot> [options]

    Parameters
    ----------
    input_file: (mandatory)
        Input image.
    output_fileroot: (mandatory)
        Output directory.
    outline: bool (optional, default False)
        Generate brain surface outline overlaid onto original image.
    mask: bool (optional, default False)
        Generate binary brain mask.
    skull: bool (optional, default False)
        Generate approximate skull image.
        (not as clean as what betsurf generates).
    nooutput: bool (optional, default False)
        Don't generate segmented brain image output.
    f: float (optional, default 0.5)
        Fractional intensity threshold (0->1).
        Smaller values give larger brain outline estimates.
    g: int (optional, default 0)
        Vertical gradient in fractional intensity threshold (-1->1).
        Positive values give larger brain outline at bottom, smaller at top.
    radius: (optional)
        Head radius (mm not voxels).
        Initial surface sphere is set to half of this.
    smooth : float (optional, default 1)
        Smoothness factor.
        Values smaller than 1 produce more detailed brain surface, values
        larger than one produce smoother, less detailed surface.
    c: (optional)
        Centre-of-gravity (voxels not mm) of initial mesh surface (x, y, z).
    threshold: bool (optional, default False)
        Apply thresholding to segmented brain image and mask.
    mesh: bool (optional, default False).
        Generates brain surface as mesh in .vtk format.
    shfile: (optional, default DEFAULT_FSL_PATH)
        The FSL configuration batch.


    Returns
    -------
    output: str
        the extracted brain volume.
    mask_file: str
        the binary mask of the extracted brain volume.
    mesh_file: str
        the brain surface as a vtk mesh.
    outline_file: str
        the brain surface outline overlaid onto original image.
    inskull_mask_file, inskull_mesh_file,
    outskull_mask_file, outskull_mesh_file,
    outskin_mask_file, outskin_mesh_file,
    skull_mask_file: str
        rough skull image.
    shfile: str (optional, default local path)
        the path to the FSL 'fsl.sh' configuration file.
    """
    # Check the input parameter
    if not os.path.isfile(input_file):
        raise ValueError("'{0}' is not a valid input file.".format(
                         input_file))

    # Check that the output directory exists
    if not os.path.isdir(output_fileroot):
        os.mkdir(output_fileroot)
    output_fileroot = os.path.join(
        output_fileroot,
        os.path.basename(input_file).split(".")[0] + "_brain")

    # Define the FSL command
    cmd = ["bet",
           input_file,
           output_fileroot,
           "-f", str(f),
           "-g", str(g),
           "-R"]

    # Set optional arguments
    if outline:
        cmd += ["-o"]
    if mask:
        cmd += ["-m"]
    if skull:
        cmd += ["-s"]
    if nooutput:
        cmd += ["-n"]
    if mesh:
        cmd += ["-e"]
    if threshold:
        cmd += ["-t"]

    if c is not None:
        cmd += ["-c", c]
    if radius is not None:
        cmd += ["-r", radius]
    if smooth is not None:
        cmd += ["-s", smooth]

    # Call bet2
    fslprocess = FSLWrapper(shfile=shfile)
    fslprocess(cmd=cmd)

    # Check the FSL environment variable
    if "FSLOUTPUTTYPE" not in fslprocess.environment:
        raise ValueError("'{0}' variable not declared in FSL "
                         "environ.".format("FSLOUTPUTTYPE"))

    # Format outputs
    image_ext = fslprocess.output_ext[fslprocess.environment["FSLOUTPUTTYPE"]]

    # Create format outputs
    if outline:
        outline_file = output_fileroot + "_outline" + image_ext
    else:
        outline_file = None

    if mask:
        mask_file = output_fileroot + "_mask" + image_ext
    else:
        mask_file = None

    if skull:
        inskull_mask_file = output_fileroot + "_inskull_mask" + image_ext
        inskull_mesh_file = output_fileroot + "_inskull_mesh" + image_ext
        outskull_mask_file = output_fileroot + "_outskull_mask" + image_ext
        outskull_mesh_file = output_fileroot + "_outskull_mesh" + image_ext
        outskin_mask_file = output_fileroot + "_outskin_mask" + image_ext
        outskin_mesh_file = output_fileroot + "_outskin_mesh" + image_ext
        skull_mask_file = output_fileroot + "_skull_mask" + image_ext
    else:
        inskull_mask_file = None
        inskull_mesh_file = None
        outskull_mask_file = None
        outskull_mesh_file = None
        outskin_mask_file = None
        outskin_mesh_file = None
        skull_mask_file = None

    if nooutput:
        output = None
    else:
        output_fileroot += image_ext
        output = output_fileroot

    if mesh:
        mesh_file = output_fileroot + "_mesh.vtk"
    else:
        mesh_file = None

    return (output, mask_file, mesh_file, outline_file, inskull_mask_file,
            inskull_mesh_file, outskull_mask_file, outskull_mesh_file,
            outskin_mask_file, outskin_mesh_file, skull_mask_file)


def get_region_names_of_lausanne_2008_atlas():
    """ Get the ordered region names of the Lausanne 2008 atlas as in
    standard papers.

    It corresponds to the Desikan atlas in the cortex, without the corpus
    callosum along with 7 subcortical regions.

    Returns
    -------
    atlas_names: list of str
        Ordered region names of the Lausanne 2008 atlas.
    """
    # All left cortical regions of the Desikan atlas except the corpus callosum
    lh_ctx_rois = [
        'ctx-lh-lateralorbitofrontal',
        'ctx-lh-parsorbitalis',
        'ctx-lh-frontalpole',
        'ctx-lh-medialorbitofrontal',
        'ctx-lh-parstriangularis',
        'ctx-lh-parsopercularis',
        'ctx-lh-rostralmiddlefrontal',
        'ctx-lh-superiorfrontal',
        'ctx-lh-caudalmiddlefrontal',
        'ctx-lh-precentral',
        'ctx-lh-paracentral',
        'ctx-lh-rostralanteriorcingulate',
        'ctx-lh-caudalanteriorcingulate',
        'ctx-lh-posteriorcingulate',
        'ctx-lh-isthmuscingulate',
        'ctx-lh-postcentral',
        'ctx-lh-supramarginal',
        'ctx-lh-superiorparietal',
        'ctx-lh-inferiorparietal',
        'ctx-lh-precuneus',
        'ctx-lh-cuneus',
        'ctx-lh-pericalcarine',
        'ctx-lh-lateraloccipital',
        'ctx-lh-lingual',
        'ctx-lh-fusiform',
        'ctx-lh-parahippocampal',
        'ctx-lh-entorhinal',
        'ctx-lh-temporalpole',
        'ctx-lh-inferiortemporal',
        'ctx-lh-middletemporal',
        'ctx-lh-bankssts',
        'ctx-lh-superiortemporal',
        'ctx-lh-transversetemporal',
        'ctx-lh-insula'
    ]

    # Same for right hemisphere
    rh_ctx_rois = [x.replace("ctx-lh-", "ctx-rh-") for x in lh_ctx_rois]

    # Ordered left subcortical regions of Lausanne 2008 scale 33 atlas
    lh_subctx_rois = [
        'Left-Thalamus-Proper',
        'Left-Caudate',
        'Left-Putamen',
        'Left-Pallidum',
        'Left-Accumbens-area',
        'Left-Hippocampus',
        'Left-Amygdala',
    ]

    # Ordered right subcortical regions
    rh_subctx_rois = [x.replace("Left-", "Right-") for x in lh_subctx_rois]

    # Non-hemispheric subcortical region
    axial_subctx_rois = ['Brain-Stem']

    atlas_names = (["Unknown"] + lh_ctx_rois + lh_subctx_rois + rh_ctx_rois +
                   rh_subctx_rois + axial_subctx_rois)

    return atlas_names


def create_lausanne2008_lut(outdir, freesurfer_lut=None):
    """ Create a Look Up Table for the Lausanne2008 atlas. It has the same
    format as the FreeSurfer LUT (FreeSurferColorLUT.txt), but it lists only
    the regions of the Lausanne2008 atlas and the integer labels are the
    row/col positions of the regions in the connectome.

    Parameters
    ----------
    outdir: str
        Path to directory where to write "Lausanne2008LUT.txt"
    freesurfer_lut: str, default None
        Path to the FreeSurfer Look Up Table. If not passed, try to use
        $FREESURFER_HOME/FreeSurferColorLUT.txt. If not found raise Exception.
    """

    # Ordered ROIs (i.e. nodes of the connectome) of the Lausanne 2008 atlas
    roi_names = get_region_names_of_lausanne_2008_atlas()

    # Path to the FreeSurfer LUT
    freesurfer_lut = get_or_check_path_of_freesurfer_lut(freesurfer_lut)

    # Load table
    table = numpy.loadtxt(freesurfer_lut, dtype=str)

    # Keep rows that corresponds to regions of the atlas
    table = numpy.array([r for r in table if r[1] in set(roi_names)])

    # Order rows (i.e. regions) of the LUT like Lausanne2008 atlas
    table = numpy.array(sorted(table, key=lambda r: roi_names.index(r[1])))

    # Replace FreeSurfer label by row/col position in connectome
    table[:, 0] = numpy.arange(table.shape[0])

    # Header lines
    header_1 = "# Look up Table for Lausanne 2008 atlas\n"
    header_2 = "#<Label> <Label Name> <R> <G> <B> <A>\n"

    # Save as .txt file
    lausanne2008_lut = os.path.join(outdir, "Lausanne2008LUT.txt")
    with open(lausanne2008_lut, "wt") as f:
        f.write(header_1)
        f.write(header_2)
        # Maintain the indentation
        line_format = "{0: <8} {1: <50} {2: <4} {3: <4} {4: <4} {5: <4}\n"
        for i, row in enumerate(table, start=1):
            f.write(line_format.format(*row))

    return lausanne2008_lut
