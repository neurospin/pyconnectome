##########################################################################
# NSAp - Copyright (C) CEA, 2013-2015
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################


"""
Modules that provides tools to generate movies.
"""


# System imports
from __future__ import print_function
import os
import re
import glob
import subprocess


class ImageMagickWrapper(object):
    """ Parent class for the wrapping of ImageMagick commands.
    """
    def __init__(self, cmd):
        """ Initialize the ImageMagickWrapper class.

        Parameters
        ----------
        cmd: list of str (mandatory)
            the ImageMagick command to execute.
        """
        self.cmd = cmd
        self.available_cmd = ["animate", "compare", "composite", "conjure",
                              "convert", "display", "identify", "import",
                              "mogrify", "montage", "stream"]
        self.check_cmd()
        self.check_installation()

    def __call__(self):
        """ Run the ImageMagick command.

        Returns
        -------
        output: str
            the ImageMagick process function.
        """
        try:
            output = subprocess.check_output(self.cmd)
        except subprocess.CalledProcessError:
            print('***')
            print("Command {0} failed with parameters : {1}\n"
                  .format(self.cmd[0], " ".join(self.cmd[1:])))
            print('***')
            raise
        else:
            return output

    def check_installation(self):
        """ Check if ImageMagick is installed.
        """
        try:
            subprocess.check_output(["convert", "--version"])
        except OSError:
            print("ImageMagick was not found, please install it first.")
            raise

    def check_cmd(self):
        """ Check if it's an ImageMagick command.
        """
        program = self.cmd[0]
        if program not in self.available_cmd:
            raise ValueError("{0} is not a known ImageMagick command."
                             .format(program))


def images_to_gif(input_img_list, output_file, delay=100):
    """ Convert input images to a unique gif file using ImageMagick.

    Parameters
    ----------
    input_img_list: list of str (mandatory)
        list of the input images to combine.
    output_file: str (mandatory)
        the output gif file path.
    delay: int (optional, default 100)
        this option is useful for regulating the animation of image
        sequences ticks/ticks-per-second seconds must expire before the
        display of the next image. The default is no delay between each
        showing of the image sequence. The default ticks-per-second is 100.
    """
    gif_extension = ".gif"
    if not output_file.endswith(gif_extension):
        output_file += gif_extension
    cmd = ["convert"]
    cmd += input_img_list
    cmd += ["+repage", "-set", "delay", "{0}".format(delay), output_file]

    # Execute the ImageMagick command
    magick = ImageMagickWrapper(cmd)
    magick()


def split_image(input_image, outdir, outprefix, outwidth, outheight):
    """ Split one image into multiple images using ImageMagick convert.

    Parameters
    ----------
    input_image: str (mandatory)
        path to the image.
    outdir: str (mandatory)
        the output directory.
    outprefix: str (mandatory)
        the output images prefix.
    outwidth: int (mandatory)
        the width in pixels of the output images.
    outheight: int (mandatory)
        the height in pixels of the output images.

    Returns
    -------
    output_images: list of str
        list of the splitted images.
    """
    inext = os.path.splitext(os.path.basename(input_image))[-1]
    outpattern = os.path.join(outdir, "{0}-%03d{1}".format(outprefix, inext))

    # Construct the ImageMagick command
    cmd = ["convert", input_image,
           "-crop", "{0}x{1}".format(outwidth, outheight),
           outpattern]

    # Execute the ImageMagick command
    magick = ImageMagickWrapper(cmd)
    magick()

    return sorted(glob.glob(outpattern.replace("%03d", "*")))


def get_image_dimensions(input_image):
    """ Get image dimensions using ImageMagick identify.

    Parameters
    ----------
    input_image: str (mandatory)
        input image.

    Returns
    -------
    width, height: (int, int)
        the image dimensions.
    """
    # Construct the ImageMagick command
    cmd = ["identify", "-ping", "-format", "'%wx%h'", input_image]

    # Execute the ImageMagick command
    magick = ImageMagickWrapper(cmd)
    dim_str = magick()
    width, height = (int(dimension) for dimension
                     in re.search("(\d+)x(\d+)", dim_str).groups())

    return width, height
