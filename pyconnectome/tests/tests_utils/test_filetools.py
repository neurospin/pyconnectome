##########################################################################
# NSAp - Copyright (C) CEA, 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
import unittest
import sys

# COMPATIBILITY: since python 3.3 mock is included in unittest module
python_version = sys.version_info
if python_version[:2] <= (3, 3):
    import mock
    from mock import patch
    mock_builtin = "__builtin__"
else:
    import unittest.mock as mock
    from unittest.mock import patch
    mock_builtin = "builtins"

# Package import
from pyconnectome.utils.filetools import fslreorient2std, apply_mask


class Fslreorient2std(unittest.TestCase):
    """ Test the FSL reorient the image to standard:
    'pyconnectome.utils.filetools.fslreorient2std'
    """
    def setUp(self):
        """ Run before each test - the mock_popen will be available and in the
        right state in every test<something> function.
        """
        # Mocking popen
        self.popen_patcher = patch("pyconnectome.wrapper.subprocess.Popen")
        self.mock_popen = self.popen_patcher.start()
        mock_process = mock.Mock()
        attrs = {
            "communicate.return_value": ("mock_OK", "mock_NONE"),
            "returncode": 0
        }
        mock_process.configure_mock(**attrs)
        self.mock_popen.return_value = mock_process

        # Mocking set environ
        self.env_patcher = patch(
            "pyconnectome.wrapper.FSLWrapper._fsl_version_check")
        self.mock_env = self.env_patcher.start()
        self.mock_env.return_value = {}

        # Define function parameters
        self.kwargs = {
            "input_image": "/my/path/mock_input_image",
            "output_image": "/my/path/mock_output_image",
            "save_trf": True,
            "fslconfig": "/my/path/mock_shfile",
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    @mock.patch("pyconnectome.utils.filetools.os.path.isfile")
    def test_badfileerror_raise(self, mock_isfile):
        """Bad input file -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isfile.side_effect = [False]

        # Test execution
        self.assertRaises(ValueError, fslreorient2std, **self.kwargs)

    @mock.patch("{0}.open".format(mock_builtin))
    @mock.patch("numpy.savetxt")
    @mock.patch("pyconnectome.utils.filetools.flirt2aff")
    @mock.patch("pyconnectome.utils.filetools.glob.glob")
    @mock.patch("pyconnectome.utils.filetools.os.path.isfile")
    def test_normal_execution(self, mock_isfile, mock_glob, mock_aff,
                              mock_savetxt, mock_open):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isfile.side_effect = [True]
        mock_glob.return_value = ["/my/path/mock_output"]
        mock_context_manager = mock.Mock()
        mock_open.return_value = mock_context_manager
        mock_file = mock.Mock()
        mock_file.read.return_value = "WRONG"
        mock_enter = mock.Mock()
        mock_enter.return_value = mock_file
        mock_exit = mock.Mock()
        setattr(mock_context_manager, "__enter__", mock_enter)
        setattr(mock_context_manager, "__exit__", mock_exit)
        mock_aff.flirt2aff.return_value = ""

        # Test execution
        fslreorient2std(**self.kwargs)

        self.assertEqual([
            mock.call(["which", "fslreorient2std"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["fslreorient2std",
                      self.kwargs["input_image"],
                      self.kwargs["output_image"] + ".nii.gz"],
                      cwd=None, env={}, stderr=-1, stdout=-1),
            mock.call(["which", "fslreorient2std"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["fslreorient2std",
                      self.kwargs["input_image"]],
                      cwd=None, env={}, stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)


class FslApplyMask(unittest.TestCase):
    """ Test the FSL apply mask:
    'pyconnectome.utils.filetools.apply_mask'
    """
    def setUp(self):
        """ Run before each test - the mock_popen will be available and in the
        right state in every test<something> function.
        """
        # Mocking popen
        self.popen_patcher = patch("pyconnectome.wrapper.subprocess.Popen")
        self.mock_popen = self.popen_patcher.start()
        mock_process = mock.Mock()
        attrs = {
            "communicate.return_value": ("mock_OK", "mock_NONE"),
            "returncode": 0
        }
        mock_process.configure_mock(**attrs)
        self.mock_popen.return_value = mock_process

        # Mocking set environ
        self.env_patcher = patch(
            "pyconnectome.wrapper.FSLWrapper._fsl_version_check")
        self.mock_env = self.env_patcher.start()
        self.mock_env.return_value = {}

        # Define function parameters
        self.kwargs = {
            "input_file": "/my/path/mock_input_file",
            "mask_file": "/my/path/mock_mask_file",
            "output_fileroot": "/my/path/mock_output_fileroot",
            "fslconfig": "/my/path/mock_shfile",
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    def test_badfileerror_raise(self):
        """Bad input file -> raise valueError.
        """
        # Test execution
        self.assertRaises(ValueError, apply_mask, **self.kwargs)

    @mock.patch("pyconnectome.utils.filetools.glob.glob")
    @mock.patch("pyconnectome.utils.filetools.os.path.isfile")
    def test_normal_execution(self, mock_isfile, mock_glob):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isfile.side_effect = [True, True]
        mock_glob.return_value = ["/my/path/mock_output"]

        # Test execution
        apply_mask(**self.kwargs)

        self.assertEqual([
            mock.call(["which", "fslmaths"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["fslmaths",
                       self.kwargs["input_file"],
                       "-mas", self.kwargs["mask_file"],
                       self.kwargs["output_fileroot"]],
                      cwd=None, env={}, stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
