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
import os
import pwd

# COMPATIBILITY: since python 3.3 mock is included in unittest module
python_version = sys.version_info
if python_version[:2] <= (3, 3):
    import mock
    from mock import patch
else:
    import unittest.mock as mock
    from unittest.mock import patch

# Package import
from pyconnectome.models.deconvolution import bedpostx
from pyconnectome.models.deconvolution import bedpostx_datacheck


class FslBedpostx(unittest.TestCase):
    """ Test the FSL diffusion deconvolution model:
    'pyconnectome.models.deconvolution.bedpostx'
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
            "subjectdir": "/my/path/mock_subjectdir",
            "n": 3,
            "w": 1,
            "b": 1000,
            "j": 1250,
            "s": 25,
            "model": 2,
            "g": True,
            "c": True,
            "rician": True,
            "fslconfig": "/my/path/mock_fslconfig",
            "fsl_parallel": True
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    def test_baddirerror_raise(self):
        """ Bad subject dir -> raise valueError.
        """
        # Test execution
        self.assertRaises(ValueError, bedpostx, **self.kwargs)

    @mock.patch("pyconnectome.models.deconvolution.os.path.isdir")
    def test_normal_execution(self, mock_isdir):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isdir.side_effect = [True, False]

        # Test execution
        login = pwd.getpwuid(os.getuid())[0]
        (outdir, merged_th, merged_ph,
         merged_f, mean_th, mean_ph,
         mean_f, mean_d, mean_S0, dyads) = bedpostx(**self.kwargs)

        self.assertEqual([
            mock.call(["which", "condor_qsub"],
                      env={"FSLPARALLEL": "condor", "USER": login},
                      stderr=-1, stdout=-1),
            mock.call(["which", "bedpostx"],
                      env={"FSLPARALLEL": "condor", "USER": login},
                      stderr=-1, stdout=-1),
            mock.call(["bedpostx",
                       self.kwargs["subjectdir"],
                       "-n", str(self.kwargs["n"]),
                       "-w", str(self.kwargs["w"]),
                       "-b", str(self.kwargs["b"]),
                       "-j", str(self.kwargs["j"]),
                       "-s", str(self.kwargs["s"]),
                       "-model", str(self.kwargs["model"]),
                       "--rician",
                       "-g",
                       "-c"],
                      cwd=None, env={"FSLPARALLEL": "condor", "USER": login},
                      stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)


class FslBedpostxDatacheck(unittest.TestCase):
    """ Test the FSL diffusion deconvolution model:
    'pyconnectome.models.deconvolution.bedpostx_datacheck'
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
            "data_dir": "/my/path/mock_data_dir",
            "fslconfig": "/my/path/mock_fslconfig"
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    @mock.patch("pyconnectome.models.deconvolution.os.path.isdir")
    def test_badfileerror_raise(self, mock_isdir):
        """Bad data dir -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isdir.side_effect = [False]

        # Test execution
        self.assertRaises(ValueError, bedpostx_datacheck, **self.kwargs)

    @mock.patch("pyconnectome.models.deconvolution.os.path.isdir")
    def test_normal_execution(self, mock_isdir):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isdir.side_effect = [True]

        # Test execution
        bedpostx_datacheck(**self.kwargs)
        self.assertEqual([
            mock.call(["which", "bedpostx_datacheck"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["bedpostx_datacheck", self.kwargs["data_dir"]],
                      cwd=None, env={}, stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
