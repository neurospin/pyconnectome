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
else:
    import unittest.mock as mock
    from unittest.mock import patch

# Package import
from pyconnectome.utils.regtools import flirt


class FslFlirt(unittest.TestCase):
    """ Test the FSL rigid/affine registration:
    'pyconnectome.utils.regtools.flirt'
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
            "in_file": "/my/path/mock_in_file",
            "ref_file": "/my/path/mock_ref_file",
            "omat": None,
            "out": None,
            "init": None,
            "cost": "corratio",
            "anglerep": "euler",
            "usesqform": False,
            "displayinit": False,
            "bins": 256,
            "interp": "trilinear",
            "dof": 12,
            "applyxfm": False,
            "applyisoxfm": None,
            "verbose": 0,
            "shfile": "/my/path/mock_shfile"
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    def test_badfileerror_raise(self):
        """ Bad input file -> raise valueError.
        """
        # Test execution
        self.assertRaises(ValueError, flirt, **self.kwargs)

    @mock.patch("pyconnectome.utils.regtools.os.path.isfile")
    def test_normal_execution(self, mock_isfile):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isfile.side_effect = [True, True, False]

        # Test execution
        out, omat = flirt(**self.kwargs)
        self.assertEqual([
            mock.call(["which", "flirt"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["flirt",
                       "-in", self.kwargs["in_file"],
                       "-ref", self.kwargs["ref_file"],
                       "-cost", self.kwargs["cost"],
                       "-searchcost", self.kwargs["cost"],
                       "-anglerep", self.kwargs["anglerep"],
                       "-bins", str(self.kwargs["bins"]),
                       "-interp", self.kwargs["interp"],
                       "-dof", str(self.kwargs["dof"]),
                       "-verbose", str(self.kwargs["verbose"]),
                       "-out", out,
                       "-omat", omat],
                      cwd=None, env={}, stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)


if __name__ == "__main__":
    unittest.main()
