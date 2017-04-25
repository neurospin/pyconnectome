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

# package import
from pyconnectome.tractography.probabilist import probtrackx2


class FslProbtrackx2(unittest.TestCase):
    """ Test the FSL diffusion tensor model:
    'pyconnectome.tractography.probabilist.probtrackx2'
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
            "samples": "/my/path/mock_samples",
            "mask": "/my/path/mock_mask",
            "seed": "/my/path/mock_seed",
            "out": "fdt_paths",
            "dir": "logdir",
            "nsamples": 5000,
            "nsteps": 2000,
            "steplength": 0.5,
            "distthresh": 0.0,
            "cthr": 0.2,
            "fibthresh": 0.01,
            "sampvox": 0.0,
            "randfib": 0,
            "shfile": "/my/path/mock_shfile"}

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    @mock.patch("pyconnectome.tractography.probabilist.os.path.isfile")
    def test_badfileerror_raise(self, mock_isfile):
        """Bad input file -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isfile.side_effect = [False, True]

        # Test execution
        self.assertRaises(ValueError, probtrackx2, **self.kwargs)

    @mock.patch("pyconnectome.tractography.probabilist.os.mkdir")
    @mock.patch("pyconnectome.tractography.probabilist.os.path.isdir")
    @mock.patch("pyconnectome.tractography.probabilist.os.path.isfile")
    def test_normal_execution(self, mock_isfile, mock_isdir, mock_mkdir):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isfile.side_effect = [True, True]

        # Test execution
        returned_files = probtrackx2(**self.kwargs)
        self.assertEqual([
            mock.call(["which", "probtrackx2"],
                      env={}, stderr=-1, stdout=-1),
            mock.call(["probtrackx2",
                       "-s", self.kwargs["samples"],
                       "-m", self.kwargs["mask"],
                       "-x", self.kwargs["seed"],
                       "--out=%s" % self.kwargs["out"],
                       "--dir=%s" % self.kwargs["dir"],
                       "--nsamples=%i" % self.kwargs["nsamples"],
                       "--nsteps=%i" % self.kwargs["nsteps"],
                       "--steplength=%f" % self.kwargs["steplength"],
                       "--distthresh=%f" % self.kwargs["distthresh"],
                       "--cthr=%f" % self.kwargs["cthr"],
                       "--fibthresh=%f" % self.kwargs["fibthresh"],
                       "--sampvox=%f" % self.kwargs["sampvox"],
                       "--randfib=%i" % self.kwargs["randfib"]],
                      cwd=None, env={}, stderr=-1, stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)
        self.assertEqual(len(returned_files), 2)


if __name__ == "__main__":
    unittest.main()
