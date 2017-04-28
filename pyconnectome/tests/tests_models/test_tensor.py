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

# COMPATIBILITY: since python 3.3 mock is included in unittest module
python_version = sys.version_info
if python_version[:2] <= (3, 3):
    import mock
    from mock import patch
else:
    import unittest.mock as mock
    from unittest.mock import patch

# Package import
from pyconnectome.models.tensor import dtifit


class FslDtiFit(unittest.TestCase):
    """ Test the FSL diffusion tensor model:
    'pyconnectome.models.tensor.dtifit'
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
            "data": "/my/path/mock_data",
            "bvals": "/my/path/mock_bvals",
            "bvecs": "/my/path/mock_bvecs",
            "mask": "/my/path/mock_mask",
            "out": "/my/path/mock_out",
            "wls": True,
            "save_tensor": True,
        }

    def tearDown(self):
        """ Run after each test.
        """
        self.popen_patcher.stop()
        self.env_patcher.stop()

    @mock.patch("pyconnectome.models.tensor.os.path.isfile")
    def test_badfileerror_raise(self, mock_isfile):
        """Bad input file -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isfile.side_effect = [True, True, True, False]

        # Test execution
        self.assertRaises(ValueError, dtifit, **self.kwargs)

    @mock.patch("pyconnectome.models.tensor.os.mkdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isfile")
    def test_baddirerror_raise(self, mock_isfile, mock_isdir, mock_mkdir):
        """Bad directory -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isfile.side_effect = [True, True, True, True]
        mock_isdir.side_effect = [False]

        # Test execution
        self.assertRaises(ValueError, dtifit, **self.kwargs)

    @mock.patch("pyconnectome.models.tensor.os.mkdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isfile")
    def test_nofsltype_raise(self, mock_isfile, mock_isdir, mock_mkdir):
        """Bad FSL extension error  -> raise valueError.
        """
        # Set the mocked functions returned values
        mock_isfile.side_effect = [True, True, True, True]
        mock_isdir.side_effect = [True]
        self.mock_env.return_value = {"EXT": "NIFTI"}

        # Test execution
        self.assertRaises(ValueError, dtifit, **self.kwargs)

    @mock.patch("pyconnectome.models.tensor.os.mkdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isdir")
    @mock.patch("pyconnectome.models.tensor.os.path.isfile")
    def test_normal_execution(self, mock_isfile, mock_isdir, mock_mkdir):
        """ Test the normal behaviour of the function.
        """
        # Set the mocked function returned values.
        mock_isfile.side_effect = [True, True, True, True]
        mock_isdir.side_effect = [True]
        self.mock_env.return_value = {"FSLOUTPUTTYPE": "NIFTI"}

        # Test execution
        returned_files = dtifit(**self.kwargs)
        self.assertEqual([
            mock.call(["which", "dtifit"],
                      env={"FSLOUTPUTTYPE": "NIFTI"}, stderr=-1, stdout=-1),
            mock.call(["dtifit",
                       "-k", self.kwargs["data"],
                       "-r", self.kwargs["bvecs"],
                       "-b", self.kwargs["bvals"],
                       "-m", self.kwargs["mask"],
                       "-o", os.path.join(self.kwargs["out"], "dtifit"),
                       "--wls",
                       "--save_tensor"],
                      cwd=None, env={"FSLOUTPUTTYPE": "NIFTI"}, stderr=-1,
                      stdout=-1)],
            self.mock_popen.call_args_list)
        self.assertEqual(len(self.mock_env.call_args_list), 1)
        self.assertEqual(len(returned_files), 11)
        self.assertEqual(
            returned_files,
            (os.path.join(self.kwargs["out"], "dtifit" + "_V1.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_V2.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_V3.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_L1.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_L2.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_L3.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_MD.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_FA.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_S0.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_tensor.nii"),
             os.path.join(self.kwargs["out"], "dtifit" + "_M0.nii"))
        )


if __name__ == "__main__":
    unittest.main()
