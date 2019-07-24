##########################################################################
# NSAp - Copyright (C) CEA, 2013
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Modules that provides tools to wrap external softwares.
"""


# System import
import os
import pwd
import json
import warnings
import subprocess


# Module import
from .info import FSL_RELEASE
from .info import DEFAULT_FSL_PATH
from .configuration import environment
from .configuration import concat_environment
from .exceptions import FSLRuntimeError
from .exceptions import FSLDependencyError
from .exceptions import FSLConfigurationError


class FSLWrapper(object):
    """ Parent class for the wrapping of FSL functions.
    """
    output_ext = {
        "NIFTI_PAIR": ".hdr",
        "NIFTI": ".nii",
        "NIFTI_GZ": ".nii.gz",
        "NIFTI_PAIR_GZ": ".hdr.gz",
    }

    def __init__(self, cmd=None, shfile=DEFAULT_FSL_PATH, fsl_parallel=False,
                 env=None):
        """ Initialize the FSLWrapper class by setting properly the
        environment.

        Parameters
        ----------
        cmd: list of str (optional, default None)
            The FreeSurfer command to execute.
        shfile: str (optional, default NeuroSpin path)
            The path to the FSL 'fsl.sh' configuration file.
        fsl_parallel: bool (optional, default False)
            If set use Condor to parallelize FSL on your local workstation.
        env: dict (optional, default None)
            The current environment in which the FSL command will be executed.
            Default None, an empty environment.
        """
        self.cmd = cmd
        self.version = None
        self.shfile = shfile
        self.environment = self._fsl_version_check()
        if env is not None:
            self.environment = concat_environment(self.environment, env)
        # self.environment["FSLOUTPUTTYPE"] = "NIFTI_GZ"

        # Condor specific setting
        if fsl_parallel:
            self.environment["FSLPARALLEL"] = "condor"
            self.environment["USER"] = pwd.getpwuid(os.getuid())[0]
            process = subprocess.Popen(
                ["which", "condor_qsub"],
                env=self.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            self.stdout, self.stderr = process.communicate()
            self.exitcode = process.returncode
            if self.exitcode != 0:
                raise FSLDependencyError("condor_qsub", "Condor")

    def __call__(self, cmd=None, cwdir=None):
        """ Run the FSL command.

        Parameters
        ----------
        cmd: list of str (optional, default None)
            The FreeSurfer command to execute.
        cwdir: str (optional, default None)
            the working directory that will be passed to the subprocess.
        """
        # Set command
        if cmd is not None:
            self.cmd = cmd
        if self.cmd is None:
            raise FSLConfigurationError("None")

        # Check FSL has been configured so the command can be found
        process = subprocess.Popen(["which", self.cmd[0]],
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.stdout, self.stderr = process.communicate()
        self.exitcode = process.returncode
        if self.exitcode != 0:
            raise FSLConfigurationError(self.cmd[0])

        # Execute the command
        process = subprocess.Popen(self.cmd,
                                   env=self.environment,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   cwd=cwdir)
        self.stdout, self.stderr = process.communicate()
        self.exitcode = process.returncode

        # Raise exception of exitcode is not zero
        if self.exitcode != 0:
            raise FSLRuntimeError(
                self.cmd[0], " ".join(self.cmd[1:]), self.stderr + self.stdout)

    def _environment(self):
        """ Return a dictionary of the environment needed by FSL binaries.

        In order not to parse the configuration ntimes, it is stored in the
        'FREESURFER_CONFIGURED' environment variable.
        """
        # Check if FSL has already been configures
        env = os.environ.get("FSL_CONFIGURED", None)

        # Configure FSL
        if env is None:

            # FSL directory
            fsldir = os.environ.get("FSLDIR", None)
            env = {}
            if fsldir is not None:
                env["FSLDIR"] = fsldir

            # Parse configuration file
            env = environment(self.shfile)

            # Save the result
            os.environ["FSL_CONFIGURED"] = json.dumps(env)

        # Load configuration
        else:
            env = json.loads(env)

        return env

    def _fsl_version_check(self):
        """ Check that a tested FreeSurfer version is installed. This method
        also returns the FSL environment.

        Returns
        -------
        environment: dict
            The configured FSL environment.
        """
        # If a configuration file is passed
        if os.path.isfile(self.shfile):

            # Parse FSL environment
            environment = self._environment()

            # Check FSL version
            version_file = os.path.join(environment["FSLDIR"], "etc",
                                        "fslversion")
            if not os.path.isfile(version_file):
                message = ("Can't detect 'FSL' version from version file "
                           "'{0}'. You have not provided a valid "
                           "configuration file.".format(version_file))
                raise ValueError(message)
            else:
                self.version = open(version_file).read().strip("\n")
                if self.version != FSL_RELEASE:
                    message = ("Installed '{0}' version of FSL "
                               "not tested. Currently supported version "
                               "is '{1}'.".format(self.version,
                                                  FSL_RELEASE))
                    warnings.warn(message)

        # Configuration file is not a file
        else:
            message = ("'{0}' is not a valid file, can't configure "
                       "FSL.".format(self.shfile))
            raise ValueError(message)

        return environment
