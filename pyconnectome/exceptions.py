##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2015
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module custom exceptions.
"""


class FSLError(Exception):
    """ Base exception type for the package.
    """
    def __init__(self, message):
        super(FSLError, self).__init__(message)


class FSLRuntimeError(FSLError):
    """ Error thrown when call to the FSL software failed.
    """
    def __init__(self, algorithm_name, parameters, error=None):
        message = (
            "FSL call for '{0}' failed, with parameters: '{1}'. Error:: "
            "{2}.".format(algorithm_name, parameters, error))
        super(FSLRuntimeError, self).__init__(message)


class FSLConfigurationError(FSLError):
    """ Error thrown when call to the FSL software failed.
    """
    def __init__(self, command_name):
        message = "FSL command '{0}' not found.".format(command_name)
        super(FSLConfigurationError, self).__init__(message)


class FSLResultError(FSLError):
    """ Error thrown when the FSL software has returned a strange result.
    """
    def __init__(self, command):
        message = ("FSL command '{0}' may have returned a strange "
                   "result.".format(command))
        super(FSLResultError, self).__init__(message)


class FSLDependencyError(FSLError):
    """ Error thrown when the FSL software needs a dependency.
    """
    def __init__(self, command, package_name):
        message = ("FSL software uses {0}. Please make sure that {1} package "
                   "is installed ".format(command, package_name))
        super(FSLDependencyError, self).__init__(message)
