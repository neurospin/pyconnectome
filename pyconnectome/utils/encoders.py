##########################################################################
# NSAp - Copyright (C) CEA, 2013 - 2016
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define some encoders to save/load results.
"""

# System import
import json
import numpy


class NetworkResultEncoder(json.JSONEncoder):
    """ Deal with Numpy.array and set in json.
    """
    def default(self, obj):
        # Array special case
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()

        # Set special case
        if isinstance(obj, set):
            return list(obj)

        # Call the base class default method
        return json.JSONEncoder.default(self, obj)
