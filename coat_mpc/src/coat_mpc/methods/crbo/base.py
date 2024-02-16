#  Confidence Region Bayesian Optimization -- Reference Implementation
#  Copyright (c) 2020 Robert Bosch GmbH
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.


class AcquisitionFunctionBase(object):
    """Base class for acquisition functions defining the interface."""

    def __init__(self):
        pass

    def __call__(self, x, gp, jac=False):
        """This implements the specific acquisition function."""
        raise NotImplementedError(
            "This is an abstract base class. Therefore, the __call__ method is not implemented."
        )

    def setup(self, gp):
        """Run this if you want to perform some step before the optimization (default: do nothing)"""
        pass
