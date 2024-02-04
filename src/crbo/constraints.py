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


import numpy as np


class BaseConstraint(object):
    """Class implementing the basic interface for constraints."""

    def __init__(self):
        pass

    def __call__(self, x, *args):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def jac(self, x, *args):
        # Note: this needs to have the same signature as __call__()
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )


class ConfidenceRegionConstraint(BaseConstraint):
    """Class implementing the confidence region constraint, C = { x | gp_std(x) <= gamma}."""

    def __init__(self, gamma):
        """
        Parameters:
        -----------
        gamma : float
            Confidence parameter that specifies the size of the confidence region.
        """
        super().__init__()
        self.gamma = gamma

    def __call__(self, x, gp):
        """
        Parameters:
        -----------
        x : ndarray, shape (n, dim)
            Locations at which to evaluate the constraint.
        gp : Gaussian process object
            Current model of the objective function.

        Returns:
        --------
        fx : ndarray, shape (n,)
            Value of the constraint function evaluated at input locations `x`.

        Notes:
        ------
        Constraining the variance instead of the standard deviation is more efficient.
        """
        x = np.atleast_2d(x)
        _, var = gp.predict(x, include_likelihood=False)
        var = np.clip(var, 1e-10, np.inf)
        fx = self.gamma ** 2 * gp.kern.variance.values.squeeze() - var.squeeze()
        return fx

    def jac(self, x, gp):
        """Jacobian of the constraint function.
        
        Parameters:
        -----------
        x : ndarray, shape (n, dim)
            Locations at which to evaluate the constraint.
        gp : Gaussian process object
            Current model of the objective function.

        Returns:
        --------
        dfdx : ndarray, shape (n, dim)
            Jacobian of the constraint function evaluated at input locations `x`.
        """
        x = np.atleast_2d(x)
        _, dvar = gp.predictive_gradients(x)
        return -dvar

    def isempty(self, gp):
        """Given a GP model, check if the confidence region is an empty set. We
        do so by checking if the constraint is fulfilled at all locations of the
        GP data.

        Parameters:
        -----------
        gp : Gaussian process object
            Current model of the objective function.

        Returns:
        --------
        isempty: bool
            True if constraint is fulfilled for all GP data points, else False
        """
        return (self(gp.X, gp) < 0).any()


