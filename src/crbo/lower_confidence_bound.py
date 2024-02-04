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

from crbo.base import AcquisitionFunctionBase


class LowerConfidenceBound(AcquisitionFunctionBase):
    def __init__(self, beta):
        """
        Lower Confidence Bound (LCB) acquisition function, lcb(x) = mu(x) - beta * sqrt(var(x))
        """
        self.__name__ = "lcb"
        self.beta = np.maximum(0.0, beta)

    def __call__(self, x, gp, jac=False):
        """
        Parameters
        ----------
        x : ndarray, shape(n, )
            Location to evaluate the acquisition function.
        gp : Gaussian process object
            Current model of the objective function.
        jac : bool, optional
            Flag if the Jacobian should be returned as well (see Notes)

        Returns
        -------
        if jac==False, double
            Function value of the acquisition function
        if jac==True, tuple (double, ndarray)
            Function value and Jacobian of the acquisition function

        Notes
        -----
        Neat interfacing trick when using scipy.optimize.minimize:
        From the scipy documentation: "If jac is a Boolean and is True, fun is assumed to return the
        gradient along with the objective function. If False, the gradient will be estimated using
        ‘2-point’ finite difference estimation."
        """
        x = np.atleast_2d(x)
        assert x.shape[1] == gp.kern.input_dim

        mean, var = gp.predict(x, include_likelihood=False)
        var = np.clip(var, 1e-10, np.inf)  # For the Jacobian we divide by sqrt(var)
        std = np.sqrt(var)
        f_val = np.squeeze(mean - self.beta * std)

        if not jac:
            return f_val
        else:
            dmean, dvar = gp.predictive_gradients(x)
            dmean = dmean[:, :, 0]
            dstd = dvar / (2 * std)
            f_jac = np.squeeze(dmean - self.beta * dstd)
            return f_val, f_jac
