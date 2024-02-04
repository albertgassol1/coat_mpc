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


class ThompsonSampling(AcquisitionFunctionBase):
    def __init__(self):
        """
        Thompson sampling acquisition function. Uses GP posterior samples on a grid.
        """
        self.__name__ = "ts"

    def __call__(self, x, gp, jac=False):
        x = np.atleast_2d(x)
        assert x.shape[1] == gp.kern.input_dim
        assert x.shape[0] > 1  # else this does not make sense at all
        assert not jac
        f_val = gp.posterior_samples_f(x, size=1).squeeze()
        return f_val
