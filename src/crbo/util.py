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
import sobol_seq


def sobol_sequence(n, lb, ub):
    """Generate a sobol sequence within upper and lower bounds."""
    assert lb.shape[0] == ub.shape[0]
    x0 = sobol_seq.i4_sobol_generate(lb.shape[0], n)
    return lb + x0 * (ub - lb)


def project_into_bounds(x, bounds):
    """Makes sure that the points in `x` are within the bounds. If not, project onto them."""
    return np.clip(x.copy(), bounds.lb, bounds.ub)


def normalize_input(x, bounds):
    """Map input `x` in `bounds` to unit square [0, 1]^d."""
    x = np.atleast_2d(x.copy())
    return (x - bounds.lb) / (bounds.ub - bounds.lb)


def denormalize_input(x, bounds):
    """Map input `x` in unit square [0, 1]^d to `bounds`."""
    x = np.atleast_2d(x.copy())
    return x * (bounds.ub - bounds.lb) + bounds.lb


def normalize_output(y, mode):
    """Normalize values in `y` such that var(y) = 1.0. `mode` determines the shift."""
    if mode == "neutral":
        mu = np.median(y)
    elif mode == "optimistic":
        mu = np.min(y)
    elif mode == "pessimistic":
        mu = np.max(y)
    elif mode is None:
        mu = np.median(y)
        return y.copy() - mu, mu, 1.0
    else:
        raise ValueError("Please specify valid normalizer_type.")

    std = np.std(y)
    std = 1.0 if std < 1e-6 else std
    y0 = (y.copy() - mu) / std
    return y0, mu, std


def denormalize_output(y, mu, std):
    """Denormalize values in `y`. Shifting by `mu` and scaling by `std`."""
    return mu + y.copy() * std


def mlp_model_to_theta(model):
    """Extract weights of last linear layer from the model's policy.

    :param model: (BaseRLModel) Model with the MLP policy.
    :return: (np.ndarray) Vector containing the policy parameters.
    """
    param_dict = model.get_parameters()
    w = param_dict["model/pi/w:0"].reshape(-1, 1)
    b = param_dict["model/pi/b:0"].reshape(-1, 1)
    theta = np.vstack((w, b)).T
    return theta
