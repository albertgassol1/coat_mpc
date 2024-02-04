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


class LevelsetSampler(object):
    """Sample uniformly from a levelset given by f(x, *f_args) >= 0.0."""

    def __init__(self, fun, fun_args=()):
        """
        Parameters
        ----------
        fun : callable
            The constraint function defining the levelset, F = { x | f(x) >= 0.0 }.

                ``fun(x, *fun_args) -> ndarray``

            where x is an 2-D array with shape (n, dim) and `fun_args` is a tuple of the fixed
            parameters need to completely specify the function.
        fun_args : tuple, optional
            Extra arguments passed to the constraint function and its derivatives.
        """
        self._fun = fun
        self._fun_args = fun_args
        self.f = self._get_constraint_wrapper()

    def __call__(self, n, *args):
        """Draws `n` samples from the levelset uniformly at random.
                
        Parameters
        ----------
        n : int
            Number of samples
        args : optional arguments
            Additional parameters used by the specific implementation of the sampler.

        Returns
        -------
        x_samples : ndarray, shape (n, dim)
            Array containing the desired number of samples.
        """
        raise NotImplementedError("Implement this in a sub-class.")

    @property
    def fun_args(self):
        return self._fun_args

    @fun_args.setter
    def fun_args(self, fun_args):
        """We possibly want to update the arguments for the constraint function."""
        self._fun_args
        self.f = self._get_constraint_wrapper()

    def _get_constraint_wrapper(self):
        """Creates wrapper function for the constraint function that passes the arguments."""
        return lambda x: self._fun(x, *self._fun_args)


class LevelsetSubspaceSampler(LevelsetSampler):
    """
    This implementation essentially corresponds to the 'Hit-and-Run' sampling scheme (Smith, 1984)
    Short review paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=873287&tag=1 
    """

    def __init__(self, fun, fun_args=(), w=0.1):
        """
        Parameters
        ----------
        fun : callable
            The constraint function defining the levelset, F = { x | f(x) >= 0.0 }.

                ``fun(x, *fun_args) -> ndarray``

            where x is an 2-D array with shape (n, dim) and `fun_args` is a tuple of the fixed
            parameters need to completely specify the function.
        fun_args : tuple, optional
            Extra arguments passed to the constraint function and its derivatives.
        
        w : float
            Step size for increasing 1d subspace.

        Notes
        -----
        The step size `w` has large influence on performance. Rather choose too large, rejejection
        sampling in 1D is very efficient.
        """
        super().__init__(fun, fun_args)
        self.w = w 
        self.max_rejection_steps = 50
        self.max_doublings = 10

    def __call__(self, n, x0):
        """
        Parameters
        ----------
        n : int
            Number of samples
        x0 : ndarray, shape(m, dim) or (list of) shape(dim, )
            Array of starting points. All need to fulfill f(x0) >= 0

        Returns
        -------
        x_samples : ndarray, shape (n, dim)
            Array containing the desired number of samples.
        """
        x0 = np.array(x0) if isinstance(x0, list) else x0
        x0 = np.atleast_2d(x0).copy()
        assert (self.f(x0) > 0).all()

        m, dim = x0.shape
        step_size = self.w * np.ones((m, 1))

        x_samples = []
        N = n // m + 1

        def _find_bounds(init_stepsize):
            """From the initial samples, walk in the direction of `directions` until we
            found outer `bounds` for the constraint. These are used for rejection sampling."""
            bounds = init_stepsize.copy()
            within_constraints = self.f(x0 + bounds * directions) > 0
            for _ in range(self.max_doublings):
                if not within_constraints.any():
                    break
                bounds[within_constraints] *= 2.
                within_constraints = self.f(x0 + bounds * directions) > 0
            return bounds
        
        for _ in range(N):
            # Random direction on the unit sphere
            directions = np.random.randn(m, dim)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)

            # Find bounds for rejection sampling
            upper = _find_bounds(step_size * np.ones((m, 1)))
            lower = _find_bounds(-step_size * np.ones((m, 1)))
                
            # Rejection sampling on subspace
            rand_multiplier = np.random.uniform(lower, upper)
            xi = x0 + rand_multiplier * directions
            
            outside = self.f(xi) < 0  # Mask for points that are outside constraints
            for _ in range(self.max_rejection_steps):
                # Check if we are done
                if not outside.any():
                    break

                # Perform rejection sampling on random 1d subspaces
                rand_multiplier[outside] = np.random.uniform(lower[outside], upper[outside])
                xi[outside] = x0[outside] + rand_multiplier[outside] * directions[outside]

                # Adapt bounds for rejected points
                lower_indices = np.logical_and(outside, rand_multiplier.squeeze() < 0)
                lower[lower_indices] = rand_multiplier[lower_indices]
                upper_indices = np.logical_and(outside, rand_multiplier.squeeze() >= 0)
                upper[upper_indices] = rand_multiplier[upper_indices]

                # Evaluate constraint function at new locations
                outside[outside == True] = self.f(xi[outside]) < 0

            if outside.any():
                warning_text = "There are still points outside the levelset. " 
                warning_text += "Maybe increase number of rejection steps."
                raise RuntimeWarning(warning_text)

            x_samples.append(xi[~outside])
            x0 = xi.copy()
        return np.concatenate(x_samples)[:n]


class LevelsetRejectionSampler(LevelsetSampler):
    """Sample uniformly from a levelset given by f(x) >= 0.0 using rejection sampling."""

    def __init__(self, fun, fun_args, lb, ub):
        """
        Parameters
        ----------
        lb : ndarray, shape(dim(x), )
            Lower bound for uniform proposal distribution.
        ub : ndarray, shape(dim(x), )
            Upper bound for uniform proposal distribution.

        Notes
        -----
        This implementation is based on a box defined by `lb` and `ub`. It then uses rejection
        sampling, which can be very inefficient in high dimensions.
        """
        super().__init__(fun, fun_args)
        self.lb = lb
        self.ub = ub

    def __call__(self, n):
        """
        Parameters
        ----------
        n : int
            Number of samples

        Returns
        -------
        x_samples : ndarray, shape (n, dim)
            Array containing the desired number of samples.
        """
        x_samples = np.zeros((n, self.lb.shape[0]))
        i = 0
        while i < n:
            xi = np.random.uniform(self.lb, self.ub)
            if self.f(xi) >= 0.0:
                x_samples[i] = xi
                i += 1
        return x_samples
