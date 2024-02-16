#!/usr/bin/env python3
 #
 # MIT License
 #
 # Copyright (c) 2023 Authors:
 #   - Albert Gassol Puigjaner <agassol@ethz.ch>
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.

import os

import coat_mpc.methods.crbo.sampling as sampling
import coat_mpc.methods.crbo.util as util
import GPy
import numpy as np
import scipy.optimize as sciopt
from coat_mpc.methods.crbo.constraints import ConfidenceRegionConstraint
from coat_mpc.methods.crbo.lower_confidence_bound import LowerConfidenceBound
from coat_mpc.methods.crbo.thompson_sampling import ThompsonSampling

# Package classes
from coat_mpc.methods.base_optimizer import BaseOptimizer
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc.utils.io import read_object


class BayesianOptimization(BaseOptimizer):
    def __init__(
        self,
        config: Config,
        tunable_weights: TunableWeights,
        acq_type="lcb",
        optimizer_strategy="sample_local",
        normalizer_type="optimistic",
        kernel_variance=1.0,
    ):
        self.__name__ = "bo-base"

        # Call parent constructor
        super().__init__(config, tunable_weights)

        # Convert bounds to scipy object
        self.bounds = sciopt.Bounds(lb=self.domain[:, 0], ub=self.domain[:, 1])

        # MPC weights as theta and laptimes
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))

        self.n_init = 2  # Number of initial design points
        self.normalizer_type = normalizer_type
        self.optimizer_strategy = optimizer_strategy

        # Sample points are used for optimization of the acquisition function
        if self.optimizer_strategy == "sample_only":
            self._n_max_samples = min(
                200 * self.tunable_weights.num_parameters, 2000
            )  # Maximum number of samples
        elif self.optimizer_strategy == "sample_local":
            self._n_max_samples = 1000
        self._sample_points = None

        # Store the time it takes to optimize acquisition function and update model
        self._iter_duration = []

        # Choose acquisition function
        if acq_type == "lcb":
            self.acq_fun = LowerConfidenceBound(
                beta=self.config.optimization_config.beta
            )
        elif acq_type == "ts":
            self.acq_fun = ThompsonSampling()
            if optimizer_strategy == "sample_local":
                raise ValueError("TS on grid not compatible with local optimizer.")
        else:
            raise ValueError()

        self.kernel_variance = self.config.optimization_config.kernel_lengthscale
        self.model = None
        self.norm_mu = None
        self.norm_std = None

        # Load data from previous runs
        if self.config.interface_config.load_prior_data:
            self.load_prior_data()

    def load_prior_data(self) -> None:
        """
        Option to load data from previous runs of the optimization.
        The data is added to the surrogate models
        """

        # Read pickable objects
        # Check if domains are the same
        domain = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "domain.np")
        )
        if not np.array_equal(domain, self.domain):
            # Raise error if domains are different
            raise ValueError("Loaded domain is different from computed domain")

        theta: np.ndarray = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "theta.np")
        )
        laptime_samples: np.ndarray = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "laptimes.np")
        )
        for i in range(1, theta.shape[0]):
            self.current_theta[np.newaxis, :] = theta[i]
            self.add_data_point(laptime_samples[i])

    def add_data_point(self, laptime_sample: float):
        super().add_data_point(laptime_sample)

        if self.theta.shape[0] == 0:
            self.current_theta = self._initial_design()[-1, :]
            return
        if self.model is None:
            normalized_laptimes, self.norm_mu, self.norm_std = util.normalize_output(
                self.laptime_samples, mode=self.normalizer_type
            )
            # Build model (noise and signal variance are scaled)
            kernel = GPy.kern.Matern52(
                input_dim=self.tunable_weights.num_parameters,
                ARD=False,
                variance=self.kernel_variance / self.norm_std**2,
                lengthscale=self.config.optimization_config.kernel_lengthscale,
            )
            self.model = GPy.models.GPRegression(
                self.theta,
                self.laptime_samples,
                kernel=kernel,
                noise_var=self.config.optimization_config.gp_variance**2
                / self.norm_std**2,
            )

        # Suggest new point by optimizing acquisition function
        self._update_model(
            self.current_theta, laptime_sample
        )  # At this point, y_new is NOT normalized
        self.current_theta = self._next_point()

        return False

    def _create_results_dictionary(self):
        res_dict = {
            "x_eval": util.denormalize_input(self.model.X, self.bounds).tolist(),
            "y_eval": util.denormalize_output(
                self.model.Y, self.norm_mu, self.norm_std
            ).tolist(),
            "opt_type": self.__name__,
            "acq_type": self.acq_fun.__name__,
            "objective": "lap_time",
            "gp_model": self.model.to_dict(),
            "normalizer_type": self.normalizer_type,
            "bounds": {"lb": self.bounds.lb.tolist(), "ub": self.bounds.ub.tolist()},
            "iter_duration": self._iter_duration,
        }
        return res_dict

    def _update_model(self, x_new, y_new):
        x_new = np.atleast_2d(x_new)
        y_new = np.atleast_2d(y_new)
        assert x_new.shape[0] == y_new.shape[0]
        X_norm = np.vstack((self.model.X, x_new))
        Y = util.denormalize_output(self.model.Y, self.norm_mu, self.norm_std)
        Y = np.vstack((Y, y_new))
        Y_norm, self.norm_mu, self.norm_std = util.normalize_output(
            Y, mode=self.normalizer_type
        )
        self.model.set_XY(X_norm, Y_norm)
        if not self.config.optimization_config.constant_lengthscale:
            self.model.optimize()
        else:
            signal_var = self.kernel_variance / self.norm_std**2
            self.model.kern.variance = signal_var

            noise_var = (
                self.config.optimization_config.gp_variance**2 / self.norm_std**2
            )
            self.model.likelihood.variance[:] = noise_var

        ell = np.clip(self.model.kern.lengthscale.values, 0.05, 0.5)
        self.model.kern.lengthscale[:] = ell

    def _next_point(self):
        self.acq_fun.setup(self.model)
        if self._sample_points is None:
            self._sample_points = self._space_filling_design(n=self._n_max_samples)
        else:
            self._sample_points = self._resample()

        fs = self.acq_fun(self._sample_points, self.model)
        x_opt = self._sample_points[np.argmin(fs)]

        # Optionally, start local, gradient-basd optimizer from best point
        if self.optimizer_strategy == "sample_local":
            x_opt = self._local_opt(x_opt)[0]

        return x_opt

    def _initial_design(self):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _space_filling_design(self, n):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _resample(self):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )

    def _local_opt(self, x0):
        raise NotImplementedError(
            "This is an abstract base class. Therefore, this method is not implemented."
        )


class ConfidenceRegionBayesianOptimization(BayesianOptimization):
    """Confidence Region BO as proposed in our paper."""

    def __init__(self, config: Config, tunable_weights: TunableWeights, gamma=0.6):
        self.conf_region = ConfidenceRegionConstraint(gamma)
        self.__name__ = "crbo"

        super().__init__(config, tunable_weights)

    def _create_results_dictionary(self):
        res_dict = super()._create_results_dictionary()
        res_dict["gamma"] = self.conf_region.gamma
        return res_dict

    def _initial_design(self):
        """Given the initial data point, sample n_init - 1 more points around it. Specifically,
        if only one point is given, the confidence region is given by a Ball. For the SE kernel
        we can actually compute the radius of this ball analytically and we then sample the
        initial points on the surface of said ball."""
        if self.current_theta is None:
            raise ValueError("CRBO requires exactly one initial data point.")
        if (self.current_theta is not None) and (self.current_theta.shape[0] > 1):
            raise ValueError("CRBO requires exactly one initial data point.")
        if self.current_theta is None:
            print(
                "You did not specify how many initial data points should be used. It is highly"
            )
            print("recommended to use more than only one.")
            x0 = util.denormalize_input(self.current_theta, self.bounds)
        else:
            print(f"You provided 1 initial data point but required {self.n_init}.")
            print(f"Generating {self.n_init - 1} more points.")
            # Assuming that we have a RBF kernel and an initial lengthscale `ell`, compute
            # radius of the confidence region (for one data points it's just a sphere).
            ell = 0.2
            r0 = ell * np.sqrt(-np.log(1 - self.conf_region.gamma**2))
            x0 = np.random.randn(self.n_init - 1, self.tunable_weights.num_parameters)
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            x0 *= r0

            x_init_normalized = self.current_theta
            x0 += x_init_normalized
            x0 = np.vstack((x_init_normalized, x0))

        x0 = np.clip(x0, 0.0, 1.0)
        return x0

    def _space_filling_design(self, n):
        """Use LevelsetSubspace (or Hit-and-Run) sampler to uniformly fill the confidence region
        with sample points. For improved coverage, we start multiple sample chains starting from
        the data points of the current GP model. Since the sampler makes use of parallel function
        evaluations, it is recommended to have multiple chains even if only few data ponints exist.
        We'll therefore randomly sample multiple data points with replacement.
        Note that the confidence region actually might be empty. This happens sometimes when 1) the
        the confidence region is chosen rather small (like 0.3) or 2) in the beginning of the
        optimization, when the noise variance is quite large with respect to to the signal variance.
        To deal with this issue, we inflate the signal variance until the confidence region
        constraint is fulfilled for all data points of the GP."""
        n_increases, max_increases = 0, 50
        while self.conf_region.isempty(self.model) and n_increases < max_increases:
            # Important to fix the model when setting the variance by hand
            self.model.update_model(False)
            self.model.kern.variance.values[:] *= 1.2
            self.model.update_model(True)
            n_increases += 1
            if n_increases == max_increases:
                print(self.model)
                raise RuntimeError(
                    "Confidence region still empty after the maximum amount of increases "
                    "of the confidence region."
                )

        sampler = sampling.LevelsetSubspaceSampler(
            fun=self.conf_region, fun_args=(self.model,), w=0.1
        )
        x0 = self.model.X[
            np.random.choice(self.model.X.shape[0], size=(256,), replace=True)
        ]
        x_samples = sampler(n, x0)
        unit_bounds = sciopt.Bounds(
            lb=np.zeros((self.tunable_weights.num_parameters,)),
            ub=np.ones((self.tunable_weights.num_parameters,)),
        )
        x_samples = util.project_into_bounds(x_samples, unit_bounds)
        return x_samples

    def _resample(self):
        """Due to the (possibly) changing design space, the percentage of re-sampled points
        for CRBO is larger in comparison to BoxBO. Further, we need to be careful that old
        sample points are still within the confidence region which can happen when use hyper
        parameters are inferred and the lengthscale decreases."""

        # When re-sampling, we need to check for samples that are now out of the conf region
        ind_out = np.where(self.conf_region(self._sample_points, self.model) <= 0)[0]
        n_new_samples = max(ind_out.shape[0], self._n_max_samples // 2)

        # Re-sample points that are now out of bounds and some new ones
        xs = self._space_filling_design(n=n_new_samples)
        ind_rand = np.random.choice(
            self._n_max_samples, (n_new_samples - ind_out.shape[0],), replace=False
        )
        ind_rand = np.concatenate((ind_out, ind_rand)) if ind_rand.size else ind_out
        self._sample_points[ind_rand] = xs
        return self._sample_points

    def _local_opt(self, x0):
        """Around the initial guess x0, find local optimum of the acquisition function given the
        confidence region consraint and the outer unit box consraint."""
        constraints_scipy = {
            "type": "ineq",
            "fun": self.conf_region,
            "jac": self.conf_region.jac,
            "args": (self.model,),
        }
        options = {"maxiter": 100}
        tol = 1e-4
        acq_args = (self.model, True)

        # Put small box in order to force the optimizer to stay close to initial guess
        lb = np.clip(x0 - 0.5 * self.model.kern.lengthscale, 0.0, +np.inf)
        ub = np.clip(x0 + 0.5 * self.model.kern.lengthscale, -np.inf, 1.0)
        small_bounds = sciopt.Bounds(lb, ub)

        res = sciopt.minimize(
            self.acq_fun,
            x0,
            args=acq_args,
            bounds=small_bounds,
            jac=acq_args[1],
            constraints=constraints_scipy,
            tol=tol,
            options=options,
        )

        f0 = self.acq_fun(x0, self.model)

        if res["fun"] < f0 and res["success"]:
            return res["x"], res["fun"]
        if res["fun"] > f0 and res["success"]:
            return x0, f0
        elif res["success"] and res["nit"] == 1:
            return res["x"], res["fun"]
        elif res["status"] == 9 and res["fun"] < f0:
            return res["x"], res["fun"]
        else:
            return x0, f0
