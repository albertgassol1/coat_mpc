#!/usr/bin/env python3
#
# BSD 2-Clause License
#
# Copyright (c) 2024
#   - Albert Gassol Puigjaner <agassol@ethz.ch>

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Libraries
import os
from typing import Optional, Tuple

import GPy
import numpy as np
import rospy
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from coat_mpc.methods.acquisition_functions import AcquisitionFunction

# Package classes
from coat_mpc.methods.base_optimizer import BaseOptimizer
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc.utils.io import read_object
from coat_mpc.utils.utils import standardize
from coat_mpc.methods.safeopt.gp_opt import SafeOpt
from coat_mpc.methods.safeopt.utilities import linearly_spaced_combinations


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Bayesian optimizer. Computes new sets of parameters to try to minimize the laptime
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Call parent constructor
        super().__init__(config, tunable_weights)

        # GP model for lap_time and constraints functions
        self.surrogate_model: SingleTaskGP = None
        # Safe opt object
        self.safe_opt: Optional[SafeOpt] = None

        # Acquisition function
        self.acquisition_function_object = AcquisitionFunction(
            self.config.optimization_config.acquisition_function,
            self.config.optimization_config.beta,
            self.config.optimization_config.first_lap_multiplier,
        )

        # We save the sampled theta and laptime
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))
        self.normalized_laptime_samples = np.zeros((0, 1))

        # Mean and stddev of samples
        self.mean = torch.empty((0, 1))
        self.stddev = torch.empty((0, 1))

        # Load data from previous runs
        if self.config.interface_config.load_prior_data:
            self.load_prior_data()

        # Optimal laptime
        self.min_laptime = self.config.interface_config.optimal_time

        # Use CUDA if available
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.theta = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "theta.np")
        )
        self.normalized_laptime_samples = read_object(
            os.path.join(
                self.config.interface_config.prior_data_path, "normalized_laptimes.np"
            )
        )
        self.laptime_samples = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "laptimes.np")
        )

        # Fit models. SAFEOPT uses GPy, so we need to check which method we are using
        if self.config.optimization_config.acquisition_function == "SAFEOPT":
            # Define kernel
            kernel = GPy.kern.Matern52(
                input_dim=self.domain.shape[0],
                ARD=False,
                variance=self.config.optimization_config.kernel_variance,
                lengthscale=self.config.optimization_config.kernel_lengthscale,
            )

            # Define prior mean function: constant
            mf = GPy.core.Mapping(self.tunable_weights.num_parameters, 1)
            if self.config.optimization_config.prior_mean:
                mf.f = (
                    lambda x: -(
                        self.config.optimization_config.first_lap_multiplier - 0.02
                    )
                    * self.laptime_samples[0]
                )
            else:
                mf.f = lambda x: 0.0
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0

            # Create surrogate model
            self.surrogate_model = GPy.models.GPRegression(
                self.theta,
                self.laptime_samples * -1,
                kernel,
                mean_function=mf,
                noise_var=self.config.optimization_config.gp_variance**2,
            )
        elif (
            self.config.optimization_config.acquisition_function == "UCB"
            or self.config.optimization_config.acquisition_function == "EIC"
        ):
            # For UCB and EIC
            if (
                self.theta.shape[0]
                < self.config.optimization_config.standardization_batch
            ):
                standardized_samples = torch.from_numpy(self.normalized_laptime_samples)
            elif (
                self.theta.shape[0]
                % self.config.optimization_config.standardization_batch
            ) == 0:
                standardized_samples, self.mean, self.stddev = standardize(
                    torch.from_numpy(self.normalized_laptime_samples)
                )
            else:
                standardized_samples, _, _ = standardize(
                    torch.from_numpy(self.normalized_laptime_samples),
                    self.mean,
                    self.stddev,
                )

            if self.config.optimization_config.acquisition_function == "EIC":
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta),
                    torch.hstack(
                        (
                            -torch.from_numpy(self.normalized_laptime_samples),
                            -torch.from_numpy(self.normalized_laptime_samples),
                        )
                    ),
                )
            else:
                custom_mean = ConstantMean()
                custom_mean.initialize(
                    constant=-float(
                        (self.config.optimization_config.first_lap_multiplier - 0.02)
                        * self.normalized_laptime_samples[0]
                    )
                )
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta), -standardized_samples
                )

            if self.config.optimization_config.constant_lengthscale:
                self.surrogate_model.covar_module.base_kernel.lengthscale = (
                    torch.ones((1, self.domain.shape[0]), dtype=torch.float64)
                    * self.config.optimization_config.kernel_lengthscale
                )
            else:
                mll = ExactMarginalLogLikelihood(
                    self.surrogate_model.likelihood, self.surrogate_model
                )
                fit_gpytorch_model(mll)

        # Get new set of tuning parameters
        self.current_theta, _ = self.next_recommendation()
        rospy.loginfo("LOAD DATA")
        rospy.loginfo(self.current_theta)
        rospy.loginfo(self.laptime_samples)

    def add_data_point(self, laptime_sample: float) -> bool:
        """
        Adds lap time of current theta to the database and updates the surrogate model.
        The optimization is triggered to get new parameters
        :param laptime_sample: lap time of current theta
        :return: stopping variable
        """
        super().add_data_point(laptime_sample)

        self.normalized_laptime_samples = np.vstack(
            (self.normalized_laptime_samples, laptime_sample / self.min_laptime)
        )

        # If we use SAFEOPT as optimization method
        if (
            self.config.optimization_config.acquisition_function == "SAFEOPT"
            and self.safe_opt is None
        ):
            # Define kernel
            kernel = GPy.kern.Matern52(
                input_dim=self.domain.shape[0],
                ARD=False,
                variance=self.config.optimization_config.kernel_variance,
                lengthscale=self.config.optimization_config.kernel_lengthscale,
            )
            # Define prior mean function: constant
            mf = GPy.core.Mapping(self.tunable_weights.num_parameters, 1)
            if self.config.optimization_config.prior_mean:
                mf.f = (
                    lambda x: -(
                        self.config.optimization_config.first_lap_multiplier - 0.02
                    )
                    * self.laptime_samples[0]
                )
            else:
                mf.f = lambda x: 0.0
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0

            # Create surrogate model
            self.surrogate_model = GPy.models.GPRegression(
                self.theta,
                self.laptime_samples * -1,
                kernel,
                mean_function=mf,
                noise_var=self.config.optimization_config.gp_variance**2,
            )

        elif (
            self.config.optimization_config.acquisition_function == "UCB"
            or self.config.optimization_config.acquisition_function == "EIC"
        ):
            # For UCB and EIC
            if (
                self.theta.shape[0]
                < self.config.optimization_config.standardization_batch
            ):
                standardized_samples = torch.from_numpy(self.normalized_laptime_samples)
            elif (
                self.theta.shape[0]
                % self.config.optimization_config.standardization_batch
            ) == 0:
                standardized_samples, self.mean, self.stddev = standardize(
                    torch.from_numpy(self.normalized_laptime_samples)
                )
            else:
                standardized_samples, _, _ = standardize(
                    torch.from_numpy(self.normalized_laptime_samples),
                    self.mean,
                    self.stddev,
                )

            if self.config.optimization_config.acquisition_function == "EIC":
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta),
                    torch.hstack(
                        (
                            -torch.from_numpy(self.normalized_laptime_samples),
                            -torch.from_numpy(self.normalized_laptime_samples),
                        )
                    ),
                )
            else:
                custom_mean = ConstantMean()
                custom_mean.initialize(
                    constant=-float(
                        (self.config.optimization_config.first_lap_multiplier - 0.02)
                        * self.normalized_laptime_samples[0]
                    )
                )
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta), -standardized_samples
                )

            if self.config.optimization_config.constant_lengthscale:
                self.surrogate_model.covar_module.base_kernel.lengthscale = (
                    torch.ones((1, self.domain.shape[0]), dtype=torch.float64)
                    * self.config.optimization_config.kernel_lengthscale
                )

            else:
                mll = ExactMarginalLogLikelihood(
                    self.surrogate_model.likelihood, self.surrogate_model
                )
                fit_gpytorch_model(mll)

        # Get new set of tuning parameters
        theta, stop = self.next_recommendation()
        self.current_theta = theta
        return stop

    def next_recommendation(self) -> Tuple[np.ndarray, bool]:
        """
        Get next set of parameters. For safeopt, the first iteration does not perform optimization.
        A close value to the current theta is returned to collect more data
        :return: new set of parameters, stopping variable
        """

        # For SAFEOPT case
        if (
            self.theta.shape[0] == 1
            and self.config.optimization_config.acquisition_function == "SAFEOPT"
        ):
            return self.current_theta + 0.00005, False
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self) -> Tuple[np.ndarray, bool]:
        """
        Optimize acquisition function and return the new set of weights
        :return: new set of weights, stopping variable
        """
        # Stopping variable
        stop = False
        # SAFEOPT case
        if self.config.optimization_config.acquisition_function == "SAFEOPT":
            # Perform safeopt optimization

            # try:
            if self.safe_opt is None:
                # Discretize parameter space
                initial_laptime = -self.laptime_samples[0]
                bounds = list(
                    map(
                        tuple,
                        np.hstack(
                            (
                                np.zeros((self.domain.shape[0], 1)),
                                np.ones((self.domain.shape[0], 1)),
                            )
                        ),
                    )
                )
                parameter_set = linearly_spaced_combinations(
                    bounds, self.config.optimization_config.grid_size
                )
                # Perform greedy optimization
                self.safe_opt = SafeOpt(
                    self.surrogate_model,
                    parameter_set,
                    fmin=self.config.optimization_config.first_lap_multiplier
                    * initial_laptime,
                    beta=self.config.optimization_config.beta,
                    lipschitz=self.config.optimization_config.lipschitz_constant,
                    minimum_variance=self.config.optimization_config.minimum_variance,
                )
            else:
                self.safe_opt.add_new_data_point(
                    self.theta[-1, :], -self.laptime_samples[-1]
                )
            candidate, stop = self.safe_opt.optimize(
                ucb=self.config.optimization_config.use_ucb
            )

            candidate = candidate.reshape(1, self.domain.shape[0])

        else:
            # Get acquisition function
            acquisition_function = (
                self.acquisition_function_object.acquisition_function(
                    self.surrogate_model
                )
            )

            # Get bounds (between 0 and 1 as we are using normalized values)
            bounds = torch.stack(
                [torch.zeros(self.domain.shape[0]), torch.ones(self.domain.shape[0])]
            )

            # Get next recommendation
            candidate, _ = optimize_acqf(
                acquisition_function,
                bounds=bounds,
                q=1,
                num_restarts=self.config.optimization_config.number_bo_restarts,
                raw_samples=self.config.optimization_config.raw_samples,
            )
            candidate = candidate.numpy()
        return candidate, stop
