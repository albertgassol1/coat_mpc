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

# Libraries
import os
import pickle
from typing import Optional, Tuple, Union

import GPy
import numpy as np
import rospy
import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
from mpc_autotuner.acquisition_functions import AcquisitionFunction
# Package classes
from mpc_autotuner.dataclasses import Config, TunableWeights
from mpc_autotuner.utils import denormalize, normalize
from safeopt.gp_opt import SafeOpt
from safeopt.utilities import linearly_spaced_combinations


class BayesianOptimizer:

    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Bayesian optimizer. Computes new sets of parameters to try to minimize the laptime
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Save tunable weights and config
        self.tunable_weights = tunable_weights
        self.config = config

        # GP model for lap_time and constraints functions
        self.surrogate_model: SingleTaskGP = None
        # Safe opt object
        self.safe_opt: Optional[SafeOpt] = None

        # Acquisition function
        self.acquisition_function_object = AcquisitionFunction(
            self.config.optimization_config.acquisition_function,
            self.config.optimization_config.beta,
            self.config.optimization_config.first_lap_multiplier)

        # Domain of parameters and initial theta
        self.domain = self.tunable_weights.get_domain()

        # We save the sampled theta and laptime
        self.current_theta = normalize(self.tunable_weights.get_initial_weights(), self.domain)
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
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_prior_data(self) -> None:
        """
        Option to load data from previous runs of the optimization.
        The data is added to the surrogate models
        """

        def read_object(file_path: str) -> np.ndarray:
            """
            Read numpy array from file path
            :param file_path: path to numpy array pickable object
            """
            with open(os.path.join(self.config.interface_config.prior_data_path, file_path),
                      'rb') as data_file:
                array: np.ndarray = pickle.load(data_file)
                return array

        # Read pickable objects
        # Check if domains are the same
        domain = read_object('domain.np')
        if not np.array_equal(domain, self.domain):
            # Raise error if domains are different
            raise ValueError('Loaded domain is different from computed domain')

        self.theta = read_object('theta.np')
        self.normalized_laptime_samples = read_object('normalized_laptimes.np')
        self.laptime_samples = read_object('laptimes.np')

        # Fit models. SAFEOPT uses GPy, so we need to check which method we are using
        if self.config.optimization_config.acquisition_function == "SAFEOPT":
            # Define kernel
            kernel = GPy.kern.Matern52(
                input_dim=self.domain.shape[0],
                ARD=False,
                variance=self.config.optimization_config.kernel_variance,
                lengthscale=self.config.optimization_config.kernel_lengthscale)

            # Define prior mean function: constant
            mf = GPy.core.Mapping(self.tunable_weights.num_parameters, 1)
            if self.config.optimization_config.prior_mean:
                mf.f = lambda x: - (self.config.optimization_config.first_lap_multiplier - 0.02) * \
                       self.laptime_samples[0]
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
                noise_var=self.config.optimization_config.gp_variance**2)
        elif self.config.optimization_config.acquisition_function == "UCB" or \
                self.config.optimization_config.acquisition_function == "EIC":
            # For UCB and EIC
            if self.theta.shape[0] < self.config.optimization_config.standardization_batch:
                standardized_samples = torch.from_numpy(self.normalized_laptime_samples)
            elif (self.theta.shape[0] % self.config.optimization_config.standardization_batch) == 0:
                standardized_samples, self.mean, self.stddev = \
                    self.standardize(torch.from_numpy(self.normalized_laptime_samples))
            else:
                standardized_samples, _, _ = \
                    self.standardize(torch.from_numpy(self.normalized_laptime_samples),
                                     self.mean, self.stddev)

            if self.config.optimization_config.acquisition_function == "EIC":
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta),
                    torch.hstack((-torch.from_numpy(self.normalized_laptime_samples),
                                  -torch.from_numpy(self.normalized_laptime_samples))))
            else:
                custom_mean = ConstantMean()
                custom_mean.initialize(constant=-float((self.config.optimization_config.first_lap_multiplier - 0.02) *
                                                       self.normalized_laptime_samples[0]))
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta), -standardized_samples)

            if self.config.optimization_config.constant_lengthscale:
                self.surrogate_model.covar_module.base_kernel.lengthscale = \
                    torch.ones((1, self.domain.shape[0]), dtype=torch.float64) * \
                    self.config.optimization_config.kernel_lengthscale
            else:
                mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood,
                                                 self.surrogate_model)
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
        assert self.current_theta.shape[1] == self.tunable_weights.num_parameters

        # Save new sample into vector of theta
        self.theta = np.vstack((self.theta, self.current_theta))

        # Add lap time sample to database
        self.laptime_samples = np.vstack((self.laptime_samples, laptime_sample))
        self.normalized_laptime_samples = np.vstack(
            (self.normalized_laptime_samples, laptime_sample / self.min_laptime))

        # If we use SAFEOPT as optimization method
        if self.config.optimization_config.acquisition_function == "SAFEOPT"\
                and self.safe_opt is None:
            # Define kernel
            kernel = GPy.kern.Matern52(
                input_dim=self.domain.shape[0],
                ARD=False,
                variance=self.config.optimization_config.kernel_variance,
                lengthscale=self.config.optimization_config.kernel_lengthscale)
            # Define prior mean function: constant
            mf = GPy.core.Mapping(self.tunable_weights.num_parameters, 1)
            if self.config.optimization_config.prior_mean:
                mf.f = lambda x: - (self.config.optimization_config.first_lap_multiplier - 0.02) * \
                                 self.laptime_samples[0]
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
                noise_var=self.config.optimization_config.gp_variance**2)

        elif self.config.optimization_config.acquisition_function == "UCB" or \
                self.config.optimization_config.acquisition_function == "EIC":
            # For UCB and EIC
            if self.theta.shape[0] < self.config.optimization_config.standardization_batch:
                standardized_samples = torch.from_numpy(self.normalized_laptime_samples)
            elif (self.theta.shape[0] % self.config.optimization_config.standardization_batch) == 0:
                standardized_samples, self.mean, self.stddev = \
                    self.standardize(torch.from_numpy(self.normalized_laptime_samples))
            else:
                standardized_samples, _, _ = \
                    self.standardize(torch.from_numpy(self.normalized_laptime_samples),
                                     self.mean, self.stddev)

            if self.config.optimization_config.acquisition_function == "EIC":
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta),
                    torch.hstack((-torch.from_numpy(self.normalized_laptime_samples),
                                  -torch.from_numpy(self.normalized_laptime_samples))))
            else:
                custom_mean = ConstantMean()
                custom_mean.initialize(constant=-float((self.config.optimization_config.first_lap_multiplier - 0.02) *
                                                self.normalized_laptime_samples[0]))
                self.surrogate_model = SingleTaskGP(
                    torch.from_numpy(self.theta), -standardized_samples)

            if self.config.optimization_config.constant_lengthscale:
                self.surrogate_model.covar_module.base_kernel.lengthscale = \
                    torch.ones((1, self.domain.shape[0]), dtype=torch.float64) * \
                    self.config.optimization_config.kernel_lengthscale

            else:
                mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood,
                                                 self.surrogate_model)
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
        if self.theta.shape[0] == 1 and \
                self.config.optimization_config.acquisition_function == "SAFEOPT":
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
                bounds = list(map(tuple, np.hstack((np.zeros((self.domain.shape[0], 1)),
                                                    np.ones((self.domain.shape[0], 1))))))
                parameter_set = linearly_spaced_combinations(bounds,
                                                                self.config.optimization_config.
                                                                grid_size)
                # Perform greedy optimization
                self.safe_opt = SafeOpt(
                    self.surrogate_model,
                    parameter_set,
                    fmin=self.config.optimization_config.first_lap_multiplier * initial_laptime,
                    beta=self.config.optimization_config.beta,
                    lipschitz=self.config.optimization_config.lipschitz_constant,
                    minimum_variance=self.config.optimization_config.minimum_variance)
            else:
                self.safe_opt.add_new_data_point(self.theta[-1, :],
                                                 -self.laptime_samples[-1])
            candidate, stop = \
                self.safe_opt.optimize(ucb=self.config.optimization_config.use_ucb)

            candidate = candidate.reshape(1, self.domain.shape[0])

        else:
            # Get acquisition function
            acquisition_function = \
                self.acquisition_function_object.acquisition_function(self.surrogate_model)

            # Get bounds (between 0 and 1 as we are using normalized values)
            bounds = torch.stack(
                [torch.zeros(self.domain.shape[0]),
                 torch.ones(self.domain.shape[0])])

            # Get next recommendation
            candidate, _ = \
                optimize_acqf(acquisition_function,
                              bounds=bounds,
                              q=1,
                              num_restarts=self.config.optimization_config.number_bo_restarts,
                              raw_samples=self.config.optimization_config.raw_samples)
            candidate = candidate.numpy()
        return candidate, stop

    def get_final_solution(self) -> Tuple[np.ndarray, float]:
        """
        Computes argmin of sampled lap times
        :return: (weights that minimize the laptime, minimum laptime)
        """
        theta_opt_ind = np.argmin(self.normalized_laptime_samples)
        return self.theta[theta_opt_ind], self.laptime_samples[theta_opt_ind]

    @staticmethod
    def standardize(samples: torch.Tensor,
                    mean: Optional[torch.Tensor] = None,
                    stddev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,
                                                                    torch.Tensor,
                                                                    torch.Tensor]:
        """
        Standardizes (zero mean, unit variance) a tensor by dim=-2.
        If the tensor is single-dimensional, simply standardizes the tensor.
        If for some batch index all elements are equal (or if there is only a single
        data point), this function will return 0 for that batch index
        :param samples: A `batch_shape x n x m`-dim tensor
        :param mean: mean to standardize with
        :param stddev: standard deviation to standardize with
        :return: The standardized `Y`, mean and stddev
        """
        if mean is not None and stddev is not None:
            return samples - mean / stddev, mean, stddev

        stddim = -1 if samples.dim() < 2 else -2
        samples_std = samples.std(dim=stddim, keepdim=True)
        samples_std = samples_std.where(samples_std >= 1e-9, torch.full_like(samples_std, 1.0))
        samples_mean = samples.mean(dim=stddim, keepdim=True)
        return (samples - samples_mean) / samples_std, samples_mean, samples_std
