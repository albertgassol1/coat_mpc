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
from typing import Optional, Tuple

import numpy as np
import rospy
from mpc_autotuner.dataclasses import Config, TunableWeights
from mpc_autotuner.utils import normalize


class WeightedMaximumLikelihood:

    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Weighted maximum likelihood policy search method to tune weights.
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Save tunable weights and config
        self.tunable_weights = tunable_weights
        self.config = config

        # Domain of parameters and initial theta
        self.domain = self.tunable_weights.get_domain()

        # Theta, laptime samples and reward
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))
        self.reward = np.empty((0, 1))

        # Current theta
        self.current_theta = normalize(self.tunable_weights.get_initial_weights(), self.domain)

        # Set initial mean and variance for the parameters policy
        self.policy_mean, self.policy_covariance = self.get_initial_policy_parameters()

        # Load data from previous runs
        if self.config.interface_config.load_prior_data:
            self.load_prior_data()

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
        self.laptime_samples = read_object('laptimes.np')
        self.policy_mean = read_object('policy_mean.np')
        self.policy_covariance = read_object('policy_covariance.np')
        self.reward = self.compute_rewards(-self.laptime_samples)[:, np.newaxis]

        batch_size = int(self.config.wml_config.N - self.theta.shape[0] % self.config.wml_config.N)
        new_sample_batch = self.sample_new_parameters(batch_size)
        self.theta = np.vstack((self.theta, new_sample_batch))
        self.current_theta = self.theta[self.laptime_samples.shape[0]][np.newaxis, :]

    def get_initial_policy_parameters(self, loc: Optional[float] = 0.5,
                                      scale: Optional[float] = 0.1) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Compute initial policy parameters (mean and covariance matrix)
        :param loc: mean
        :param scale: variance
        :return: initial policy parameters
        """
        mean = loc * np.ones(self.tunable_weights.num_parameters)
        covariance = np.eye(self.tunable_weights.num_parameters) * scale

        return mean, covariance

    def add_data_point(self, laptime_sample: float) -> None:
        """
        Perform Metropolis Hastings to get next set of parameters
        :param laptime_sample: laptime of current theta
        """

        assert self.current_theta.shape[1] == self.tunable_weights.num_parameters
        self.laptime_samples = np.vstack((self.laptime_samples, laptime_sample))

        # Print current policy parameters
        rospy.logwarn(f"\n"
                      f"Current policy parameters: \n"
                      f"\t- mean: {self.policy_mean} \n"
                      f"\t- covariance: {self.policy_covariance}")

        if self.theta.shape[0] == 0:
            self.theta = np.vstack((self.theta, self.current_theta))

        if self.theta.shape[0] == 1 or (self.laptime_samples.shape[0] %
                                        self.config.wml_config.N) == 0:
            if self.theta.shape[0] > 1:
                self.update_policy()
                batch_size = self.config.wml_config.N
            else:
                batch_size = self.config.wml_config.N - 1

            new_sample_batch = self.sample_new_parameters(batch_size)
            self.theta = np.vstack((self.theta, new_sample_batch))

        self.current_theta = self.theta[self.laptime_samples.shape[0]][np.newaxis, :]

    def update_policy(self) -> None:
        """
        Update policy parameters using closed form solution
        """
        # Compute rewards
        rewards = self.compute_rewards(-self.laptime_samples[-self.config.wml_config.N:, :])
        self.reward = np.vstack((self.reward, rewards[:, np.newaxis]))

        # Get last parameters associated with the rewards
        last_parameters_batch = self.theta[-self.config.wml_config.N:, :]

        # Compute normalizing factor
        rewards_sum = rewards.sum()
        normalizing_factor = (rewards_sum ** 2 - (rewards ** 2).sum()) / rewards_sum

        # Update mean, covariance
        self.policy_mean = (rewards[:, np.newaxis] * last_parameters_batch).sum(axis=0) / rewards_sum
        new_policy_covariance = np.zeros(self.policy_covariance.shape)
        for i in range(self.config.wml_config.N):
            new_policy_covariance += \
                rewards[i] * np.matmul((last_parameters_batch[i] - self.policy_mean)[:, np.newaxis],
                                       (last_parameters_batch[i] - self.policy_mean)[np.newaxis, :])

        self.policy_covariance = new_policy_covariance / normalizing_factor

        rospy.logwarn(f"New policy mean: {self.policy_mean}")
        rospy.logwarn(f"New policy covariance: {self.policy_covariance}")

    def compute_rewards(self, performance: np.ndarray) -> np.ndarray:
        """
        Compute rewards for given performance metric
        :param performance: performance metric vector
        :return: reward vector
        """
        reward: np.ndarray = np.exp(self.config.wml_config.beta * performance).squeeze()
        return reward

    def sample_new_parameters(self, batch_size: int) -> np.ndarray:
        """
        Sample N new parameters given the current policy
        :param batch_size: number of samples
        :return: new batch of parameters
        """
        new_parameter_batch = np.zeros((batch_size,
                                       self.tunable_weights.num_parameters))
        for i in range(batch_size):
            new_parameter_batch[i, :] = \
                self.bounded_sample_multivariate_normal(self.policy_mean, self.policy_covariance)
        return new_parameter_batch

    def bounded_sample_multivariate_normal(self, mean: np.ndarray,
                                           cov: np.ndarray) -> np.ndarray:
        """
        Sample from bounded MVN
        :param mean: mean vector
        :param cov: covariance matrix
        :return: sample vector
        """
        new_sample = np.random.multivariate_normal(mean=mean, cov=cov, size=1)
        if np.any(new_sample < 0) or np.any(new_sample > 1):
            return self.bounded_sample_multivariate_normal(mean, cov)

        return new_sample

    def get_final_solution(self) -> Tuple[np.ndarray, float]:
        """normalized_laptime_samples
        Computes argmin of sampled lap times
        :return: (weights that minimize the laptime, minimum laptime)
        """
        theta_opt_ind = np.argmin(self.laptime_samples)
        return self.theta[theta_opt_ind], self.laptime_samples[theta_opt_ind]
