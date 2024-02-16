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

import numpy as np
import rospy
from coat_mpc.methods.base_optimizer import BaseOptimizer
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc.utils.io import read_object


class WeightedMaximumLikelihood(BaseOptimizer):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Weighted maximum likelihood policy search method to tune weights.
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Call parent constructor
        super().__init__(config, tunable_weights)

        # Theta, laptime samples and reward
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))
        self.reward = np.empty((0, 1))

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
        self.laptime_samples = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "laptimes.np")
        )
        self.policy_mean = read_object(
            os.path.join(self.config.interface_config.prior_data_path, "policy_mean.np")
        )
        self.policy_covariance = read_object(
            os.path.join(
                self.config.interface_config.prior_data_path, "policy_covariance.np"
            )
        )
        self.reward = self.compute_rewards(-self.laptime_samples)[:, np.newaxis]

        batch_size = int(
            self.config.wml_config.N - self.theta.shape[0] % self.config.wml_config.N
        )
        new_sample_batch = self.sample_new_parameters(batch_size)
        self.theta = np.vstack((self.theta, new_sample_batch))
        self.current_theta = self.theta[self.laptime_samples.shape[0]][np.newaxis, :]

    def get_initial_policy_parameters(
        self, loc: Optional[float] = 0.5, scale: Optional[float] = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        rospy.logwarn(
            f"\n"
            f"Current policy parameters: \n"
            f"\t- mean: {self.policy_mean} \n"
            f"\t- covariance: {self.policy_covariance}"
        )

        if self.theta.shape[0] == 0:
            self.theta = np.vstack((self.theta, self.current_theta))

        if (
            self.theta.shape[0] == 1
            or (self.laptime_samples.shape[0] % self.config.wml_config.N) == 0
        ):
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
        rewards = self.compute_rewards(
            -self.laptime_samples[-self.config.wml_config.N :, :]
        )
        self.reward = np.vstack((self.reward, rewards[:, np.newaxis]))

        # Get last parameters associated with the rewards
        last_parameters_batch = self.theta[-self.config.wml_config.N :, :]

        # Compute normalizing factor
        rewards_sum = rewards.sum()
        normalizing_factor = (rewards_sum**2 - (rewards**2).sum()) / rewards_sum

        # Update mean, covariance
        self.policy_mean = (rewards[:, np.newaxis] * last_parameters_batch).sum(
            axis=0
        ) / rewards_sum
        new_policy_covariance = np.zeros(self.policy_covariance.shape)
        for i in range(self.config.wml_config.N):
            new_policy_covariance += rewards[i] * np.matmul(
                (last_parameters_batch[i] - self.policy_mean)[:, np.newaxis],
                (last_parameters_batch[i] - self.policy_mean)[np.newaxis, :],
            )

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
        new_parameter_batch = np.zeros(
            (batch_size, self.tunable_weights.num_parameters)
        )
        for i in range(batch_size):
            new_parameter_batch[i, :] = self.bounded_sample_multivariate_normal(
                self.policy_mean, self.policy_covariance
            )
        return new_parameter_batch

    def bounded_sample_multivariate_normal(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
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
