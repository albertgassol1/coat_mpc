#!/usr/bin/env python3

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
import os
import numpy as np
import rospy

# Package classes
from coat_mpc.methods.base_optimizer import BaseOptimizer
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc.utils.utils import denormalize, normalize
from coat_mpc.utils.io import read_object


class MetropolisHastings(BaseOptimizer):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Metropolis Hastings method. Computes new sets of parameters to try to minimize the laptime
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Call parent constructor
        super().__init__(config, tunable_weights)

        # MPC weights as theta and laptimes
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))

        # MH theta at time t and laptime at time t
        self.theta_t = normalize(
            self.tunable_weights.get_initial_weights(), self.domain
        )
        self.cost_t = 0.0
        self.accepted_theta = np.empty((0, self.tunable_weights.num_parameters))
        self.accepted_laptimes = np.empty((0, 1))

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
        self.accepted_theta = read_object(
            os.path.join(
                self.config.interface_config.prior_data_path, "accepted_theta.np"
            )
        )
        self.accepted_laptimes = read_object(
            os.path.join(
                self.config.interface_config.prior_data_path, "accepted_laptimes.np"
            )
        )
        self.cost_t = self.compute_cost(self.accepted_laptimes[-1])
        self.theta_t = self.accepted_theta[np.newaxis, -1]
        self.current_theta = self.sample_from_transition_model()

    def add_data_point(self, laptime_sample: float) -> None:
        """
        Perform Metropolis Hastings to get next set of parameters
        :param laptime_sample: laptime of current theta
        """
        super().add_data_point(laptime_sample)

        # If first iteration, just save the current laptime and
        # get next parameter sample with transition model
        if self.theta.shape[0] == 1:
            self.current_theta = self.sample_from_transition_model()
            self.accepted_theta = np.vstack((self.accepted_theta, self.current_theta))
            self.accepted_laptimes = np.vstack((self.accepted_laptimes, laptime_sample))
            self.cost_t = self.compute_cost(laptime_sample)
            return

        # Compute cost and see if we accept current sample
        laptime_cost = self.compute_cost(laptime_sample)

        # If we accept the sample, update current MH state
        if self.accept_sample(laptime_cost):
            self.theta_t = self.current_theta.copy()
            self.cost_t = laptime_cost
            self.accepted_laptimes = np.vstack((self.accepted_laptimes, laptime_sample))
            self.accepted_theta = np.vstack((self.accepted_theta, self.current_theta))
            rospy.logwarn(
                f"Updated MH state. New theta_t={denormalize(self.theta_t, self.domain)},"
                f" new laptime_t={laptime_sample}.\n"
                f" Acceptance rate="
                f"{self.accepted_theta.shape[0] / self.theta.shape[0] * 100}%"
            )
        else:
            rospy.logwarn(
                f"Rejected theta={denormalize(self.current_theta, self.domain)} "
                f"with laptime={laptime_sample}"
            )

        # Sample from updated transition model
        self.current_theta = self.sample_from_transition_model()

    def sample_from_transition_model(self) -> np.ndarray:
        """
        Use transition model (Gaussian) to get new parameter value
        :return: new parameters
        """
        next_theta = np.zeros(self.current_theta.shape, dtype=np.float64)

        for i in range(self.tunable_weights.num_parameters):
            next_theta[0, i] = self.bounded_gaussian(
                self.theta_t[0, i], self.config.mh_config.sigma, np.array([0.0, 1.0])
            )
        return next_theta

    def bounded_gaussian(self, mean: float, sigma: float, bounds: np.ndarray) -> float:
        """
        Sample from a gaussian with bounds
        :param mean: mean of gaussian
        :param sigma: standard deviation of gaussian
        :param bounds:bounds of Gaussian
        :return: sample from gaussian
        """
        new_parameter = np.random.normal(mean, sigma)

        # Check bounds
        if bounds[0] <= new_parameter <= bounds[1]:
            return new_parameter
        return self.bounded_gaussian(mean, sigma, bounds)

    def accept_sample(self, cost: float) -> bool:
        """
        Return true if we accept this sample
        :param cost: laptime cost to compare
        :return: acceptance bool
        """
        if cost < self.cost_t:
            return True
        else:
            accept = np.random.uniform(0, 1)
            return accept < (self.cost_t / cost)

    @staticmethod
    def compute_cost(laptime: float) -> float:
        """
        Compute exponential score
        :param laptime: laptime value
        :return: cost of laptime
        """
        res: float = np.exp(-2 * np.sqrt(laptime))
        return res
