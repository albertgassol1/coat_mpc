#!/usr/bin/env python3

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
import pickle
from typing import Tuple

import numpy as np
import rospy
# Package classes
from mpc_autotuner.dataclasses import Config, TunableWeights
from mpc_autotuner.utils import denormalize, normalize


class MetropolisHastings:

    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Metropolis Hastings method. Computes new sets of parameters to try to minimize the laptime
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Save tunable weights and config
        self.tunable_weights = tunable_weights
        self.config = config

        # Domain of parameters and initial theta
        self.domain = self.tunable_weights.get_domain()

        # MPC weights as theta and laptimes
        self.current_theta = normalize(self.tunable_weights.get_initial_weights(), self.domain)
        self.theta = np.empty((0, self.tunable_weights.num_parameters))
        self.laptime_samples = np.empty((0, 1))

        # MH theta at time t and laptime at time t
        self.theta_t = normalize(self.tunable_weights.get_initial_weights(), self.domain)
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
        self.accepted_theta = read_object('accepted_theta.np')
        self.accepted_laptimes = read_object('accepted_laptimes.np')
        self.cost_t = self.compute_cost(self.accepted_laptimes[-1])
        self.theta_t = self.accepted_theta[np.newaxis, -1]
        self.current_theta = self.sample_from_transition_model()

    def add_data_point(self, laptime_sample: float) -> None:
        """
        Perform Metropolis Hastings to get next set of parameters
        :param laptime_sample: laptime of current theta
        """
        assert self.current_theta.shape[1] == self.tunable_weights.num_parameters

        # Store current theta and laptime
        self.theta = np.vstack((self.theta, self.current_theta))
        self.laptime_samples = np.vstack((self.laptime_samples, laptime_sample))

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
            rospy.logwarn(f"Updated MH state. New theta_t={denormalize(self.theta_t, self.domain)},"
                          f" new laptime_t={laptime_sample}.\n"
                          f" Acceptance rate="
                          f"{self.accepted_theta.shape[0] / self.theta.shape[0] * 100}%")
        else:
            rospy.logwarn(f"Rejected theta={denormalize(self.current_theta, self.domain)} "
                          f"with laptime={laptime_sample}")

        # Sample from updated transition model
        self.current_theta = self.sample_from_transition_model()

    def sample_from_transition_model(self) -> np.ndarray:
        """
        Use transition model (Gaussian) to get new parameter value
        :return: new parameters
        """
        next_theta = np.zeros(self.current_theta.shape, dtype=np.float64)

        for i in range(self.tunable_weights.num_parameters):
            next_theta[0, i] = self.bounded_gaussian(self.theta_t[0, i],
                                                     self.config.mh_config.sigma,
                                                     np.array([0.0, 1.0]))
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

    def get_final_solution(self) -> Tuple[np.ndarray, float]:
        """
        Computes argmin of sampled lap times
        :return: (weights that minimize the laptime, minimum laptime)
        """
        theta_opt_ind = np.argmin(self.laptime_samples)
        return self.theta[theta_opt_ind], self.laptime_samples[theta_opt_ind]

    @staticmethod
    def compute_cost(laptime: float) -> float:
        """
        Compute exponential score
        :param laptime: laptime value
        :return: cost of laptime
        """
        res: float = np.exp(-2 * np.sqrt(laptime))
        return res
