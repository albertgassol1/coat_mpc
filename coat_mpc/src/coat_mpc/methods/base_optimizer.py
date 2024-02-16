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
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

# Package classes
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc.utils.utils import normalize


class BaseOptimizer(ABC):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Base class for the optimizer
        :param config: config parameters
        :param tunable_weights: tunable weights
        """
        # Save tunable weights and config
        self.tunable_weights = tunable_weights
        self.config = config

        # Domain of parameters and initial theta
        self.domain = self.tunable_weights.get_domain()

        # Current theta
        self.current_theta = normalize(
            self.tunable_weights.get_initial_weights(), self.domain
        )

    @abstractmethod
    def load_prior_data(filename: str) -> None:
        """
        Load prior data from file
        :param filename: file to load
        """
        pass

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

    def get_final_solution(self) -> Tuple[np.ndarray, float]:
        """
        Computes argmin of sampled lap times
        :return: (weights that minimize the laptime, minimum laptime)
        """
        theta_opt_ind = np.argmin(self.laptime_samples)
        return self.theta[theta_opt_ind], self.laptime_samples[theta_opt_ind]
