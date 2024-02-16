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
