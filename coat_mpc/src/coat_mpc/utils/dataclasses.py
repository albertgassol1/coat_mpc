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

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TunableWeights:
    names: List[np.ndarray]
    lower_bounds: List[np.ndarray]
    upper_bounds: List[np.ndarray]
    initial_values: List[np.ndarray]
    reconfigure_names: List[str]
    num_parameters: int = 0

    def __post_init__(self) -> None:
        """
        Initialize num_parameters
        """
        for name_list in self.names:
            self.num_parameters += len(name_list)

    def names_to_list(self) -> List[str]:
        """
        Convert all parameters names to a unique list of strings
        :return: list of names
        """
        all_names: List[str] = list()
        for name_list in self.names:
            all_names += name_list.tolist()

        return all_names

    def get_domain(self) -> np.ndarray:
        """
        Get domain of weights -> [lower bound, upper bound]
        :return: domain of weights
        """
        domain = np.empty((self.num_parameters, 2))
        i = 0
        for lower_bound in self.lower_bounds:
            domain[i : (i + len(lower_bound)), 0] = lower_bound
            i += len(lower_bound)
        i = 0
        for upper_bound in self.upper_bounds:
            domain[i : (i + len(upper_bound)), 1] = upper_bound
            i += len(upper_bound)
        return domain

    def get_initial_weights(self) -> np.ndarray:
        """
        Return initial weights as a numpy array
        :return: initial weights
        """
        initial_weights = np.empty((1, self.num_parameters))
        i = 0
        for initial_value in self.initial_values:
            initial_weights[0, i : (i + len(initial_value))] = initial_value
            i += len(initial_value)
        return initial_weights


@dataclass(frozen=True)
class OptimizationConfig:
    method: str
    beta: float
    acquisition_function: str
    standardization_batch: int
    constant_lengthscale: bool
    first_lap_multiplier: float
    grid_size: int
    lipschitz_constant: float
    use_ucb: bool
    prior_mean: bool
    minimum_variance: float
    number_bo_restarts: int
    raw_samples: int
    kernel_lengthscale: float
    kernel_variance: float
    gp_variance: float


@dataclass(frozen=True)
class MHConfig:
    sigma: float


@dataclass(frozen=True)
class WMLConfig:
    N: int
    beta: float


@dataclass(frozen=True)
class InterfaceConfig:
    simulation: bool
    max_time: float
    max_iterations: int
    number_of_laps: int
    max_deviation: float
    use_deviation_penalty: bool
    load_prior_data: bool
    optimal_time: float
    visualization_objects_path: str
    prior_data_path: str = ""


@dataclass(frozen=True)
class Topics:
    velocity_estimation: str
    lap_counter: str
    autotuner_state: str
    car_command: str
    penalty_and_laps: str
    autotuner_go: str
    track: str
    track_namespace: str


@dataclass(frozen=True)
class Frames:
    world: str
    base_link: str


@dataclass(frozen=True)
class Interfaces:
    topics: Topics
    simulation_package: str
    simulation_launch: str
    crs_package: str
    crs_launch: str


@dataclass(frozen=True)
class Config:
    interfaces: Interfaces
    interface_config: InterfaceConfig
    optimization_config: OptimizationConfig
    mh_config: MHConfig
    wml_config: WMLConfig
