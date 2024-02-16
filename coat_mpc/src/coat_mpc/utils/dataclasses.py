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
    state_estimation: str
    lap_counter: str
    penalty_and_laps: str
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
    run_package: str
    run_launch: str


@dataclass(frozen=True)
class Config:
    interfaces: Interfaces
    interface_config: InterfaceConfig
    optimization_config: OptimizationConfig
    mh_config: MHConfig
    wml_config: WMLConfig
