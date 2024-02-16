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

import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import rospkg

# ROS Include
import rospy
import yaml
from coat_mpc.utils.dataclasses import (
    Config,
    InterfaceConfig,
    Interfaces,
    MHConfig,
    OptimizationConfig,
    Topics,
    TunableWeights,
    WMLConfig,
)
from coat_mpc_ros_interface.interfaces import interface_dict, BaseInterface


class AutotunerNode:
    def __init__(self) -> None:
        """
        Autotuner ROS node. Read config and launch interfaces.
        """
        # Init ROS node
        rospy.init_node("autotuner_node", anonymous=True)

        # Load config params
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("coat_mpc_ros_interface")
        # Create visualization folder
        folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.makedirs(os.path.join(package_path, "visualization/objects"), exist_ok=True)
        os.mkdir(os.path.join(package_path, "visualization/objects", folder_name))
        (
            interface_config,
            optimization_config,
            mh_config,
            wml_config,
        ) = self.get_general_config(
            os.path.join(package_path, "config/config.yaml"),
            os.path.join(package_path, "visualization/objects", folder_name),
        )
        interfaces = self.get_interfaces(
            os.path.join(package_path, "config/interfaces.yaml")
        )
        config = Config(
            interfaces, interface_config, optimization_config, mh_config, wml_config
        )

        # Load weights to tune
        tunable_weights = self.get_tunable_weights(
            os.path.join(package_path, "config/tunable_weights.yaml")
        )

        self.interface: BaseInterface = interface_dict[
            config.interface_config.simulation
        ](config, tunable_weights)

        rospy.spin()

    @staticmethod
    def get_general_config(
        config_file_path: str, visualization_objects_path: str
    ) -> Tuple[InterfaceConfig, OptimizationConfig, MHConfig, WMLConfig]:
        """
        Get config parameters
        :param config_file_path: path to config file
        :param visualization_objects_path: path to directory to save visualization objects
        :return: (InterfaceConfig, OptimizationConfig).
        Tuple containing the config parameters
        """
        with open(config_file_path) as config_file:
            config = yaml.safe_load(config_file)

        # Save config to visualization folder
        with open(
            os.path.join(visualization_objects_path, "config.yaml"), "w+"
        ) as output_config_file:
            yaml.safe_dump(config, output_config_file)

        # Read config
        optimization_config = config["optimization_config"]
        interface_config = config["interface_config"]
        mh_config = config["metropolis_hastings"]
        wml_config = config["wml_config"]

        return (
            InterfaceConfig(
                interface_config["simulation"],
                interface_config["max_time"],
                interface_config["max_iterations"],
                interface_config["number_of_laps"],
                interface_config["max_deviation"],
                interface_config["use_deviation_penalty"],
                interface_config["load_prior_data"],
                interface_config["optimal_time"],
                visualization_objects_path,
                interface_config["prior_data_path"],
            ),
            OptimizationConfig(
                optimization_config["method"],
                optimization_config["beta"],
                optimization_config["acquisition_function"],
                optimization_config["standardization_batch"],
                optimization_config["constant_lengthscale"],
                optimization_config["first_lap_multiplier"],
                optimization_config["grid_size"],
                optimization_config["lipschitz_constant"],
                optimization_config["use_ucb"],
                optimization_config["prior_mean"],
                optimization_config["minimum_variance"],
                optimization_config["number_bo_restarts"],
                optimization_config["raw_samples"],
                optimization_config["kernel_lengthscale"],
                optimization_config["kernel_variance"],
                optimization_config["gp_variance"],
            ),
            MHConfig(mh_config["sigma"]),
            WMLConfig(wml_config["N"], wml_config["beta"]),
        )

    @staticmethod
    def get_interfaces(interfaces_file_path: str) -> Interfaces:
        """
        Get interfaces
        :param interfaces_file_path: path to interfaces file
        :return: interfaces
        """
        with open(interfaces_file_path) as interfaces_file:
            interfaces = yaml.safe_load(interfaces_file)
            topics = interfaces["topics"]
            simulation = interfaces["simulation"]
            crs = interfaces["crs"]

        return Interfaces(
            Topics(
                topics["velocity_estimation"],
                topics["lap_counter"],
                topics["autotuner_state"],
                topics["car_command"],
                topics["penalty_and_laps"],
                topics["autotuner_go"],
                topics["track"],
                topics["namespace"],
            ),
            simulation["package"],
            simulation["launch"],
            crs["package"],
            crs["launch"],
        )

    @staticmethod
    def get_tunable_weights(tunable_weights_file_path: str) -> TunableWeights:
        """
        Get tunable weights
        :param tunable_weights_file_path: path to tunable weights file
        :return: tunable weights
        """

        with open(tunable_weights_file_path) as tunable_weights_file:
            tunable_weights = yaml.safe_load(tunable_weights_file)

        weights = tunable_weights["weights"]

        names: List[np.ndarray] = list()
        lower_bounds: List[np.ndarray] = list()
        upper_bounds: List[np.ndarray] = list()
        initial_values: List[np.ndarray] = list()
        for name, params in weights.items():
            names.append(np.asarray(name, dtype=str))
            lower_bounds.append(np.asarray(params["lower_bound"], dtype=np.float64))
            upper_bounds.append(np.asarray(params["upper_bound"], dtype=np.float64))
            initial_values.append(np.asarray(params["initial_value"], dtype=np.float64))
        reconfigure_names = tunable_weights["reconfigure_names"]

        return TunableWeights(
            names, lower_bounds, upper_bounds, initial_values, reconfigure_names
        )
