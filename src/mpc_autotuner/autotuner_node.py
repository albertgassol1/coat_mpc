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

import os
from datetime import datetime
from typing import Tuple, Union

import numpy as np
import rospkg
# ROS Include
import rospy
import yaml
from mpc_autotuner.car_interface import CarInterface
from mpc_autotuner.dataclasses import Config, InterfaceConfig, Interfaces, MHConfig, OptimizationConfig, \
    Topics, TunableWeights, WMLConfig
from mpc_autotuner.fssim_interface import FssimInterface


class AutotunerNode:

    def __init__(self) -> None:
        """
        Autotuner ROS node. Read config and launch interfaces.
        """
        # Init ROS node
        rospy.init_node('autotuner_node', anonymous=True)

        # Load config params
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ros_cbo')
        # Create visualization folder
        folder_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.makedirs(os.path.join(package_path, 'visualization/objects'), exist_ok=True)
        os.mkdir(os.path.join(package_path, 'visualization/objects', folder_name))
        interface_config, optimization_config, mh_config, wml_config = \
            self.get_general_config(os.path.join(package_path, 'config/config.yaml'),
                                    os.path.join(package_path, 'visualization/objects',
                                                 folder_name))
        interfaces = self.get_interfaces(os.path.join(package_path, 'config/interfaces.yaml'))
        config = Config(interfaces, interface_config, optimization_config, mh_config, wml_config)

        # Load weights to tune
        tunable_weights = self.get_tunable_weights(
            os.path.join(package_path, 'config/tunable_weights.yaml'))

        if config.interface_config.simulation:
            # FSSIM interface object
            self.interface: Union[FssimInterface, CarInterface] = FssimInterface(config, tunable_weights)
        else:
            # Car interface object
            self.interface = CarInterface(config, tunable_weights)

        rospy.spin()

    @staticmethod
    def get_general_config(config_file_path: str, visualization_objects_path: str) \
            -> Tuple[InterfaceConfig, OptimizationConfig, MHConfig, WMLConfig]:
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
        with open(os.path.join(visualization_objects_path,
                               'config.yaml'), 'w+') as output_config_file:
            yaml.safe_dump(config, output_config_file)

        # Read config
        optimization_config = config['optimization_config']
        interface_config = config['interface_config']
        mh_config = config['metropolis_hastings']
        wml_config = config['wml_config']

        return InterfaceConfig(interface_config['simulation'],
                               interface_config['max_time'],
                               interface_config['max_iterations'],
                               interface_config['number_of_laps'],
                               interface_config['max_deviation'],
                               interface_config['use_deviation_penalty'],
                               interface_config['load_prior_data'],
                               interface_config['optimal_time'],
                               visualization_objects_path,
                               interface_config['prior_data_path']),\
            OptimizationConfig(optimization_config['method'],
                               optimization_config['beta'],
                               optimization_config['acquisition_function'],
                               optimization_config['standardization_batch'],
                               optimization_config['constant_lengthscale'],
                               optimization_config['first_lap_multiplier'],
                               optimization_config['grid_size'],
                               optimization_config['lipschitz_constant'],
                               optimization_config['use_ucb'],
                               optimization_config['prior_mean'],
                               optimization_config['minimum_variance'],
                               optimization_config['number_bo_restarts'],
                               optimization_config['raw_samples'],
                               optimization_config['kernel_lengthscale'],
                               optimization_config['kernel_variance'],
                               optimization_config['gp_variance']), \
            MHConfig(mh_config['sigma']), \
            WMLConfig(wml_config['N'], wml_config['beta'])

    @staticmethod
    def get_interfaces(interfaces_file_path: str) -> Interfaces:
        """
        Get interfaces
        :param interfaces_file_path: path to interfaces file
        :return: interfaces
        """
        with open(interfaces_file_path) as interfaces_file:
            interfaces = yaml.safe_load(interfaces_file)
            topics = interfaces['topics']
            simulation = interfaces['simulation']
            crs = interfaces['crs']

        return Interfaces(
            Topics(topics['velocity_estimation'], topics['lap_counter'], topics['autotuner_state'],
                   topics['car_command'], topics['penalty_and_laps'], topics['autotuner_go'],
                   topics['track'], topics['namespace']),
            simulation['package'], simulation['launch'],
            crs['package'], crs['launch'])

    @staticmethod
    def get_tunable_weights(tunable_weights_file_path: str) -> TunableWeights:
        """
        Get tunable weights
        :param tunable_weights_file_path: path to tunable weights file
        :return: tunable weights
        """

        with open(tunable_weights_file_path) as tunable_weights_file:
            tunable_weights = yaml.safe_load(tunable_weights_file)

        names = [np.asarray(names, dtype=str) for names in tunable_weights['weights_names']]
        lower_bounds = [
            np.asarray(lower_bound, dtype=np.float64)
            for lower_bound in tunable_weights['lower_bounds']
        ]
        upper_bounds = [
            np.asarray(upper_bound, dtype=np.float64)
            for upper_bound in tunable_weights['upper_bounds']
        ]
        initial_values = [
            np.asarray(initial_value, dtype=np.float64)
            for initial_value in tunable_weights['initial_values']
        ]
        reconfigure_names = tunable_weights['reconfigure_names']

        return TunableWeights(names, lower_bounds, upper_bounds, initial_values, reconfigure_names)
