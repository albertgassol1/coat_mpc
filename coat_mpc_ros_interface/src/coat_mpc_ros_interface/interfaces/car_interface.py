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

# System libraries
import os

# Libraries
from typing import Tuple

import numpy as np

import rosnode
import rospkg

# ROS Include
import rospy

# ROS messages
from coat_mpc_msgs.msg import car_state

# Package classes
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc_ros_interface.interfaces.base_interface import BaseInterface


class CarInterface(BaseInterface):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Car interface. Handle simulation and trigger optimization
        :param config: config parameters
        :param tunable_weights: tunable weights
        """
        # ROS timers
        self.start_time = rospy.Time(0)
        # Call parent constructor
        super().__init__(config, tunable_weights)

    def get_car_launch_settings(self) -> Tuple[str]:
        """
        Returns path to launch files
        :return: car launch path
        """
        rospack = rospkg.RosPack()
        return os.path.join(
            rospack.get_path(self.config.interfaces.crs_package),
            "launch",
            self.config.interfaces.crs_launch,
        )

    def restart(self, lap_time: float, use_lap_time: bool, optimize: bool) -> None:
        """
        Restart. First do a bayesian opt step and load new parameters" \
        :param lap_time: laptime of current iteration
        :param use_lap_time: bool to use the given laptime
        :param optimize: bool to trigger the bayesian optimization
        """

        # Kill record and penalty nodes
        nodes_to_kill = ["/deviation_penalty"]
        all_nodes = rosnode.get_node_names()
        for node in all_nodes:
            if "/record" in node:
                nodes_to_kill.append(node)
        rosnode.kill_nodes(nodes_to_kill)

        super().restart(lap_time, use_lap_time, optimize)

        # Start simulation
        try:
            self.lap_counter_object.reset()
            self.mpc_interface.reset_constraints()
            # Reset mpc interface
            self.mpc_interface.reset_interface()
            self.mpc_interface.send_new_params(self.optimizer.current_theta)
            # Save current weights
            np.save(
                os.path.join(
                    self.config.interface_config.visualization_objects_path,
                    "current_theta.np",
                ),
                self.optimizer.current_theta,
            )
            # Reset lap counter
            self.launch_deviation_penalty()
        except:
            rospy.loginfo("Something went wrong!")

    def callback_state(self, vel_est: car_state) -> None:
        """
        Read velocity estimation data to perform checks
        :param vel_est: velocity estimation ROS message
        """
        # Check if we reached the maximum number of iterations
        if self.iterations > 0 and not self.saved:
            # Save visualization data
            self.saved = True
            rospy.loginfo("LOGGING DATA")

            # For weights, we only save them once at the beginning
            if self.iterations == 1:
                self.save_weights()
            self.save_visualization_data()

    def get_lap_time_and_reset(self) -> float:
        """
        Computes current lap time and resets timers
        :return: current laptime
        """
        current_time = rospy.get_rostime().to_sec()
        lap_time: float = current_time - self.start_time.to_sec()
        self.start_time = rospy.get_rostime()
        return lap_time
