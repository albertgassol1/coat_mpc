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
            rospack.get_path(self.config.interfaces.run_package),
            "launch",
            self.config.interfaces.run_launch,
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
