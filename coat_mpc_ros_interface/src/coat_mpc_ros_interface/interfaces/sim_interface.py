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
import time
from typing import Tuple

import rosnode
import rospkg

# ROS Include
import rospy

# ROS messages
from coat_mpc_msgs.msg import car_state

# Package classes
from coat_mpc.utils.dataclasses import Config, TunableWeights
from coat_mpc_ros_interface.interfaces.base_interface import BaseInterface


class SimInterface(BaseInterface):
    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Simulator interface. Handle simulation and trigger optimization
        :param config: config parameters
        :param tunable_weights: tunable weights
        """
        # ROS timers
        self.start_time = None
        # Call parent constructor
        super().__init__(config, tunable_weights)

    def get_car_launch_settings(self) -> Tuple[str]:
        """
        Returns path to launch files
        :return: simulation launch path
        """
        rospack = rospkg.RosPack()
        return os.path.join(
            rospack.get_path(self.config.interfaces.simulation_package),
            "launch",
            self.config.interfaces.simulation_launch,
        )

    def restart(self, lap_time: float, use_lap_time: bool, optimize: bool) -> None:
        """
        Restart simulation. First do a bayesian opt step and load new parameters" \
        :param lap_time: laptime of current iteration
        :param use_lap_time: bool to use the given laptime
        :param optimize: bool to trigger the bayesian optimization
        """

        # Kill simulation and rerun it
        nodes_to_kill = rosnode.get_node_names()
        if "/rosout" in nodes_to_kill:
            nodes_to_kill.remove("/rosout")
        nodes_to_kill.remove("/coat_mpc_ros_interface")
        rosnode.kill_nodes(nodes_to_kill)
        self.lap_counter = -1
        time.sleep(3)

        super().restart(lap_time, use_lap_time, optimize)

        # Start simulation
        try:
            # Reset lap counter
            self.lap_counter_object.reset()
            self.mpc_interface.reset_constraints()
            self.start()
            time.sleep(2)

            # Reset mpc interface
            self.mpc_interface.reset_interface()

        except:
            rospy.loginfo("Something went wrong! Restart again")
            self.restart(lap_time, True, True)

    def callback_state(self, vel_est: car_state) -> None:
        """
        Read velocity estimation data to perform checks
        :param vel_est: velocity estimation ROS message
        """
        # Check if we reached the maximum number of iterations
        if self.iterations > 1 and not self.saved:
            # Save visualization data
            self.saved = True
            rospy.loginfo("LOGGING DATA")

            # For weights, we only save them once at the beginning
            if self.iterations == 2:
                self.save_weights()
            self.save_visualization_data()

        # Check if velocity is <1 and increase counter. When we reach threshold -> trigger restart
        if vel_est.vx < 0.5:
            self.vel_est_counter += 1
        else:
            self.vel_est_counter = 0

        if self.vel_est_counter >= 10000:
            rospy.logwarn("CAR STUCK BEFORE STARTING! RESTARTING SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)

        # Check vehicle status
        self.check_status(vel_est.vx, vel_est.vy)

    def check_status(self, vel_x: float, vel_y: float) -> None:
        """
        Check car status and stop iteration if the car is stuck or out of the track
        :param vel_x: current longitudinal velocity
        :param vel_y: current lateral velocity
        """

        if self.lap_counter == -1:
            return

        # Check deviation from centerline
        if (
            self.mpc_interface.track_constraint
            > self.config.interface_config.max_deviation
        ):
            rospy.logwarn("CAR OUT OF TRACK! RESTART SIMULATION")
            self.mpc_interface.reset_constraints()
            self.iterations += 1
            self.restart_simulation(-1, False, True)
            return

        # Check time
        if self.start_time is None:
            # We are not in a lap
            return

        current_time = rospy.get_rostime().to_sec()
        elapsed_time = current_time - self.start_time.to_sec()
        if elapsed_time > self.config.interface_config.max_time:
            rospy.logwarn("MAX LAP_TIME REACHED! RESTART SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)
            return

        # Check velocity
        if (
            elapsed_time > 5
            and vel_x < 0.5
            and vel_y < 0.5
            and self.vel_est_counter >= 10000
        ):
            rospy.logwarn("CAR STUCK! RESTART SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)

    def get_lap_time_and_reset(self) -> float:
        """
        Computes current lap time and resets timers
        :return: current laptime
        """
        current_time = rospy.get_rostime().to_sec()
        lap_time: float = current_time - self.start_time.to_sec()
        self.start_time = None
        return lap_time
