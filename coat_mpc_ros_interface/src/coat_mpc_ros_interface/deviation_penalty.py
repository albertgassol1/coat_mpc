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
import os
import sys
from typing import Tuple

import numpy as np
import rospkg

# ROS import
import rospy
import yaml
from coat_mpc_msgs.msg import car_state, penalty_and_lap_count
from coat_mpc.utils.dataclasses import Topics
from scipy.spatial import KDTree

# Messages
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker


class DeviationPenalty:
    def __init__(self) -> None:
        """
        Add penalties for deviating from the reference trajectory
        """

        # Init ROS node
        rospy.init_node("linear_penalty", anonymous=False)

        # Get topics
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("coat_mpc_ros_interface")
        self.topics = self.get_topics(
            os.path.join(package_path, "config/interfaces.yaml")
        )

        # Flags to check if mission started/ended
        self.finished = False

        # Create penalty publisher
        self.pub = rospy.Publisher(
            self.topics.penalty_and_laps, penalty_and_lap_count, queue_size=1
        )

        # Variable to store total penalty applied
        self.penalty = 0

        # Track center
        self.track_center = None

        # Keep number of laps
        self.number_of_laps = -1

        # Get number of laps and config file
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("coat_mpc_ros_interface")
        self.number_of_laps, self.slope = self.get_config(
            os.path.join(package_path, "config/config.yaml")
        )

        # Subscribe to topics
        self.subscribe_to_topics()

    @staticmethod
    def get_config(file_path: str) -> Tuple[int, float]:
        """
        Read config yaml file
        :param file_path: path to yaml file
        :return: (number of laps, penalty slope)
        """
        with open(file_path) as config_file:
            config = yaml.safe_load(config_file)["interface_config"]

        return config["number_of_laps"], config["linear_penalty_slope"]

    @staticmethod
    def get_topics(file_path: str) -> Topics:
        """
        Get topics from yaml file
        :param file_path: path to yaml file
        :return: topics
        """
        with open(file_path) as input_file:
            topics = yaml.safe_load(input_file)["topics"]

        return Topics(
            topics["velocity_estimation"],
            topics["lap_counter"],
            topics["autotuner_state"],
            topics["car_command"],
            topics["penalty_and_laps"],
            topics["autotuner_go"],
            topics["track"],
            topics["namespace"],
        )

    def subscribe_to_topics(self) -> None:
        """
        Subscribe to autotuner state and lap counter ROS topics
        """
        rospy.Subscriber(
            self.topics.velocity_estimation, car_state, self.callback_state
        )
        rospy.Subscriber(self.topics.lap_counter, Int32, self.callback_lap_counter)
        rospy.Subscriber(self.topics.track, Marker, self.callback_track)

    def callback_lap_counter(self, laps: Int32) -> None:
        """callback
        Set start finish flags and publish message
        :param laps: Number of laps ROS message
        """
        self.number_of_laps = laps.data
        if laps.data == self.number_of_laps:
            self.finished = True

        self.publish_penalty(laps.data)

    def callback_state(self, state: car_state) -> None:
        """
        Read state to keep adding penalties
        :param state: current car state wrt the reference trajectory
        """
        if self.finished or self.track_center is None or self.number_of_laps < 1:
            return

        # Get distance to closest trajectory point
        distance, _ = self.track_center.query(np.asarray([state.x, state.y]))

        if distance != np.inf:
            distance = distance if distance > 0.12 else 0.0
            self.penalty += self.linear(distance)

    def callback_track(self, track_info: Marker) -> None:
        """
        Read track middle line and store it
        :param track_info: track information
        """
        if (
            self.track_center is not None
            or track_info.ns != self.topics.track_namespace
        ):
            return

        self.track_center = KDTree(
            np.asarray([np.asarray([point.x, point.y]) for point in track_info.points])
        )

    def publish_penalty(self, laps: int) -> None:
        """
        Publish penalty and number of laps
        :param laps: number of completed laps
        """
        # Publish penalty
        message = penalty_and_lap_count()
        message.header.stamp = rospy.Time.now()
        message.cones_down = 0
        message.laps = laps
        message.penalty = self.penalty

        self.pub.publish(message)

    def linear(self, deviation: float) -> float:
        """
        Apply linear penalty
        :param deviation: deviation from reference trajectory
        :return: time penalty
        """
        return self.slope * deviation


if __name__ == "__main__":
    linear_penalty = DeviationPenalty()
    try:
        while not rospy.is_shutdown():
            rospy.spin()
    except rospy.ROSInterruptException as e:
        print(str(e))
        sys.exit(1)
