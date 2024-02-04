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
import os
import sys
from typing import Tuple

import numpy as np
import rospkg
# ROS import
import rospy
import yaml
from crs_msgs.msg import car_state_cart, penalty_and_lap_count
from mpc_autotuner.dataclasses import Topics
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
        rospy.init_node('linear_penalty', anonymous=False)

        # Get topics
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ros_cbo')
        self.topics = self.get_topics(os.path.join(package_path, 'config/interfaces.yaml'))

        # Flags to check if mission started/ended
        self.finished = False

        # Create penalty publisher
        self.pub = rospy.Publisher(self.topics.penalty_and_laps, penalty_and_lap_count, queue_size=1)

        # Variable to store total penalty applied
        self.penalty = 0

        # Track center
        self.track_center = None

        # Keep number of laps
        self.number_of_laps = -1

        # Get number of laps and config file
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('ros_cbo')
        self.number_of_laps, self.slope = self.get_config(os.path.join(package_path, 'config/config.yaml'))

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
            config = yaml.safe_load(config_file)['interface_config']

        return config['number_of_laps'], config['linear_penalty_slope']

    @staticmethod
    def get_topics(file_path: str) -> Topics:
        """
        Get topics from yaml file
        :param file_path: path to yaml file
        :return: topics
        """
        with open(file_path) as input_file:
            topics = yaml.safe_load(input_file)['topics']

        return Topics(topics['velocity_estimation'], topics['lap_counter'],
                      topics['autotuner_state'], topics['car_command'], topics['penalty_and_laps'],
                      topics['autotuner_go'], topics['track'], topics['namespace'])

    def subscribe_to_topics(self) -> None:
        """
        Subscribe to autotuner state and lap counter ROS topics
        """
        rospy.Subscriber(self.topics.velocity_estimation, car_state_cart, self.callback_state)
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

    def callback_state(self, state: car_state_cart) -> None:
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
        if self.track_center is not None or track_info.ns != self.topics.track_namespace:
            return

        self.track_center = KDTree(np.asarray([np.asarray([point.x, point.y]) for point in track_info.points]))

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


if __name__ == '__main__':
    linear_penalty = DeviationPenalty()
    try:
        while not rospy.is_shutdown():
            rospy.spin()
    except rospy.ROSInterruptException as e:
        print(str(e))
        sys.exit(1)
