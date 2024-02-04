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

from math import cos, sin
# Libraries
from typing import Dict, List, Tuple

# Dynamic reconfigure library
import dynamic_reconfigure.client
import numpy as np
# ROS Include
import rospy
# ROS messages
from crs_msgs.msg import car_input, car_state_cart
# Module imports
from mpc_autotuner.dataclasses import Config, TunableWeights
from scipy.spatial import KDTree
from visualization_msgs.msg import Marker


class MpcInterface:

    def __init__(self, config: Config, tunable_weights: TunableWeights,
                 domain: np.ndarray) -> None:
        """
        Mpc interface to send new parameters and monitor its state
        :param config: config params
        :param tunable_weights: weights to tune
        :param domain: domain of the weights to tune
        """

        # Save config
        self.config = config
        # Save tunable weights
        self.tunable_weights = tunable_weights
        # Save weights domain
        self.domain = domain

        # List of dictionaries
        self.weights_dictionaries: List[Dict[str, float]] = list()

        # Dynamic reconfigure clients
        self.dynamic_reconfigure_clients: List[dynamic_reconfigure.cient.Client] = list()

        # Constraints on track boundaries
        self.track_constraint = 0

        # Flag to know if we are receiving commands
        self.command_received = False

        # Track center kd-tree
        self.track_center = None

        # Subscribe to topics
        self.subscribe_to_topics()

    def subscribe_to_topics(self) -> None:
        """
        Subscribe to car commands and autotuner state ROS topics
        """
        # ROS Subscriber to read MPC states (needed for BO constraints)
        rospy.Subscriber(self.config.interfaces.topics.velocity_estimation, car_state_cart,
                         self.callback_state)
        rospy.Subscriber(self.config.interfaces.topics.car_command, car_input,
                         self.callback_car_command)
        rospy.Subscriber(self.config.interfaces.topics.track, Marker,
                         self.callback_track)

    def connect_to_server(self) -> None:
        """
        Gets dynamic reconfigure clients and stores them
        """

        for i, client_name in enumerate(self.tunable_weights.reconfigure_names):
            if i == 0:
                client = dynamic_reconfigure.client.Client(client_name,
                                                           timeout=30,
                                                           config_callback=self.config_callback)
            else:
                client = dynamic_reconfigure.client.Client(client_name, timeout=30)
            self.dynamic_reconfigure_clients.append(client)

    def config_callback(self, _: None) -> None:
        """
        Config callback. This just helps to print the weights
        """
        rospy.loginfo("SENT NEW PARAMS")
        rospy.loginfo(self.weights_dictionaries)

    def callback_state(self, state: car_state_cart) -> None:
        """
        Reads state message from mpc and computes track constraints.
        Adds them to track_constraints vector
        :param state: car state ROS message
        """
        if self.track_center is None or state.vx_w < 0.5:
            return
        # Get the distance to the closest trajectory point
        distance, _ = self.track_center.query(np.asarray([state.x, state.y]))
        self.track_constraint = distance

    def callback_car_command(self, msg: car_input) -> None:
        """
        Read car command to see when the car starts moving
        :param msg: car_input ROS message
        """
        if msg.torque > -15:
            self.command_received = True

    def callback_track(self, track_info: Marker) -> None:
        """
        Read track middle line and store it
        :param track_info: track information
        """
        if self.track_center is not None or track_info.ns != self.config.interfaces.topics.track_namespace:
            return
        self.track_center = KDTree(np.asarray([np.asarray([point.x, point.y]) for point in track_info.points]))

    def reset_interface(self) -> None:
        """
        Reset command received to False
        """
        self.command_received = False

    def received_command(self) -> bool:
        """
        Returns received command bool
        :return: received command bool
        """
        return self.command_received

    def reset_constraints(self) -> None:
        """
        Reset track constraints vector
        """
        self.track_constraint = 0

    def send_new_params(self, theta: np.ndarray) -> None:
        """
        Sends new set of parameters to MPC controller using dynamic reconfigure
        :param theta: new parameters to send
        """
        # Denormalize theta
        theta = np.squeeze(theta) * (self.domain[:, 1] - self.domain[:, 0]) + self.domain[:, 0]

        # Create dynamic reconfigure dictionaries
        self.weights_dictionaries = list()
        i = 0
        for names in self.tunable_weights.names:
            reconfigure_dictionary = dict(zip(names, theta[i:(i + len(names))]))
            self.weights_dictionaries.append(reconfigure_dictionary)
            i += len(names)

        # Send reconfigure
        # Save parameters
        rospy.loginfo("SENDING")
        try:
            for i, weight_dictionary in enumerate(self.weights_dictionaries):
                self.dynamic_reconfigure_clients[i].\
                    update_configuration(weight_dictionary)
        except:
            rospy.logerr("COULDN'T SEND NEW PARAMS, SOMETHING FAILED!")
