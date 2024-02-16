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

# ROS Include
import rospy

# ROS messages
from coat_mpc_msgs.msg import car_state
from coat_mpc.utils.dataclasses import Topics
from std_msgs.msg import Int32


class LapCounter:
    def __init__(self, topics: Topics) -> None:
        """
        Lap counter class. Count laps
        :param topics: topics to subscribe and publish
        """

        # Save topics
        self.topics = topics

        # Number of laps
        self.laps = -1

        # Start flag
        self.start_flag = False

        # Timer
        self.start_time = None

        # Laps publisher
        self.lap_counter_pub = rospy.Publisher(
            self.topics.lap_counter, Int32, queue_size=10
        )

        self.subscribe_to_topics()

    def reset(self) -> None:
        """
        Resets laps and start flag
        """
        self.laps = -1
        self.start_flag = False
        self.start_time = None

    def subscribe_to_topics(self) -> None:
        """
        Subscribe to necessary topics
        """
        rospy.Subscriber(
            self.topics.velocity_estimation, car_state, self.callback_state
        )

    def callback_state(self, state: car_state) -> None:
        """
        Read state and update lap counter in necessary
        :param state: state ROS message
        """
        if not self.start_flag and state.vx_b > 1.3:
            self.start_flag = True

        if not self.start_flag:
            return

        if (0.15 < state.x < 1.0) and (-2.0 < state.y < -0.3):
            if self.start_time is None:
                self.publish_laps()
                return
            elif (rospy.Time.now() - self.start_time).to_sec() > 2.5:
                self.publish_laps()
                return

    def publish_laps(self) -> None:
        """
        Publish number of laps
        """
        self.laps += 1
        self.start_time = rospy.Time.now()
        msg = Int32()
        msg.data = self.laps
        self.lap_counter_pub.publish(msg)
