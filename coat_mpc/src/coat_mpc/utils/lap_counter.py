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
