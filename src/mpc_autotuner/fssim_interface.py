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
import pickle
import time
from typing import Tuple, Union

import numpy as np
import roslaunch
import rosnode
import rospkg
# ROS Include
import rospy
from crbo.crbo import ConfidenceRegionBayesianOptimization
# ROS messages
from crs_msgs.msg import car_state_cart, penalty_and_lap_count
from mpc_autotuner.bayesopt import BayesianOptimizer
# Package classes
from mpc_autotuner.dataclasses import Config, TunableWeights
from mpc_autotuner.lap_counter import LapCounter
from mpc_autotuner.metropolis_hastings import MetropolisHastings
from mpc_autotuner.wml import WeightedMaximumLikelihood
from mpc_autotuner.mpc_interface import MpcInterface


class FssimInterface:

    def __init__(self, config: Config, tunable_weights: TunableWeights) -> None:
        """
        Simulator interface. Handle simulation and trigger optimization
        :param config: config parameters
        :param tunable_weights: tunable weights
        """

        # Save config and tunable weights
        self.config = config
        self.tunable_weights = tunable_weights

        # Optimizer method
        if self.config.optimization_config.method == 'BO':
            self.optimizer: Union[BayesianOptimizer,
                                  MetropolisHastings,
                                  ConfidenceRegionBayesianOptimization,
                                  WeightedMaximumLikelihood] = \
                BayesianOptimizer(config, tunable_weights)
        elif self.config.optimization_config.method == 'MH':
            self.optimizer = MetropolisHastings(config, tunable_weights)
        elif self.config.optimization_config.method == "CRBO":
            self.optimizer = ConfidenceRegionBayesianOptimization(config, tunable_weights)
        elif self.config.optimization_config.method == "WML":
            self.optimizer = WeightedMaximumLikelihood(config, tunable_weights)
        else:
            raise NotImplementedError(f"Method: {self.config.optimization_config.method} "
                                      f"not implemented")

        # MPC interface object (send and receive messages from MPC)
        self.mpc_interface = MpcInterface(self.config, self.tunable_weights,
                                          self.tunable_weights.get_domain())

        # Subscribe to ROS topics
        self.subscribe_to_topics()

        # ROS timers
        self.start_time = None

        # Completed laps
        self.lap_counter = -1

        # Lap counter object
        self.lap_counter_object = LapCounter(self.config.interfaces.topics)

        # Counter for velocity near 0 (car stuck)
        self.vel_est_counter = 0

        # Number of iterations of the BO
        self.iterations = 0
        self.saved = False

        # Array to save penalty at each iteration
        self.deviation_penalty = np.empty(0)

        # Load prior data
        if self.config.interface_config.load_prior_data:
            self.load_prior_data(self.config.interface_config.prior_data_path)

        # Launch paths
        self.simulation_launch, self.penalty_launch = self.get_launch_settings()

        # Start new simulation
        self.start_simulation()

        rospy.spin()

    def get_launch_settings(self) -> Tuple[str, str]:
        """
        Returns path to launch files
        :return: (simulation launch path, linear penalty launch path)
        """
        rospack = rospkg.RosPack()
        penalty_path = os.path.join(rospack.get_path('ros_cbo'), "launch",
                                    "deviation_penalty.launch")
        simulation_path = os.path.join(rospack.get_path(self.config.interfaces.simulation_package),
                                       "launch", self.config.interfaces.simulation_launch)
        return simulation_path, penalty_path

    def subscribe_to_topics(self) -> None:
        """
        Subscribes to ROS topics
        """
        # Create subscribers
        rospy.Subscriber(self.config.interfaces.topics.penalty_and_laps,
                         penalty_and_lap_count,
                         self.callback_counter,
                         queue_size=10)

        rospy.Subscriber(self.config.interfaces.topics.velocity_estimation,
                         car_state_cart,
                         self.callback_state,
                         queue_size=1)

    def load_prior_data(self, path: str) -> None:
        """
        Option to load data from previous runs of the optimization
        :param path: path to file with prior data
        """

        # Read pickable objects
        with open(path + "/deviation_penalty.np", 'rb') as infile:
            self.deviation_penalty = pickle.load(infile)

    def start_simulation(self) -> None:
        """
        Launch simulation and connect to reconfigure server
        """
        # Prepare to launch mission
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        # Launch simulation
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.simulation_launch])
        # Needed to launch nodes prom this node
        roslaunch.pmon._init_signal_handlers = self.dummy_function
        rospy.loginfo("Starting simulation")
        launch.start()
        self.launch_deviation_penalty()

        # Connect MPC interface to MPC server for dynamic reconfigure
        self.mpc_interface.connect_to_server()
        self.mpc_interface.send_new_params(self.optimizer.current_theta)

    def launch_deviation_penalty(self) -> None:
        """
        Launch penalty node
        """
        # Prepare to launch mission
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        # Launch simulation
        launch = roslaunch.parent.ROSLaunchParent(uuid, [self.penalty_launch])
        # Needed to launch nodes prom this node
        roslaunch.pmon._init_signal_handlers = self.dummy_function
        rospy.loginfo("Starting deviation penalty node")
        launch.start()

    def dummy_function(self) -> None:
        """
        Needed for launching missions
        """
        # Needed to launch nodes
        pass

    def restart_simulation(self, lap_time: float, use_lap_time: bool, optimize: bool) -> None:
        """
        Restart simulation. First do a bayesian opt step and load new parameters" \
        :param lap_time: laptime of current iteration
        :param use_lap_time: bool to use the given laptime
        :param optimize: bool to trigger the bayesian optimization
        """

        # If the maximum number of iterations is reached
        if self.iterations == self.config.interface_config.max_iterations:
            rospy.loginfo("OPTIMIZATION ENDED!")
            optimal_weights, optimal_laptime = self.optimizer.get_final_solution()
            rospy.loginfo(f"Optimal weights are {optimal_weights}. "
                  f"These weights achieve a laptime of {optimal_laptime} seconds")
            time.sleep(2)
            os.system("rosnode kill -a")
            return

        # If we are restarting due to the car getting stuck
        rospy.loginfo("ITERATIONS: " + str(self.iterations))
        if not use_lap_time:
            lap_time = self.config.interface_config.max_time

            # This helps to visualize when the car got stuck
            self.deviation_penalty = np.hstack((self.deviation_penalty, -1.0))

        # Kill simulation and rerun it
        nodes_to_kill = rosnode.get_node_names()
        if '/rosout' in nodes_to_kill:
            nodes_to_kill.remove('/rosout')
        nodes_to_kill.remove('/ros_cbo')
        rosnode.kill_nodes(nodes_to_kill)
        self.lap_counter = -1

        time.sleep(3)

        # Run optimization
        stop = False
        if optimize:
            # Pass laptime and constraints to the bayesian optimizer
            stop = self.optimizer.add_data_point(lap_time*4)

        # If we have reached the stopping criteria, stop the optimization
        if stop:
            rospy.loginfo("STOPPING CRITERIA REACHED")
            optimal_weights, optimal_laptime = self.optimizer.get_final_solution()
            rospy.loginfo(f"Optimal weights are {optimal_weights}. "
                  f"These weights achieve a laptime of {optimal_laptime} seconds")
            time.sleep(2)
            os.system("rosnode kill -a")
            return

        # Restart velocity estimation counter
        self.vel_est_counter = 0
        self.saved = False

        # Start simulation
        try:
            # Reset lap counter
            self.lap_counter_object.reset()
            self.mpc_interface.reset_constraints()
            self.start_simulation()
            time.sleep(2)

            # Reset mpc interface
            self.mpc_interface.reset_interface()

        except:
            rospy.loginfo("Something went wrong! Restart again")
            self.restart_simulation(lap_time, True, True)

    def save_weights(self) -> None:
        """
        Save weights to pickle objects
        """
        with open(
                os.path.join(self.config.interface_config.visualization_objects_path, 'domain.np'),
                'wb') as domain_file:
            pickle.dump(self.optimizer.domain, domain_file)
        with open(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'parameters.lst'), 'wb') as parameters_list_file:

            pickle.dump(self.tunable_weights.names_to_list(), parameters_list_file)

    def save_visualization_data(self) -> None:
        """
        Save visualization data to pickle objects
        """

        def save_to_file(file_name: str, data: np.ndarray) -> None:
            """
            Save data to specific file
            :param file_name: file name path
            :param data: data to save
            """
            with open(file_name, 'wb') as output_file:
                pickle.dump(data, output_file)

        save_to_file(
            os.path.join(self.config.interface_config.visualization_objects_path, 'theta.np'),
            self.optimizer.theta)
        save_to_file(
            os.path.join(self.config.interface_config.visualization_objects_path, 'laptimes.np'),
            self.optimizer.laptime_samples)

        save_to_file(
            os.path.join(self.config.interface_config.visualization_objects_path, 'deviation_penalty.np'),
            self.deviation_penalty)

        if self.config.optimization_config.method == "BO":
            save_to_file(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'normalized_laptimes.np'),
                self.optimizer.normalized_laptime_samples)
        elif self.config.optimization_config.method == "MH":
            save_to_file(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'accepted_theta.np'),
                self.optimizer.accepted_theta)
            save_to_file(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'accepted_laptimes.np'),
                self.optimizer.accepted_laptimes)
        elif self.config.optimization_config.method == "WML":
            save_to_file(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'policy_mean.np'),
                self.optimizer.policy_mean)
            save_to_file(
                os.path.join(self.config.interface_config.visualization_objects_path,
                             'policy_covariance.np'),
                self.optimizer.policy_covariance)

    def callback_state(self, vel_est: car_state_cart) -> None:
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
        if vel_est.vx_b < 0.5:
            self.vel_est_counter += 1
        else:
            self.vel_est_counter = 0

        if self.vel_est_counter >= 10000:
            rospy.logwarn("CAR STUCK BEFORE STARTING! RESTARTING SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)

        # Check vehicle status
        self.check_status(vel_est.vx_b, vel_est.vy_b)

    def check_status(self, vel_x: float, vel_y: float) -> None:
        """
        Check car status and stop iteration if the car is stuck or out of the track
        :param vel_x: current longitudinal velocity
        :param vel_y: current lateral velocity
        """

        if self.lap_counter == -1:
            return

        # Check deviation from centerline
        if self.mpc_interface.track_constraint > self.config.interface_config.max_deviation:
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
        elapsed_time = (current_time - self.start_time.to_sec())
        if elapsed_time > self.config.interface_config.max_time:
            rospy.logwarn("MAX LAP_TIME REACHED! RESTART SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)
            return

        # Check velocity
        if elapsed_time > 5 and vel_x < 0.5 and vel_y < 0.5 and self.vel_est_counter >= 10000:
            rospy.logwarn("CAR STUCK! RESTART SIMULATION")
            self.iterations += 1
            self.restart_simulation(-1, False, True)

    def callback_counter(self, msg: penalty_and_lap_count) -> None:
        """
        Read lap counter to compute lap time, read penalty and add penalty
        :param msg: ROS message containing counter and penalty info
        """
        # Save number of laps
        self.lap_counter = msg.laps

        # If we are not in the initial lap or final lap, don't take it into account
        if self.lap_counter not in (1, self.config.interface_config.number_of_laps):
            return

        # Initial lap, just initialize timers
        if self.lap_counter == 1:
            self.start_time = rospy.get_rostime()
            return

        # Lap is finalized, compute lap time (2s penalty for each cone down)
        lap_time = self.get_lap_time_and_reset()
        optimize = True

        rospy.loginfo(f"laptime: {lap_time}, penalty: {msg.penalty}")
        self.deviation_penalty = np.hstack((self.deviation_penalty, msg.penalty))

        # Add penalty if needed
        if self.config.interface_config.use_deviation_penalty:
            lap_time = lap_time + msg.penalty

        rospy.loginfo(f"Lapcounter and laptime: {msg.laps}, {lap_time}s")
        self.iterations += 1

        # Restart simulation with new set of parameters
        self.restart_simulation(lap_time, True, optimize)

    def get_lap_time_and_reset(self) -> float:
        """
        Computes current lap time and resets timers
        :return: current laptime
        """
        current_time = rospy.get_rostime().to_sec()
        lap_time: float = current_time - self.start_time.to_sec()
        self.start_time = None
        return lap_time
        