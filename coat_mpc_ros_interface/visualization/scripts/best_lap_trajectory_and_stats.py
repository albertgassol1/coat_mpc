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

import os
import pickle
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from bagpy import bagreader


@dataclass
class Pose:
    x: np.ndarray
    y: np.ndarray


@dataclass
class Velocity:
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray


@dataclass
class Estimation:
    pose: Pose
    vel: Velocity


def read_bag(rosbag_path: str) -> Tuple[Estimation, Pose, Pose]:
    """
    Read bag and parse it to estimation object
    :param rosbag_path: path to rosbag
    :return: estimation object, track center and track boundary from rosbag data
    """

    bag = bagreader(rosbag_path)

    def parse_bag(topic: str, remove: Optional[bool] = False) -> pd.DataFrame:
        """
        Read bag and return dataframe
        :param topic: topic to read
        :param remove: falg to remove rosbag csv directory
        :return: dataframe with rosbag data
        """
        csv_file = bag.message_by_topic(topic)
        dataframe = pd.read_csv(csv_file)
        if remove:
            csv_directory = os.path.split(csv_file)[0]
            shutil.rmtree(csv_directory)
        return dataframe

    # Read estimation message
    estimation_msgs = parse_bag("/car_1/estimation_node/best_state")
    estimation = Estimation(
        Pose(np.asarray(estimation_msgs["x"]), np.asarray(estimation_msgs["y"])),
        Velocity(
            np.asarray(estimation_msgs["vx_b"]),
            np.asarray(estimation_msgs["vy_b"]),
            np.rad2deg(np.asarray(estimation_msgs["dyaw"])),
        ),
    )

    def parse_points(message: str) -> Pose:
        """
        Parse marker array message to get pose points
        :param message: message to parse
        :return: poses
        """
        res = re.split(", |\n", message.removeprefix("[").removesuffix("]"))
        array = np.asarray([float(point[3:]) for point in res])
        return Pose(array[0::3], array[1::3])

    # Read track message
    track_msg = parse_bag("/track_info", remove=True)
    track_center_index = np.where(np.asarray(track_msg["ns"] == "track_center"))[0][0]
    track_boundary_index = np.where(np.asarray(track_msg["ns"] == "track_boundary"))[0][
        0
    ]

    center_points = parse_points(track_msg["points"][track_center_index])
    boundary_points = parse_points(track_msg["points"][track_boundary_index])

    return estimation, center_points, boundary_points


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read yaml file and return it as dictionary
    :param path: path to yaml file
    :return: dictionary with parsed yaml file
    """
    with open(path) as file:
        config: Dict[str, Any] = yaml.safe_load(file)
        return config


def read_object(path: str) -> Any:
    """
    Read objects and return them
    :param path: path to object to read
    :return: object
    """
    with open(path, "rb") as object_file:
        saved_object = pickle.load(object_file)
        return saved_object


def main(
    laptimes_file: str,
    theta_file: str,
    domain_file: str,
    parameters_file: str,
    rosbag_directory: str,
) -> None:
    """
    Plot trajectory of best set of parameters and stats (velocities, acceleration, steering)
    :param laptimes_file: laptimes file
    :param theta_file: theta file
    :param domain_file: domain file
    :param parameters_file: tuned parameters
    :param rosbag_directory: directory where the rosbags are stored
    """

    # Read objects and get minimum laptime
    laptimes = read_object(laptimes_file)[:50, :]
    theta = read_object(theta_file)
    domain = read_object(domain_file)
    denormalized_theta = theta * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
    params = read_object(parameters_file)
    min_index = np.argmin(laptimes)

    # Read rosbag corresponding to minimum laptime
    estimation, track_center, track_boundary = read_bag(
        os.path.join(rosbag_directory, str(min_index) + ".bag")
    )

    # Plot car density position
    fig1, ax1 = plt.subplots()
    condition = estimation.vel.x > 1.2
    sns.kdeplot(
        x=estimation.pose.x[condition],
        y=estimation.pose.y[condition],
        cmap="Blues",
        fill=True,
        bw_adjust=0.1,
        ax=ax1,
        thresh=0.1,
    )
    ax1.plot(track_boundary.x, track_boundary.y, color="k")
    ax1.plot(track_center.x, track_center.y, ls="--", lw=2, color="red")
    ax1.set_xlabel("x-Position [m]", fontsize=14)
    ax1.set_ylabel("y-Position [m]", fontsize=14)
    fig1.suptitle(
        f"Density of the car position during \n"
        f" the fastest 4 laps. {params} = {denormalized_theta[min_index, :].tolist()}",
        fontsize=14,
    )

    # Plot velocities
    fig2, ax2 = plt.subplots(3, 1)
    time = np.linspace(0, laptimes[min_index], estimation.vel.x[condition].shape[0])
    ax2[0].plot(time, estimation.vel.x[condition], color="blue", lw=2)
    ax2[1].plot(time, estimation.vel.y[condition], color="red", lw=2)
    ax2[2].plot(time, estimation.vel.w[condition], color="green", lw=2)
    ax2[0].set_xlabel("Time [s]", fontsize=12)
    ax2[1].set_xlabel("Time [s]", fontsize=12)
    ax2[2].set_xlabel("Time [s]", fontsize=12)
    ax2[0].set_ylabel("x-Velocity [m/s]", fontsize=12)
    ax2[1].set_ylabel("y-Velocity [m/s]", fontsize=12)
    ax2[2].set_ylabel("yaw rate [deg/s]", fontsize=12)
    fig2.suptitle(
        f"Car velocities during the fastest 4 laps. '\n"
        f" {params} = {denormalized_theta[min_index, :].tolist()}",
        fontsize=14,
    )

    # Plot laps
    fig3, ax3 = plt.subplots()
    ax3.plot(
        estimation.pose.x[condition], estimation.pose.y[condition], color="blue", lw=2
    )
    ax3.plot(track_boundary.x, track_boundary.y, color="k")
    ax3.plot(track_center.x, track_center.y, ls="--", lw=2, color="red")
    ax3.set_xlabel("x-Position [m]", fontsize=14)
    ax3.set_ylabel("y-Position [m]", fontsize=14)
    fig3.suptitle(
        f"Fastest 4 laps. '\n"
        f" {params} = {denormalized_theta[min_index, :].tolist()}",
        fontsize=14,
    )

    plt.show()


if __name__ == "__main__":
    config = read_yaml(os.getcwd() + "/config/params.yaml")

    laptime_file = os.path.join(os.getcwd(), "objects", config["dir"], "laptimes.np")
    theta_file = os.path.join(os.getcwd(), "objects", config["dir"], "theta.np")
    domain_file = os.path.join(os.getcwd(), "objects", config["dir"], "domain.np")

    rosbags = os.path.join(os.getcwd(), "objects", config["dir"], "rosbags")
    parameters_file = os.path.join(
        os.getcwd(), "objects/", config["dir"], "parameters.lst"
    )

    main(laptime_file, theta_file, domain_file, parameters_file, rosbags)
