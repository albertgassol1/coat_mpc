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

import numpy as np
import pickle
import os
import yaml
from typing import Any, Dict, Tuple, Optional


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read a yaml file
    :param path: path to the yaml file
    :return: dictionary with the yaml file content
    """
    with open(path) as file:
        return yaml.safe_load(file)


def read_np(path: str) -> np.ndarray:
    """
    Read a np file
    :param path: path to the np file
    :return: np array
    """
    return np.load(path)


def load_pickle(path: str) -> Any:
    """
    Load a pickle file
    :param path: path to the pickle file
    :return: object
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def dump_pickle(path: str, obj: Any) -> None:
    """
    Dump an object to a pickle file
    :param path: path to the pickle file
    :param obj: object to be dumped
    """
    with open(path, 'wb') as file:
        pickle.dump(obj, file)


def add_penalty(theta: np.ndarray, laptimes: np.ndarray,
                normalized_laptimes: Optional[np.ndarray], current_theta: np.ndarray,
                opt_laptime: float, penalty_time: float,
                number_of_laps: int) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Add penalty to the theta matrix
    :param theta: theta matrix
    :param laptimes: laptimes matrix
    :param normalized_laptimes: normalized laptimes matrix
    :param current_theta: current theta vector
    :param opt_laptime: optimal laptime
    :param penalty_time: penalty time
    :param number_of_laps: number of laps
    :return: theta, laptimes, normalized_laptimes
    """

    # Add current theta to the theta matrix
    theta = np.vstack((theta, current_theta))

    # Add penalty to the laptimes matrix
    laptimes = np.vstack((laptimes, penalty_time))

    # Add penalty to the normalized laptimes matrix
    if normalized_laptimes is not None:
        normalized_laptimes = np.vstack((normalized_laptimes, penalty_time / opt_laptime))

    return theta, laptimes, normalized_laptimes


if __name__ == '__main__':

    # Read parameters
    directory = os.getcwd()
    directory = directory[:-len("scripts")] + 'visualization/'
    config = read_yaml(directory + '/config/params.yaml')
    safe_opt_config = read_yaml(os.path.join(directory,
                                             'objects',
                                             config['dir'],
                                             'config.yaml'))
    opt_laptime = safe_opt_config['interface_config']['optimal_time']
    penalty_time = safe_opt_config['interface_config']['max_time']
    number_of_laps = safe_opt_config['interface_config']['number_of_laps']
    theta_file = directory + '/objects/' + config['dir'] + '/theta.np'
    laptimes_file = directory + '/objects/' + config['dir'] + '/laptimes.np'
    normalized_laptimes_file = directory + '/objects/' + config['dir'] + '/normalized_laptimes.np'
    current_theta_file = directory + '/objects/' + config['dir'] + '/current_theta.np.npy'

    # Load objects
    theta = load_pickle(theta_file)
    laptimes = load_pickle(laptimes_file)
    if os.path.exists(normalized_laptimes_file):
        normalized_laptimes = load_pickle(normalized_laptimes_file)
    else:
        normalized_laptimes = None
    current_theta = read_np(current_theta_file)

    # Add penalty
    theta, laptimes, normalized_laptimes = add_penalty(theta, laptimes, normalized_laptimes, current_theta,
                                                       opt_laptime, penalty_time, number_of_laps)

    # Dump objects
    dump_pickle(theta_file, theta)
    dump_pickle(laptimes_file, laptimes)
    if os.path.exists(normalized_laptimes_file):
        dump_pickle(normalized_laptimes_file, normalized_laptimes)

