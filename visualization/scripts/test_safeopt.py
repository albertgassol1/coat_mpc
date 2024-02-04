#!/usr/bin/env python3
# pylint: skip-file
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

import os
import pickle
from itertools import product
from typing import Any, Dict, Tuple

import GPy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from botorch.utils import standardize
from gp_opt import SafeOpt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
# Libraries
from matplotlib.patches import Patch
from utilities import linearly_spaced_combinations


def read_object(path: str):
    # Read objects and return them
    with open(path, 'rb') as object_file:
        saved_object = pickle.load(object_file)
        return saved_object


def run_safe_opt(theta_path: str, laptime_path: str, domain_path: str, parameters_path: str,
                 opt_config: Dict[str, Any]) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Run safe optimization
    :param theta_path: path to the theta values
    :param laptime_path: path to the laptime values
    :param domain_path: path to the domain
    :param parameters_path: path to the parameters
    :param opt_config: configuration of the optimization
    :return: the figure and axes
    """

    # Get objects
    opt_laptime = opt_config['interface_config']['optimal_time']
    penalty_time = opt_config['interface_config']['max_time']
    opt_config = opt_config['optimization_config']
    domain = read_object(domain_path)
    theta_norm = read_object(theta_path)
    theta = (theta_norm) * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
    theta = theta_norm

    raw_laptimes = read_object(laptime_path)
    laptimes = raw_laptimes * opt_laptime
    params = read_object(parameters_path)

    # Define kernel and function
    kernel = GPy.kern.Matern52(input_dim=domain.shape[0],
                               ARD=False,
                               lengthscale=opt_config['kernel_lengthscale'],
                               variance=opt_config['kernel_variance'])
    # The statistical model of our objective function
    print(theta)
    print(-laptimes)
    mf = GPy.core.Mapping(domain.shape[0], 1)
    initial = laptimes[0][0]

    mf.f = lambda x: -(opt_config['first_lap_multiplier'] - 0.02) * initial
    # mf.f = lambda x: -penalty_time
    mf.update_gradients = lambda a, b: 0
    mf.gradients_X = lambda a, b: 0
    gp = GPy.models.GPRegression(theta[:2, :], -laptimes[:2, :], kernel,
                                 noise_var=opt_config['gp_variance']**2,
                                 mean_function=mf)

    colormesh = None
    ax2d = None
    use_ucb = False
    if domain.shape[0] == 2:

        parameter_set = linearly_spaced_combinations(bounds=[(0, 1), (0, 1)],
                                                     num_samples=opt_config['grid_size'])
        if 'use_ucb' in opt_config.keys():
            opt = SafeOpt(gp, parameter_set, -opt_config['first_lap_multiplier'] * initial,
                          beta=opt_config['beta'],
                          lipschitz=opt_config['lipschitz_constant'],
                          minimum_variance=opt_config['minimum_variance'])
            use_ucb = opt_config['use_ucb']
        else:
            opt = SafeOpt(gp, parameter_set, -opt_config['first_lap_multiplier'] * initial,
                          beta=opt_config['beta'],
                          lipschitz=opt_config['lipschitz_constant'])

        for i in range(2, len(laptimes)):
            candidate = opt.optimize(ucb=use_ucb)
            print(candidate)
            opt.add_new_data_point(theta[i, :], -laptimes[i, :])
            if i == len(laptimes) - 1:
                candidate = opt.optimize(ucb=use_ucb)
                gp1 = GPy.models.GPRegression(theta[:(i + 1), :], -laptimes[:(i + 1), :], kernel,
                                              noise_var=opt_config['gp_variance']**2,
                                              mean_function=mf)

                safe_set_indices = np.where(opt.S)[0]
                safe_set_z = np.zeros(opt_config['grid_size'] * opt_config['grid_size'])
                safe_set_z[safe_set_indices] = 1
                safe_set_z = safe_set_z.reshape(opt_config['grid_size'], opt_config['grid_size'])
                safe_set_x, safe_set_y = \
                    parameter_set[:, 0].reshape(opt_config['grid_size'], opt_config['grid_size']), \
                    parameter_set[:, 1].reshape(opt_config['grid_size'], opt_config['grid_size'])
                x1 = np.linspace(0, 1, opt_config['grid_size'])
                x2 = np.linspace(0, 1, opt_config['grid_size'])
                x1x2 = np.array(list(product(x1, x2)))
                y_pred, cov_pred = gp1.predict(x1x2)
                X0p, X1p = x1x2[:, 0].reshape(opt_config['grid_size'], opt_config['grid_size']), \
                    x1x2[:, 1].reshape(opt_config['grid_size'], opt_config['grid_size'])
                Zp = np.reshape(y_pred, (opt_config['grid_size'], opt_config['grid_size']))
                fig, ax2d = plt.subplots()

                colormesh = ax2d.pcolormesh(X0p, X1p, Zp, cmap='seismic', vmin=-35, vmax=-18.1, label='GP mean')
                ax2d.pcolormesh(safe_set_x, safe_set_y, safe_set_z,
                                cmap=ListedColormap([[1, 1, 1, 0], [0, 1, 1, 0.1]]))

                ax2d.plot(theta[:, 0], theta[:, 1], linestyle=':', c='k', linewidth=1.5,
                          label='samples trajectory')
                ax2d.scatter(theta[1:-1, 0], theta[1:-1, 1], marker='x', c='k', label='samples')
                ax2d.scatter(theta[0, 0], theta[0, 1], marker='x', c='yellow',
                             label='initial sample')
                ax2d.scatter(theta[-1, 0], theta[-1, 1], marker='x', c='purple',
                             label='last sample')
                ax2d.scatter(candidate[0], candidate[1], marker='x', c='g',
                             label='safeopt candidate')

                custom_lines = [Patch(facecolor=[0, 1, 1, 0.1], edgecolor=[0, 1, 1, 0.1]),
                                Line2D([0], [0], color='k', linestyle=':'),
                                Line2D([0], [0], color=[0, 1, 1, 0], marker='x',
                                       markerfacecolor='k', markeredgecolor='k'),
                                Line2D([0], [0], color=[0, 1, 1, 0], marker='x',
                                       markerfacecolor='yellow', markeredgecolor='yellow'),
                                Line2D([0], [0], color=[0, 1, 1, 0], marker='x',
                                       markerfacecolor='purple', markeredgecolor='purple'),
                                Line2D([0], [0], color=[0, 1, 1, 0], marker='x',
                                       markerfacecolor='g', markeredgecolor='g')]
                ax2d.legend(custom_lines, ['Safe set', 'Samples trajetory',
                                           'Samples', 'Initial sample', 'Last sample',
                                           'Safeopt candidate'],
                            loc='upper center', bbox_to_anchor=(0.5, -0.05),
                            fancybox=True, shadow=True, ncol=3)
                ax2d.set_xlim(0, 1)
                ax2d.set_ylim(0, 1)
                plt.colorbar(colormesh, ax=ax2d, label='Laptime GP mean')

    else:

        parameter_set = linearly_spaced_combinations(bounds=[(0, 1)],
                                                     num_samples=opt_config['grid_size'])
        if 'use_ucb' in opt_config.keys():

            opt = SafeOpt(gp, parameter_set, -opt_config['first_lap_multiplier'] * initial,
                          beta=opt_config['beta'],
                          lipschitz=opt_config['lipschitz_constant'],
                          minimum_variance=opt_config['minimum_variance'])
            use_ucb = opt_config['use_ucb']
        else:
            opt = SafeOpt(gp, parameter_set, -opt_config['first_lap_multiplier'] * initial,
                          beta=opt_config['beta'],
                          lipschitz=opt_config['lipschitz_constant'])

        for i in range(2, len(laptimes)):

            candidate = opt.optimize(ucb=use_ucb)
            print(candidate)
            opt.add_new_data_point(theta[i, :], -laptimes[i, :])
            if i == len(laptimes) - 1:
                candidate = opt.optimize(ucb=use_ucb)
                gp1 = GPy.models.GPRegression(theta[:(i + 1)], -laptimes[:(i + 1), :], kernel,
                                              noise_var=opt_config['gp_variance']**2,
                                              mean_function=mf)
                safe_set = opt.S.copy().astype(int) * -opt_config['first_lap_multiplier'] * initial
                safe_set[safe_set == 0] = -14
                fig, ax = plt.subplots()
                gp1.plot(plot_density=True, ax=ax, plot_limits=[0, 1])
                ax.plot(parameter_set, safe_set, color='g', label='Safe set')
                x_values = np.linspace(0, 1, opt_config['grid_size'])
                y_values = -opt_config['first_lap_multiplier'] * initial * \
                    np.ones(opt_config['grid_size'])
                ax.plot(x_values, y_values, linestyle='dashed', color='k')
                ax.set_xlim(0, 1)
                ax.set_ylim(-32.5, -17.5)
                ax.legend()

    fig2, axs2 = plt.subplots(1, 1)
    axs2.plot(np.cumsum(laptimes > opt_config['first_lap_multiplier'] * initial))
    axs2.set_xlabel("$iters$")
    axs2.set_ylabel("$Laptime constraint violations$")
    axs2.set_title("Violations")
    # if colormesh is not None and ax2d is not None:
    #     plt.colorbar(colormesh, ax=ax2d, label='Laptime GP mean')

    plt.show()
    if fig is not None:
        return fig, ax2d
    plt.show()

def read_yaml(path: str):
    with open(path) as file:
        return yaml.safe_load(file)


if __name__ == '__main__':

    config = read_yaml(os.getcwd() + '/config/params.yaml')

    theta_file = os.getcwd() + '/objects/' + config['dir'] + '/theta.np'
    laptime_file = os.getcwd() + '/objects/' + config['dir'] + '/normalized_laptimes.np'
    constraints_file = os.getcwd() + '/objects/' + config['dir'] + '/theta.np'
    domain_file = os.getcwd() + '/objects/' + config['dir'] + '/domain.np'
    patameters_file = os.getcwd() + '/objects/' + config['dir'] + '/parameters.lst'

    safe_opt_config = read_yaml(os.path.join(os.getcwd(),
                                             'objects',
                                             config['dir'],
                                             'config.yaml'))

    run_safe_opt(theta_file, laptime_file, domain_file, patameters_file, safe_opt_config)
