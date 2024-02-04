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
import fire
import json
import pickle
from typing import Any, Dict, List
import yaml
import matplotlib
matplotlib.use('pgf')
import numpy as np

import matplotlib.pyplot as plt
import plotting_utilities
from gp_visualization import visualization_gpy
from test_safeopt import run_safe_opt



def read_object(path: str) -> Any:
    """
    Read objects and return them
    :param path: path to the object
    :return: the object
    """
    # Read objects and return them
    with open(path, 'rb') as object_file:
        saved_object = pickle.load(object_file)
        return saved_object


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read a yaml file and return the dictionary
    :param path: path to the yaml file
    :return: the dictionary
    """
    with open(path) as file:
        return yaml.safe_load(file)


def denormalize(input_vector: np.ndarray,
                domain: np.ndarray) -> np.ndarray:
    """
    Denormalize input vector from [0, 1] to its domain
    :param input_vector: input vector to denormalize
    :param domain: domain of input vector
    :return: denormalized vector
    """

    return input_vector * (domain[:, 1] - domain[:, 0]) + domain[:, 0]


def stats_of_arrays(list_of_arrays: List[np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    Compute the mean and standard deviation of a list of arrays
    :param list_of_arrays: list of arrays
    :return: mean and standard deviation
    """
    max_len = max(map(len, list_of_arrays))
    accumulator = np.zeros(max_len)
    squared_accumulator = np.zeros(max_len)
    count = np.zeros(max_len)

    # Accumulating sums
    for arr in list_of_arrays:
        for i, val in enumerate(arr):
            accumulator[i] += val
            count[i] += 1

    # Using the accumulated sums to find means
    means = accumulator / count

    # Accumulating squared differences for variance
    for arr in list_of_arrays:
        for i, val in enumerate(arr):
            squared_accumulator[i] += (val - means[i]) ** 2

    # Computing variance for each position
    variances = squared_accumulator / count

    # Taking square root of variance for standard deviation
    std_devs = np.sqrt(variances)

    # Where count is 0, we set means and std_devs to 0 or any other desired value
    means[np.isnan(means)] = 0
    std_devs[np.isnan(std_devs)] = 0

    return means, std_devs


def find_min_value_and_index(list_of_arrays):
    # Initialize the minimum value with a very large number
    min_val = float('inf')
    min_list_idx = None
    min_array_idx = None

    # Iterate through each array in the list and update the minimum value and its indices
    for list_idx, arr in enumerate(list_of_arrays):
        current_min = np.min(arr)
        if current_min < min_val:
            min_val = current_min
            min_list_idx = list_idx
            min_array_idx = np.argmin(arr)  # Gets the index of the minimum value within the array

    return min_val, min_list_idx, min_array_idx


def main(plt_samples: bool = True, iterations: int = 70) -> None:
    """
    Main function
    :param plt_samples: plot the samples or not
    :param iterations: number of iterations to plot
    """
    iterations = int(iterations)
    # Read the configuration file
    config = read_yaml(os.path.join(os.getcwd(), 'config', 'params.yaml'))

    # Get data folders
    folders_directory = os.path.join(os.getcwd(), 'objects', config['dir'])
    folders = ['goose']
               #,'safeopt_original', 'ucb', 'wml', 'eic', 'crbo']

    # Iterate over the folders
    figs = list()
    axs = list()
    title = {'ucb': 'GP-UCB', 'goose': r'\textsc{COAt-MPC}',
             'safeopt_original': r'\textsc{Safe-Opt}', 'eic': r'EI\textsubscript{C}',
             'crbo': 'CRBO', 'wml': 'WML'}
    plotting_utilities.set_figure_params(fontsize=22)

    fig_cum_reg = plt.figure()
    ax_cum_reg = fig_cum_reg.add_subplot(111)
    fig_cum_reg.set_size_inches(10, 7)

    fig_laptime = plt.figure()
    ax_laptime = fig_laptime.add_subplot(111)
    fig_laptime.set_size_inches(10, 7)

    constraint_time = None
    mean_constraint_violations = list()
    best_laptimes = list()
    best_parameters = list()
    overall_mean_laptimes = list()
    overall_std_laptimes = list()
    plotted = False

    goose_laptimes = list()
    goose_cum_regret = list()

    for directory in folders:
        # Get subfolders
        subfolders = os.listdir(os.path.join(folders_directory, directory))
        # Laptimes and cumulative regret
        laptimes = None
        cum_regret = None
        constraint_violations = list()

        for i, subfolder in enumerate(subfolders):
            # Get the files
            theta_file = os.path.join(folders_directory, directory, subfolder, 'theta.np')
            laptime_file = os.path.join(folders_directory, directory, subfolder, 'laptimes.np')
            domain_file = os.path.join(folders_directory, directory, subfolder, 'domain.np')
            patameters_file = os.path.join(folders_directory, directory, subfolder, 'parameters.lst')

            # Read the configuration file
            safe_opt_config = read_yaml(os.path.join(folders_directory, directory, subfolder, 'config.yaml'))
            if plt_samples:
                if safe_opt_config['optimization_config']['acquisition_function']:
                    fig, ax = visualization_gpy(theta_file, laptime_file, domain_file,
                                                patameters_file, safe_opt_config, iterations)

                    ax.set_title(title[directory])

                else:
                    normalized_theta_file = os.path.join(folders_directory, directory,
                                                         subfolder, 'normalized_laptimes.np')
                    fig, ax = run_safe_opt(theta_file, normalized_theta_file, domain_file,
                                           patameters_file, safe_opt_config)

                figs.append(fig)
                axs.append(ax)
                save_path = os.path.join(folders_directory, directory, subfolder)
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(os.path.join(save_path, f"{iterations}.png"),
                            format='png', dpi=100, bbox_inches='tight')
                plt.close(fig)

            # Save cumulative regret and laptimes to later compute mean and std
            optimal_time = safe_opt_config['interface_config']['optimal_time'] / 4
            if directory == 'goose':
                laptime = read_object(laptime_file)[:iterations]/4
                # laptime[5:] += 0.25
                # laptime[laptime==10.25] -= 0.25
                iters = np.linspace(1.0, laptime.shape[0], laptime.shape[0])
                goose_cum_regret.append((np.cumsum(laptime - optimal_time) / iters)[:, np.newaxis])
                goose_laptimes.append(laptime)
                # Compute constraint violations
                constraint_time = goose_laptimes[i][0] * safe_opt_config['optimization_config']['first_lap_multiplier']
                constraint_violations.append(np.sum(goose_laptimes[i] > constraint_time))
            else:
                if laptimes is None:
                    laptimes = read_object(laptime_file)[:iterations]/4
                    iters = np.linspace(1.0, laptimes.shape[0], laptimes.shape[0])
                    cum_regret = (np.cumsum(laptimes - optimal_time) / iters)[:iterations, np.newaxis]
                else:
                    laptimes = np.concatenate((laptimes, read_object(laptime_file)[:iterations]/4), axis=1)
                    cum_regret = np.concatenate((cum_regret,
                                                 (np.cumsum(laptimes[:, i] - optimal_time) / iters)[:, np.newaxis]), axis=1)
                # Compute constraint violations
                constraint_time = laptimes[0, i] * safe_opt_config['optimization_config']['first_lap_multiplier']
                constraint_violations.append(np.sum(laptimes[:, i] > constraint_time))

        if not plotted:
            # Plot constraint
            constraint_time = goose_laptimes[0][0] * safe_opt_config['optimization_config']['first_lap_multiplier']
            ax_laptime.plot(np.ones(iterations) * constraint_time, linestyle='dashed',
                            color='k', label='Constraint', linewidth=2, zorder=1)
            plotted = True

        # Compute mean and std
        if directory == "goose":
            mean_laptimes, std_laptimes = stats_of_arrays(goose_laptimes)
            mean_cum_regret, std_cum_regret = stats_of_arrays(goose_cum_regret)
        else:
            mean_laptimes = np.mean(laptimes, axis=1)
            std_laptimes = np.std(laptimes, axis=1)
            mean_cum_regret = np.mean(cum_regret, axis=1)
            std_cum_regret = np.std(cum_regret, axis=1)

        # Plot mean and std
        ax_cum_reg.plot(mean_cum_regret, label=title[directory], linewidth=3)
        ax_cum_reg.fill_between(np.arange(mean_cum_regret.shape[0]), mean_cum_regret - std_cum_regret,
                                mean_cum_regret + std_cum_regret, alpha=0.08, zorder=-1)
        ax_laptime.plot(mean_laptimes, label=title[directory], linewidth=3, zorder=-1)
        ax_laptime.fill_between(np.arange(mean_laptimes.shape[0]), mean_laptimes - std_laptimes,
                                mean_laptimes + std_laptimes, alpha=0.08, zorder=-1)

        # Save mean constraint violations and best laptimes
        mean_constraint_violations.append(np.mean(np.asarray(constraint_violations)))
        if directory == "goose":
            best_laptime, subfolder_index, timestep = find_min_value_and_index(goose_laptimes)
            best_laptimes.append(best_laptime)
            best_parameters.append(read_object(os.path.join(folders_directory, directory, subfolders[subfolder_index],
                                                            'theta.np'))[timestep, :])
        else:
            timestep, subfolder_index = np.unravel_index(np.argmin(laptimes, axis=None), laptimes.shape)
            best_laptimes.append(laptimes[timestep, subfolder_index])
            best_parameters.append(read_object(os.path.join(folders_directory, directory, subfolders[subfolder_index],
                                                            'theta.np'))[timestep, :])

        # Compute mean and std of laptimes
        if directory == "goose":
            laptimes = np.concatenate(goose_laptimes, axis=0)
        overall_mean_laptimes.append(np.mean(laptimes, axis=None))
        overall_std_laptimes.append(np.std(laptimes, axis=None))

    ax_cum_reg.set_xlabel("Iterations")
    ax_cum_reg.set_ylabel("Cumulative regret")
    ax_cum_reg.set_title(r'Cumulative regret over the iterations')
    handles, labels = ax_cum_reg.get_legend_handles_labels()

    # Create dashed line for constraint label
    constraint_line = matplotlib.lines.Line2D([], [], color='k', linestyle='dashed', label='Constraint', linewidth=2)
    # handles.append(constraint_line)
    ax_cum_reg.legend(handles=handles, fontsize=20, loc='upper right', ncol=2)
    ax_cum_reg.set_xlim([0, iterations])

    ax_laptime.set_xlabel("Iterations")
    ax_laptime.set_ylabel("Laptime [s]")
    ax_laptime.set_title("Laptime over the iterations")
    handles, labels = ax_laptime.get_legend_handles_labels()
    handles.append(constraint_line)
    ax_laptime.legend(handles=handles, fontsize=20, loc='upper right', ncol=2)
    ax_laptime.set_xlim([0, iterations])
    axs = [ax_cum_reg, ax_laptime]

    # Plot the figures
    plotting_utilities.adapt_figure_size_from_axes(axs)
    # plt.tight_layout()
    fig_laptime.set_tight_layout(True)
    fig_cum_reg.set_tight_layout(True)
    # plt.show()

    if 'safeopt' in folders:
        folders.remove('safeopt')
    if plt_samples:
        i = 0
        for folder in folders:
            subfolders = os.listdir(os.path.join(folders_directory, directory))
            for subfolder in subfolders:
                figs[i].savefig(os.path.join(folders_directory, f'{folder}_{subfolder}.pdf'),
                                format='pdf', dpi=100, bbox_inches='tight')
                i += 1

    fig_cum_reg.savefig(os.path.join(folders_directory, 'cum_regret.pdf'), format='pdf', dpi=100, bbox_inches='tight')
    fig_laptime.savefig(os.path.join(folders_directory, 'laptime.pdf'), format='pdf', dpi=100, bbox_inches='tight')

    # Print and save best laptimes and parameters. Denormalize the parameters
    print('Best laptimes and parameters:')
    denormalized_best_parameters = list()
    for i, (folder, best_laptime) in enumerate(zip(folders, best_laptimes)):
        print(f'Algorithm: {folder}')
        print(f'\tBest laptime: {best_laptime}')
        denormalized_best_parameter = denormalize(best_parameters[i], read_object(domain_file))
        denormalized_best_parameters.append(denormalized_best_parameter.tolist())
        print(f'\tBest parameter: {denormalized_best_parameter}')
    # Save best laptimes and parameters. Save it as a dictionary with the keys being the algorithm names
    best_laptimes_and_parameters = dict(zip(folders, zip(best_laptimes, denormalized_best_parameters)))
    # Save as json
    with open(os.path.join(folders_directory, 'best_laptimes_and_parameters.json'), 'w') as f:
        json.dump(best_laptimes_and_parameters, f, indent=4)

    # Print mean constraint violations
    print('Mean constraint violations:')
    for i, (folder, mean_constraint_violation) in enumerate(zip(folders, mean_constraint_violations)):
        print(f'Algorithm: {folder}')
        print(f'\tMean constraint violations: {mean_constraint_violation}')
    # Save mean constraint violations. Save it as a dictionary with the keys being the algorithm names
    mean_constraint_violations = dict(zip(folders, mean_constraint_violations))
    # Save as json
    with open(os.path.join(folders_directory, 'mean_constraint_violations.json'), 'w') as f:
        json.dump(mean_constraint_violations, f, indent=4)

    # Print mean and std of laptimes
    print('Mean and std of laptimes:')
    for i, (folder, mean_laptime, std_laptime) in enumerate(zip(folders, overall_mean_laptimes, overall_std_laptimes)):
        print(f'Algorithm: {folder}')
        print(f'\tMean laptime: {mean_laptime}')
        print(f'\tStd laptime: {std_laptime}')
    # Save mean and std of laptimes. Save it as a dictionary with the keys being the algorithm names
    mean_and_std_laptimes = dict(zip(folders, zip(overall_mean_laptimes, overall_std_laptimes)))
    # Save as json
    with open(os.path.join(folders_directory, 'mean_and_std_laptimes.json'), 'w') as f:
        json.dump(mean_and_std_laptimes, f, indent=4)


if __name__ == '__main__':
    fire.Fire(main)
