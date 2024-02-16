#!/usr/bin/env python3
# pylint: skip-file
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
import pickle
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import colors


def read_object(path: str):
    # Read objects and return them
    with open(path, "rb") as object_file:
        saved_object = pickle.load(object_file)
        return saved_object


def read_yaml(path: str) -> Dict:
    with open(path) as file:
        return yaml.safe_load(file)


def run_regret(laptime_path: str, parameters_path: str, config: Dict[str, Any]) -> None:
    # Get objects
    laptimes = read_object(laptime_path)
    params = read_object(parameters_path)

    # Compute laptime cumulative regret
    iters = np.linspace(1.0, laptimes.shape[0], laptimes.shape[0])
    optimal_time = config["interface_config"]["optimal_time"]
    cum_regret = np.cumsum(laptimes - optimal_time) / iters
    inst_regret = laptimes - optimal_time
    constraint_time = (
        laptimes[0] * config["optimization_config"]["first_lap_multiplier"]
    )

    plt.rcParams["figure.figsize"] = (6, 5)
    fig1, axs1 = plt.subplots()
    axs1.plot(cum_regret, linewidth=2)
    axs1.set_xlabel("$iters$")
    axs1.set_ylabel("Cumulative regret over time")
    axs1.set_title(f"Cumulative regret over time using params {params}", fontsize=14)

    fig2, axs2 = plt.subplots()
    axs2.plot(inst_regret)
    axs2.set_xlabel("$iters$")
    axs2.set_ylabel("Instant regret")
    axs2.set_title(f"Instant regret using params {params}", fontsize=14)

    fig3, axs3 = plt.subplots()
    axs3.plot(laptimes, label="Laptime")
    axs3.plot(
        np.ones(len(laptimes)) * constraint_time,
        linestyle="dashed",
        color="k",
        label="Constraint",
    )
    axs3.set_xlabel("$iters$")
    axs3.set_ylabel("Laptime")
    axs3.set_title(f"Laptime using params {params}", fontsize=14)
    axs3.legend()

    fig4, axs4 = plt.subplots()
    axs4.plot(np.cumsum(laptimes > constraint_time))
    axs4.set_xlabel("$iters$")
    axs4.set_ylabel("Laptime constraint violations")
    axs4.set_title("Violations")

    fig5, axs5 = plt.subplots()
    axs5.plot(np.cumsum(laptimes >= config["interface_config"]["max_time"]))
    axs5.set_xlabel("$iters$")
    axs5.set_ylabel("Laptime edge case violations")
    axs5.set_title("Edge case violations")

    real_laptimes = laptimes.copy()
    n_hard_violations = (real_laptimes == config["interface_config"]["max_time"]).sum()
    real_laptimes = real_laptimes[
        real_laptimes != config["interface_config"]["max_time"]
    ]
    fig6, axs6 = plt.subplots()
    axs6.plot(np.cumsum(real_laptimes))
    axs6.set_xlabel("$iters$")
    axs6.set_ylabel("Cumulative time spent")
    axs6.set_title("Cumulative time spent")

    if np.any(laptimes > constraint_time):
        fig7, axs7 = plt.subplots()
        # N is the count in each bin, bins is the lower-limit of the bin
        N, bins, patches = axs7.hist(laptimes[laptimes > constraint_time], bins=50)
        # We'll color code by height, but you could use any scalar
        fracs = N / N.max()
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.seismic(norm(thisfrac))
            thispatch.set_facecolor(color)
        axs7.set_xlabel("laptime")
        axs7.set_ylabel("Number of times")
        axs7.set_title("Histogram of constraint violation times")

    total_time = np.cumsum(real_laptimes)[-1]
    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        f"Number of optimization iterations: {len(real_laptimes)} \n"
        f"Amount of time spent throughout the optimization: {hours} hours, {minutes} minutes,"
        f"{seconds} seconds \n"
        f"Number of hard violations: {n_hard_violations} \n"
        f"Average time per iteration: {total_time/len(real_laptimes)} seconds"
    )

    plt.show()


if __name__ == "__main__":
    config = read_yaml(os.getcwd() + "/config/params.yaml")

    laptime_file = os.getcwd() + "/objects/" + config["dir"] + "/laptimes.np"
    patameters_file = os.getcwd() + "/objects/" + config["dir"] + "/parameters.lst"

    config = read_yaml(
        os.path.join(os.getcwd(), "objects", config["dir"], "config.yaml")
    )

    run_regret(laptime_file, patameters_file, config)
