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
from itertools import product
from typing import Any, Dict, List, Tuple

import GPy
import matplotlib

# Visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# ML libraries
import torch
import yaml
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means import ConstantMean
import plotting_utilities

NUM_SAMPLES = 1000


def read_object(path: str):
    # Read objects and return them
    with open(path, "rb") as object_file:
        saved_object = pickle.load(object_file)
        return saved_object


def read_yaml(path: str):
    with open(path) as file:
        return yaml.safe_load(file)


def plot_2d_gp(
    x_samples: np.ndarray,
    y_samples: np.ndarray,
    x_points: np.ndarray,
    y_points: np.ndarray,
    predicted_laptimes: np.ndarray,
    params: List[str],
    laptimes: np.ndarray,
    plot_3d=False,
):
    names = {"Q1": r"$Q_{contour}$", "Q2": r"$Q_{lag}$", "q": r"$q$"}
    # Plot GP surface (laptime wrt 2 parameters)
    if plot_3d:
        fig1 = plt.figure()
        ax = fig1.add_subplot(projection="3d")
        ax.plot_surface(
            x_points,
            y_points,
            predicted_laptimes,
            rstride=1,
            cstride=1,
            cmap="jet",
            linewidth=0,
            antialiased=False,
        )
        ax.set_title("Surrogate model mean and samples")
        ax.set_xlabel(f"{params[0]} weight")
        ax.set_ylabel(f"{params[1]} weight")
        ax.set_zlabel("Normalized and standarized laptime")
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    fig3.set_size_inches(9.5, 8)

    ax3.set_title("Heatmap of the GP")
    if params[0] in names.keys():
        ax3.set_xlabel(names[params[0]])
    else:
        ax3.set_xlabel(f"{params[0]}")
    if params[1] in names.keys():
        ax3.set_ylabel(names[params[1]])
    else:
        ax3.set_ylabel(f"{params[1]}")

    # pc = ax3.pcolormesh(x_points, y_points, predicted_laptimes/4, cmap='viridis',
    #                     vmin=-29/4, vmax=-18.1/4, alpha=0.5, rasterized=True)
    pc = ax3.pcolormesh(
        x_points,
        y_points,
        predicted_laptimes / 4,
        cmap="viridis",
        vmin=-7,
        vmax=-11,
        alpha=0.5,
        rasterized=True,
    )
    # pc = ax3.pcolormesh(x_points, y_points, predicted_laptimes / 4, cmap='viridis',
    #                     alpha=0.5, rasterized=True)
    # pc = ax3.pcolormesh(x_points, y_points, predicted_laptimes / 4, cmap='viridis', alpha=0.5, rasterized=True)
    ax3.contour(
        x_points,
        y_points,
        predicted_laptimes / 4,
        levels=15,
        cmap="viridis",
        alpha=1.0,
        linewidths=3.5,
        zorder=-1,
    )
    ax3.plot(
        x_samples,
        y_samples,
        linestyle=":",
        c="k",
        linewidth=plotting_utilities.linewidth_in_data_units(0.003, ax3),
        label="samples trajectory",
        alpha=0.7,
        rasterized=True,
    )
    ax3.scatter(
        x_samples[1:-1],
        y_samples[1:-1],
        marker="X",
        c="k",
        label="samples",
        s=plotting_utilities.linewidth_in_data_units(0.3, ax3),
        alpha=0.8,
        zorder=1,
    )
    ax3.scatter(
        x_samples[0],
        y_samples[0],
        marker="X",
        c="yellow",
        label="initial sample",
        s=plotting_utilities.linewidth_in_data_units(0.3, ax3),
        rasterized=True,
        zorder=1,
    )
    ax3.scatter(
        x_samples[-1],
        y_samples[-1],
        marker="X",
        c="purple",
        label="last sample",
        s=plotting_utilities.linewidth_in_data_units(0.3, ax3),
        rasterized=True,
        zorder=1,
    )
    argmax_idx = np.argmax(laptimes)
    ax3.scatter(
        x_samples[argmax_idx],
        y_samples[argmax_idx],
        marker="X",
        c="red",
        label="optimal sample",
        s=plotting_utilities.linewidth_in_data_units(0.3, ax3),
        rasterized=True,
        zorder=1,
    )
    # ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #            fancybox=True, shadow=True, ncol=3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    plt.colorbar(pc)

    return fig3, ax3


def visualization(
    theta_path: str,
    laptime_path: str,
    domain_path: str,
    parameters_path: str,
    config: Dict[str, Any],
):
    # Get objects
    theta = read_object(theta_path)
    normalized_laptimes = read_object(laptime_path)
    params = read_object(parameters_path)
    domain = read_object(domain_path)
    denormalized_theta = theta * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
    opt_config = config["optimization_config"]
    optimal_time = config["interface_config"]["optimal_time"]
    # Fit GP
    torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt_config["acquisition_function"] == "EIC":
        laptimes = -torch.from_numpy(normalized_laptimes)
    else:
        laptimes = -standardize(torch.from_numpy(normalized_laptimes))

    constant_mean = ConstantMean()
    constant_mean.initialize(
        constant=-float(
            (opt_config["first_lap_multiplier"] - 0.02) * normalized_laptimes[0]
        )
    )
    objective_model = SingleTaskGP(
        torch.from_numpy(theta),
        -torch.from_numpy(normalized_laptimes),
        mean_module=constant_mean,
    )

    if opt_config["constant_lengthscale"]:
        objective_model.covar_module.base_kernel.lengthscale = (
            torch.ones((1, theta.shape[1]), dtype=torch.float64)
            * opt_config["kernel_lengthscale"]
        )
    else:
        mll = ExactMarginalLogLikelihood(objective_model.likelihood, objective_model)
        fit_gpytorch_model(mll)

    # Generate domains
    domains = []
    for i in range(2):
        domains.append(np.linspace(0, 1, NUM_SAMPLES))

    # Generate points to predict data
    X = torch.from_numpy(np.array(list(product(domains[0], domains[1]))))

    # Predict using the model
    with torch.no_grad():
        posterior = objective_model.posterior(X)
        # Get upper and lower confidence bounds (2 standard deviations from the mean)
        lower, upper = posterior.mvn.confidence_region()
        predicted_laptimes = posterior.mean.cpu().numpy() * optimal_time

    predicted_laptimes = predicted_laptimes.reshape(NUM_SAMPLES, NUM_SAMPLES)
    # upper = upper.cpu().numpy().reshape(NUM_SAMPLES, NUM_SAMPLES)
    # lower = lower.cpu().numpy().reshape(NUM_SAMPLES, NUM_SAMPLES)
    x, y = X.cpu().numpy()[:, 0].reshape(NUM_SAMPLES, NUM_SAMPLES), X.cpu().numpy()[
        :, 1
    ].reshape(NUM_SAMPLES, NUM_SAMPLES)

    # Visualize GP and samples
    plot_2d_gp(theta[:, 0], theta[:, 1], x, y, predicted_laptimes, params)

    fig2, axs2 = plt.subplots()
    axs2.plot(
        np.cumsum(
            normalized_laptimes
            > opt_config["first_lap_multiplier"] * normalized_laptimes[0]
        )
    )
    axs2.set_xlabel("$iters$")
    axs2.set_ylabel("$Laptime constraint violations$")
    axs2.set_title("Violations")
    plt.show()

    # Extract minimum
    laptime = optimal_time * normalized_laptimes
    min_index = np.argmin(laptime)
    print(
        f"Minimum laptime: {laptime[min_index, :]} \n"
        f"Paremeters: {theta[min_index, :]} [{denormalized_theta[min_index, :]}]"
    )


def visualization_gpy(
    theta_path: str,
    laptime_path: str,
    domain_path: str,
    parameters_path: str,
    config: Dict[str, Any],
    n_samples: int,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Visualize the GP and the samples
    :param theta_path: path to the samples
    :param laptime_path: path to the laptime
    :param domain_path: path to the domain
    :param parameters_path: path to the parameters
    :param config: configuration
    :return: figure and axes
    """

    # Get objects
    theta = read_object(theta_path)
    laptimes = read_object(laptime_path)[:n_samples, :]

    # if "goose" in theta_path:
    #     laptimes[5:] += 0.25*4
    #     laptimes[laptimes == 40 + 0.25*4] = 40
    theta = theta[: laptimes.shape[0], :]
    params = read_object(parameters_path)
    domain = read_object(domain_path)
    denormalized_theta = theta * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
    opt_config = config["optimization_config"]
    optimal_time = config["interface_config"]["optimal_time"]

    # Fit GP
    # Build model (noise and signal variance are scaled)
    kernel = GPy.kern.Matern52(
        input_dim=domain.shape[0],
        ARD=False,
        variance=opt_config["kernel_variance"],
        lengthscale=opt_config["kernel_lengthscale"],
    )
    penalty_time = config["interface_config"]["max_time"]
    mf = GPy.core.Mapping(domain.shape[0], 1)
    if "prior_mean" in opt_config.keys():
        if not opt_config["prior_mean"]:
            mf.f = lambda x: 0
        else:
            mf.f = lambda x: -(opt_config["first_lap_multiplier"] - 0.02) * laptimes[0]
    else:
        mf.f = lambda x: -(opt_config["first_lap_multiplier"] - 0.02) * laptimes[0]

    mf.update_gradients = lambda a, b: 0
    mf.gradients_X = lambda a, b: 0
    objective_model = GPy.models.GPRegression(
        theta,
        -laptimes,
        kernel,
        noise_var=opt_config["gp_variance"] ** 2,
        mean_function=mf,
    )
    if domain.shape[0] == 2:
        x1 = np.linspace(0, 1, NUM_SAMPLES)
        x2 = np.linspace(0, 1, NUM_SAMPLES)
        x1x2 = np.array(list(product(x1, x2)))
        y_pred, cov_pred = objective_model.predict(x1x2)
        X0p, X1p = x1x2[:, 0].reshape(NUM_SAMPLES, NUM_SAMPLES), x1x2[:, 1].reshape(
            NUM_SAMPLES, NUM_SAMPLES
        )
        Zp = np.reshape(y_pred, (NUM_SAMPLES, NUM_SAMPLES))

        # Visualize GP and samples
        fig, ax = plot_2d_gp(theta[:, 0], theta[:, 1], X0p, X1p, Zp, params, -laptimes)

        # fig2, axs2 = plt.subplots()
        # axs2.plot(np.cumsum(laptimes >
        #                     opt_config['first_lap_multiplier'] * laptimes[0]))
        # axs2.set_xlabel("$iters$")
        # axs2.set_ylabel("$Laptime constraint violations$")
        # axs2.set_title("Violations")
    else:
        fig, ax = plt.subplots()
        objective_model.plot(plot_density=True, ax=ax, plot_limits=[0, 1])
        x_values = np.linspace(0, 1, opt_config["grid_size"])
        y_values = (
            -opt_config["first_lap_multiplier"]
            * laptimes[0]
            * np.ones(opt_config["grid_size"])
        )
        ax.plot(x_values, y_values, linestyle="dashed", color="k")
        ax.set_ylim(-32.5, -17.5)
        ax.set_xlim(0, 1)
        ax.set_title("Gaussian Process")
        ax.set_xlabel(f"{params[0]} weight")
        ax.set_ylabel("Negative laptime")
        ax.legend()
        # fig2, axs2 = plt.subplots()
        # axs2.plot(np.cumsum(laptimes >
        #                     opt_config['first_lap_multiplier'] * laptimes[0]))
        # axs2.set_xlabel("$iters$")
        # axs2.set_ylabel("$Laptime constraint violations$")
        # axs2.set_title("Violations")

    # Extract minimum
    min_index = np.argmin(laptimes)
    print(
        f"Minimum laptime: {laptimes[min_index, :]} \n"
        f"Paremeters: {theta[min_index, :]} [{denormalized_theta[min_index, :]}]"
    )
    # plt.show()
    if fig is not None:
        return fig, ax


if __name__ == "__main__":
    config = read_yaml(os.getcwd() + "/config/params.yaml")

    theta_file = os.getcwd() + "/objects/" + config["dir"] + "/theta.np"
    laptime_file = os.getcwd() + "/objects/" + config["dir"] + "/laptimes.np"
    constraints_file = os.getcwd() + "/objects/" + config["dir"] + "/theta.np"
    domain_file = os.getcwd() + "/objects/" + config["dir"] + "/domain.np"
    patameters_file = os.getcwd() + "/objects/" + config["dir"] + "/parameters.lst"

    safe_opt_config = read_yaml(
        os.path.join(os.getcwd(), "objects", config["dir"], "config.yaml")
    )
    if safe_opt_config["optimization_config"]["method"] in ["CRBO", "MH", "WML", "BO"]:
        visualization_gpy(
            theta_file,
            laptime_file,
            domain_file,
            patameters_file,
            safe_opt_config,
            n_samples=100,
        )
    else:
        normalized_laptime_file = (
            os.getcwd() + "/objects/" + config["dir"] + "/normalized_laptimes.np"
        )
        visualization(
            theta_file,
            normalized_laptime_file,
            domain_file,
            patameters_file,
            safe_opt_config,
        )
    plt.show()
