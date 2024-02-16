#!/usr/bin/env python3

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

from typing import Optional, Tuple

import numpy as np
import torch


def normalize(input_vector: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """
    Normalize input vector to be between 0 and 1
    :param input_vector: input vector to normalize
    :param domain: domain of input vector
    :return: normalized vector
    """
    return (input_vector - domain[:, 0]) / (domain[:, 1] - domain[:, 0])


def denormalize(input_vector: np.ndarray, domain: np.ndarray) -> np.ndarray:
    """
    Denormalize input vector from [0, 1] to its domain
    :param input_vector: input vector to denormalize
    :param domain: domain of input vector
    :return: denormalized vector
    """

    return input_vector * (domain[:, 1] - domain[:, 0]) + domain[:, 0]


def standardize(
    samples: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    stddev: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardizes (zero mean, unit variance) a tensor by dim=-2.
    If the tensor is single-dimensional, simply standardizes the tensor.
    If for some batch index all elements are equal (or if there is only a single
    data point), this function will return 0 for that batch index
    :param samples: A `batch_shape x n x m`-dim tensor
    :param mean: mean to standardize with
    :param stddev: standard deviation to standardize with
    :return: The standardized `Y`, mean and stddev
    """
    if mean is not None and stddev is not None:
        return samples - mean / stddev, mean, stddev

    stddim = -1 if samples.dim() < 2 else -2
    samples_std = samples.std(dim=stddim, keepdim=True)
    samples_std = samples_std.where(
        samples_std >= 1e-9, torch.full_like(samples_std, 1.0)
    )
    samples_mean = samples.mean(dim=stddim, keepdim=True)
    return (samples - samples_mean) / samples_std, samples_mean, samples_std
