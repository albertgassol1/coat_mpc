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

# Libraries
from typing import Union

import torch
from botorch.acquisition import (
    ConstrainedExpectedImprovement,
    ExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.models import SingleTaskGP


class AcquisitionFunction:
    def __init__(self, acquisition_function: str, beta: float, c_th: float) -> None:
        """
        Acquisition functions to perform bayesian optimization
        :param acquisition_function: acquisition function choice
        :param beta: beta value
        :param c_th: threshold multiplier for EIC
        """

        # List of available acquisition functions (SAFEOPT is handled in bayesopt directly
        # Check that acquisition function exists
        if acquisition_function not in ["EIC", "UCB", "SAFEOPT"]:
            raise NameError("Chosen acquisition function does not exist")

        # Save chosen acquisition function
        self.chosen_acquisition_function = acquisition_function

        # Beta value for UCB
        self.beta = beta

        # Threshold multiplier for EIC
        self.c_th = c_th

    def acquisition_function(
        self, model: SingleTaskGP
    ) -> Union[
        ExpectedImprovement, UpperConfidenceBound, ConstrainedExpectedImprovement
    ]:
        """
        Compute acquisition function.
        :param model: gaussian process model for our laptimes function
        :param c_th: threshold for the constraints' probability error
        :return: acquisition function to optimize
        """

        if self.chosen_acquisition_function == "EI":
            best_f = torch.max(model.train_targets[0, :])
            return ExpectedImprovement(model, best_f, maximize=True)
        if self.chosen_acquisition_function == "UCB":
            return UpperConfidenceBound(model, beta=self.beta, maximize=True)
        if self.chosen_acquisition_function == "EIC":
            best_f = torch.max(model.train_targets[0, :])
            constraints = {1: (self.c_th * model.train_targets[1, 0], None)}
            return ConstrainedExpectedImprovement(
                model,
                best_f=best_f,
                objective_index=0,
                constraints=constraints,
                maximize=True,
            )

        raise NameError("Wrong acquisition function")
