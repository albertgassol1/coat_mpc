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

# Libraries
from typing import Union

import torch
from botorch.acquisition import ConstrainedExpectedImprovement, ExpectedImprovement, UpperConfidenceBound
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
            raise NameError('Chosen acquisition function does not exist')

        # Save chosen acquisition function
        self.chosen_acquisition_function = acquisition_function

        # Beta value for UCB
        self.beta = beta

        # Threshold multiplier for EIC
        self.c_th = c_th

    def acquisition_function(self, model: SingleTaskGP) -> \
            Union[ExpectedImprovement, UpperConfidenceBound, ConstrainedExpectedImprovement]:
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
            return ConstrainedExpectedImprovement(model,
                                                  best_f=best_f,
                                                  objective_index=0,
                                                  constraints=constraints,
                                                  maximize=True)

        raise NameError("Wrong acquisition function")
