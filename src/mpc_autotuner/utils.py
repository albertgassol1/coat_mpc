#!/usr/bin/env python3

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

def normalize(input_vector: np.ndarray,
              domain: np.ndarray) -> np.ndarray:
    """
    Normalize input vector to be between 0 and 1
    :param input_vector: input vector to normalize
    :param domain: domain of input vector
    :return: normalized vector
    """
    return (input_vector - domain[:, 0]) / (domain[:, 1] - domain[:, 0])


def denormalize(input_vector: np.ndarray,
                domain: np.ndarray) -> np.ndarray:
    """
    Denormalize input vector from [0, 1] to its domain
    :param input_vector: input vector to denormalize
    :param domain: domain of input vector
    :return: denormalized vector
    """

    return input_vector * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
