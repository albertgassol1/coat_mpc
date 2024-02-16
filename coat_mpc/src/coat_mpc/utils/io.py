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

import pickle

import numpy as np


def read_object(file_path: str) -> np.ndarray:
    """
    Read numpy array from file path
    :param file_path: path to numpy array pickable object
    """
    with open(file_path, "rb") as data_file:
        array: np.ndarray = pickle.load(data_file)
        return array


def save_to_file(file_name: str, data: np.ndarray) -> None:
    """
    Save data to specific file
    :param file_name: file name path
    :param data: data to save
    """
    with open(file_name, "wb") as output_file:
        pickle.dump(data, output_file)
