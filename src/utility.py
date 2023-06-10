# MIT License
#
# Copyright (c) [2023] [son pham, tien nguyen, bach bao]
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

"""Utility functions to analyze data
..deprecated:: 0.0
"""
import numpy as np
from scipy.optimize import curve_fit

from typing import Callable, Tuple, Iterable
from numpy.typing import NDArray


# Fitting functions
def fit_function(x_values: Iterable[float], y_values: Iterable[float],
                 function: Callable, init_params: Iterable[float]) -> Tuple:
    """ Used to plot fit line based on given x, y values
    Args:
        x_values:
        y_values:
        function: Evaluating function
        init_params: coefficient calibration

    Returns:
        fit_params: optimal values for the least squared
        y_fit: fit line
    """
    fit_parameters, *_ = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fit_parameters)
    return fit_parameters, y_fit


def average_counter(counts: Iterable, num_shots: int) -> NDArray:
    """ Simple average over an array
    Args:
        counts:
        num_shots:

    Returns:
        Averaged Array
    """
    all_exp = []
    for j in counts:
        zero = 0
        for i in j.keys():
            if i[-1] == "0":
                zero += j[i]
        all_exp.append(zero)
    return np.array(all_exp) / num_shots


def reshape_complex_vec(vec: NDArray) -> NDArray:
    """ Take in complex vector vec and return 2d array w/ real, imag entries.

    Args:
        vec:

    Returns:
        Real Array Value
    """
    length = len(vec)
    vec_reshaped = np.zeros((length, 2))
    for i in range(len(vec)):
        vec_reshaped[i] = [np.real(vec[i]), np.imag(vec[i])]
    return vec_reshaped
