"""Utility functions to analyze data
..deprecated:: 0.0
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Callable, Tuple, Iterable
from numpy.typing import NDArray


# Fitting functions
def fit_function(x_values: Iterable, y_values: Iterable, function: Callable, init_params: Iterable) -> Tuple:
    """

    :param x_values: can be List or np.ndarray
    :param y_values: can be List or np.ndarray
    :param function:
    :param init_params: lambda_list
    :return:
    """
    fit_parameters, *_ = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fit_parameters)
    return fit_parameters, y_fit


def average_counter(counts, num_shots) -> float:
    """

    :param counts:
    :param num_shots:
    :return:
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
    """reshape_complex_vec
    Take in complex vector vec and return 2d array w/ real, imag entries. This is needed for the learning.

    :param: vec (np): complex vector of data
    :return: np: vector w/ entries given by (real(vec], imag(vec))
    """
    length = len(vec)
    vec_reshaped = np.zeros((length, 2))
    for i in range(len(vec)):
        vec_reshaped[i] = [np.real(vec[i]), np.imag(vec[i])]
    return vec_reshaped

