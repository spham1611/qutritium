"""Utility functions to analyze data"""
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from typing import Callable, List, Tuple
import numpy as np


# Fitting functions
def fit_function(x_values, y_values: List, function: Callable, init_params: List) -> Tuple:
    """

    :param x_values:
    :param y_values:
    :param function:
    :param init_params:
    :return:
    """
    *fit_parameters, conv = curve_fit(function, x_values, y_values, init_params)
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


def data_mitigatory(raw_data, assign_matrix):
    """data_mitigatory _summary_

    _extended_summary_

    Returns:
        _type_: _description_
    """
    cal_mat = np.transpose(assign_matrix)
    raw_data = raw_data
    num_shots = sum(raw_data)

    def fun(x):
        return sum((raw_data - np.dot(cal_mat, x)) ** 2)

    x0 = np.random.rand(len(raw_data))
    x0 = x0 / sum(x0)
    cons = {"type": "eq",
            "fun": lambda x: num_shots - sum(x)}
    bounds = tuple((0, num_shots) for x in x0)
    res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bounds, tol=1e-6)
    data_mitigated = res.x

    return data_mitigated


def reshape_complex_vec(vec: np.ndarray) -> np:
    """reshape_complex_vec

    Take in complex vector vec and return 2d array w/ real, imag entries. This is needed for the learning.

    Args:
        vec (np): complex vector of data
    Returns:
        np: vector w/ entries given by (real(vec], imag(vec))
    """
    length = len(vec)
    vec_reshaped = np.zeros((length, 2))
    for i in range(len(vec)):
        vec_reshaped[i] = [np.real(vec[i]), np.imag(vec[i])]
    return vec_reshaped
