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

from typing import Callable, Tuple, Iterable
from numpy.typing import NDArray

from qiskit.visualization.state_visualization import generate_facecolors
from qiskit.visualization.utils import matplotlib_close_if_inline


# Fitting functions
def fit_function(x_values: Iterable[float], y_values: Iterable[float],
                 function: Callable, init_params: Iterable[float]) -> Tuple:
    """ Used to plot fit line based on given x, y values
    Args:
        x_values:
        y_values:
        function: Evaluating function
        init_params: coefficient guess

    Returns:
        fit_params: optimal values for the least squared
        y_fit: fit line
    """
    from scipy.optimize import curve_fit

    fit_parameters, *_ = curve_fit(function, x_values, y_values, init_params, maxfev=5000)
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


def plot_tomography(
        su3_matrix: NDArray,
        title="",
        fig_size=None,
        color=None,
        alpha=1,
        file_name=None,
        row_names: str = r'',
        column_names: str = '',
) -> None:
    """This function is used to plot density matrix of qutrits.
    Code resembles qiskit.visualization.state_visualization.py :: plot_state_city

    Args:
        su3_matrix:
        title:
        fig_size:
        color:
        alpha:
        file_name:
        column_names:
        row_names:

    Returns:

    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Get the number of qutrits
    if su3_matrix.ndim != 2 \
            or su3_matrix.shape[0] != su3_matrix.shape[1] \
            or su3_matrix.shape[0] % 3 != 0 \
            or su3_matrix.shape[1] % 3 != 0:
        raise ValueError("Invalid dimensions {0}".format(su3_matrix))

    # Get the real and img parts of the matrix and their respective shape
    data_real = np.real(su3_matrix)
    data_img = np.imag(su3_matrix)
    lx = len(data_real[0])
    ly = len(data_real[:, 0])

    # Get the labels
    column_names = column_names
    row_names = row_names

    # Set up mesh positions
    xpos = np.arange(0, lx, 1)
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = data_real.flatten()
    dzi = data_img.flatten()

    if color is None:
        color = ["#648fff", "#648fff"]
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        if color[0] is None:
            color[0] = "#648fff"
        if color[1] is None:
            color[1] = "#648fff"

    if fig_size is None:
        fig_size = (15, 5)

    fig = plt.figure(fig_size=fig_size)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    max_dzr = max(dzr)
    min_dzr = min(dzr)
    min_dzi = np.min(dzi)
    max_dzi = np.max(dzi)

    if ax1 is not None:
        fc1 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzr, color[0])
        # noinspection DuplicatedCode
        for idx, cur_zpos in enumerate(zpos):
            if dzr[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b1 = ax1.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzr[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b1.set_facecolors(fc1[6 * idx: 6 * idx + 6])

        xlim, ylim = ax1.get_xlim(), ax1.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc1 = Poly3DCollection(verts, alpha=0.15, facecolor="k", linewidths=1, zorder=1)

        if min(dzr) < 0 < max(dzr):
            ax1.add_collection3d(pc1)
        ax1.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax1.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dzr != min_dzr:
            ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, max_dzi))
        else:
            if min_dzr == 0:
                ax1.axes.set_zlim3d(np.min(dzr), max(np.max(dzr) + 1e-9, np.max(dzi)))
            else:
                ax1.axes.set_zlim3d(auto=True)
        ax1.get_autoscalez_on()
        ax1.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax1.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax1.set_zlabel("Re[$\\rho$]", fontsize=14)
        for tick in ax1.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)

    if ax2 is not None:
        fc2 = generate_facecolors(xpos, ypos, zpos, dx, dy, dzi, color[1])
        # noinspection DuplicatedCode
        for idx, cur_zpos in enumerate(zpos):
            if dzi[idx] > 0:
                zorder = 2
            else:
                zorder = 0
            b2 = ax2.bar3d(
                xpos[idx],
                ypos[idx],
                cur_zpos,
                dx[idx],
                dy[idx],
                dzi[idx],
                alpha=alpha,
                zorder=zorder,
            )
            b2.set_facecolors(fc2[6 * idx: 6 * idx + 6])

        xlim, ylim = ax2.get_xlim(), ax2.get_ylim()
        x = [xlim[0], xlim[1], xlim[1], xlim[0]]
        y = [ylim[0], ylim[0], ylim[1], ylim[1]]
        z = [0, 0, 0, 0]
        verts = [list(zip(x, y, z))]

        pc2 = Poly3DCollection(verts, alpha=0.2, facecolor="k", linewidths=1, zorder=1)

        if min(dzi) < 0 < max(dzi):
            ax2.add_collection3d(pc2)
        ax2.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax2.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if min_dzi != max_dzi:
            eps = 0
            ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
        else:
            if min_dzi == 0:
                ax2.set_zticks([0])
                eps = 1e-9
                ax2.axes.set_zlim3d(np.min(dzi), max(np.max(dzr) + 1e-9, np.max(dzi) + eps))
            else:
                ax2.axes.set_zlim3d(auto=True)

        ax2.xaxis.set_ticklabels(row_names, fontsize=14, rotation=45, ha="right", va="top")
        ax2.yaxis.set_ticklabels(column_names, fontsize=14, rotation=-22.5, ha="left", va="center")
        ax2.set_zlabel("Im[$\\rho$]", fontsize=14)
        for tick in ax2.zaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        ax2.get_autoscalez_on()

    fig.suptitle(title, fontsize=16)
    matplotlib_close_if_inline(fig)
    if file_name is None:
        return fig
    else:
        return fig.savefig(file_name)


def deprecate_function():
    ...


def deprecate_arguments():
    ...

