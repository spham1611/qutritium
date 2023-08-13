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
"""
Unitary Gates: Elementary matrices
"""
from __future__ import annotations
from numpy import ndarray
from numpy.linalg import inv, LinAlgError
from src.quantumcircuit.qc_elementary_matrices import *
from typing import List
import numpy as np

pi = np.pi
state_0 = [[1], [0], [0]]
state_1 = [[0], [1], [0]]
state_2 = [[0], [0], [1]]


def single_matrix_form(gate_type: str, parameter: List[float] = None, omega=np.exp(1j * 2 * pi / 3)):
    """

    Args:
        gate_type: quantum gate type as define in gate_set
        parameter: parameter of rotation gate (if needed)
        omega: omega for phase gate
    Returns: matrix form of the qutrit gate

    """
    if gate_type == 'x01':
        return np.array([[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'rx01':
        return np.array([[np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2), 0],
                         [-1j * np.sin(parameter[0] / 2), np.cos(parameter[0] / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'g01':
        return np.array([[np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2) * np.exp(-1j * parameter[1]), 0],
                         [-1j * np.sin(parameter[0] / 2) * np.exp(1j * parameter[1]), np.cos(parameter[0] / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'x12':
        return np.array([[1, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0]], dtype=complex)
    elif gate_type == 'rx12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2)],
                         [0, -1j * np.sin(parameter[0] / 2), np.cos(parameter[0] / 2)]], dtype=complex)
    elif gate_type == 'g12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter[0] / 2), -1j * np.sin(parameter[0] / 2) * np.exp(-1j * parameter[1])],
                         [0, -1j * np.sin(parameter[0] / 2) * np.exp(1j * parameter[1]), np.cos(parameter[0] / 2)]],
                        dtype=complex)
    elif gate_type == 'Identity':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'x_plus':
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]], dtype=complex)
    elif gate_type == 'x_minus':
        return np.array([[0, 1, 0],
                         [0, 0, 1],
                         [1, 0, 0]], dtype=complex)
    elif gate_type == 'z01':
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'rz01':
        return np.array([[np.exp(-1j * parameter[0] / 2), 0, 0],
                         [0, np.exp(1j * parameter[0] / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'z12':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, -1]], dtype=complex)
    elif gate_type == 'rz12':
        return np.array([[1, 0, 0],
                         [0, np.exp(-1j * parameter[0] / 2), 0],
                         [0, 0, np.exp(1j * parameter[0] / 2)]], dtype=complex)
    elif gate_type == 'y01':
        return np.array([[0, -1j, 0],
                         [1j, 0, 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'ry01':
        return np.array([[np.cos(parameter[0] / 2), -np.sin(parameter[0] / 2), 0],
                         [np.sin(parameter[0] / 2), np.cos(parameter[0] / 2), 0],
                         [0, 0, 1]], dtype=complex)
    elif gate_type == 'y12':
        return np.array([[1, 0, 0],
                         [0, 0, -1j],
                         [0, 1j, 0]], dtype=complex)
    elif gate_type == 'ry12':
        return np.array([[1, 0, 0],
                         [0, np.cos(parameter[0] / 2), -np.sin(parameter[0] / 2)],
                         [0, np.sin(parameter[0] / 2), np.cos(parameter[0] / 2)]], dtype=complex)
    elif gate_type == 'hdm':
        return (1 / np.sqrt(3)) * np.array([[1, 1, 1],
                                            [1, omega, np.conj(omega)],
                                            [1, np.conj(omega), omega]], dtype=complex)
    elif gate_type == 'sdg':
        return np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, omega]], dtype=complex)
    elif gate_type == 'tdg':
        return np.array([[1, 0, 0],
                         [0, np.power(omega, 1 / 3), 0],
                         [0, 0, np.power(omega, -1 / 3)]], dtype=complex)
    elif gate_type == 'u_d':
        return np.array([[np.exp(1j*parameter[0]), 0, 0],
                         [0, np.exp(1j*parameter[1]), 0],
                         [0, 0, np.exp(1j*parameter[2])]], dtype=complex)
    elif gate_type == 'u_ft':
        return (1 / np.sqrt(3)) * np.array([[omega, 1, np.conj(omega)],
                                            [1, 1, 1],
                                            [np.conj(omega), 1, omega]], dtype=complex)
    else:
        raise Exception("This gate is not implemented yet.")


def multi_matrix_form(gate_type: str, first_index: int, second_index: int):
    """

    Args:
        gate_type: quantum gate type as define in gate_set
        first_index: acting qubit
        second_index: control qubit

    Returns: matrix form of the control-qutrit gate

    """

    if gate_type == 'CNOT':
        if second_index == first_index:
            raise Exception("Control qutrit and acting qutrit can not be the same")
        else:
            space = np.abs(first_index - second_index) - 1
            if space == 0:
                spacing = 1
            else:
                spacing = np.eye(3 ** space)
            if second_index < first_index:
                matrix = np.kron(np.kron(state_0 @ np.transpose(state_0), spacing), np.eye(3)) + \
                         np.kron(np.kron(state_1 @ np.transpose(state_1), spacing),
                                 single_matrix_form('x01') @ single_matrix_form('x12')) + \
                         np.kron(np.kron(state_2 @ np.transpose(state_2), spacing),
                                 single_matrix_form('x12') @ single_matrix_form('x01'))
            else:
                matrix = np.kron(np.kron(np.eye(3), spacing), state_0 @ np.transpose(state_0)) + \
                         np.kron(np.kron(single_matrix_form('x01') @ single_matrix_form('x12'), spacing),
                                 state_1 @ np.transpose(state_1)) + \
                         np.kron(np.kron(single_matrix_form('x12') @ single_matrix_form('x01'), spacing),
                                 state_2 @ np.transpose(state_2))
            return np.array(matrix, dtype=complex)


def statevector_to_state(state: np.array, n_qutrit: int):
    """

    Args:
        state: State vector of qutrits
        n_qutrit: number of qutrit

    Returns: state coefficient and state in ket form

    """

    if state.shape != (3 ** n_qutrit, 1):
        raise Exception("The dimension of the state is not align with given qubits")
    state_basis = []
    state_coeff = []
    for i in range(3 ** n_qutrit):
        if abs(complex(state[i][0])) != 0.0:
            state_basis.append(i)
            state_coeff.append(state[i][0])
    state_construction = []
    for j in range(len(state_basis)):
        sta = ''
        tmp = state_basis[j]
        for k in range(n_qutrit):
            sta += str(int(tmp % 3))
            tmp = tmp / 3
        state_construction.append(sta)
    return state_coeff, state_construction


def print_statevector(state: np.array, n_qutrit: int):
    """

    Args:
        state: State vector of qutrits
        n_qutrit: number of qutrit

    Print the state vector of qutrits to ket form
    """
    state_coeff, state_cons = statevector_to_state(state, n_qutrit)
    print("State: ")
    for i in range(len(state_cons)):
        if i == len(state_cons) - 1:
            print(str(state_coeff[i]) + " |" + str(state_cons[i]) + ">")
        else:
            print(str(state_coeff[i]) + " |" + str(state_cons[i]) + "> + ")


def checking_unitary(u: ndarray):
    """

    Args:
        u: Matrix to check

    Returns: True if matrix is unitary, False otherwise

    """
    try:
        p = u @ inv(u)
    except LinAlgError:
        print("The matrix can not inverse")
        return False
    if np.absolute(np.sum(p - np.eye(3))) < 1e-5:
        return True
    else:
        return False
