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
Instruction class that can be called from VM
Used to represent a gate in quantum circuit
"""
from __future__ import annotations
import numpy as np
from typing import Union, Any, List
from numpy.typing import NDArray
# from .qc_elementary_matrices import *
from src.quantumcircuit.qc_utility import multi_matrix_form, single_matrix_form

gate_set: list[Union[str, Any]] = ['Identity',
                                   'x_plus',
                                   'x_minus',
                                   'sdg',
                                   'tdg',
                                   'CNOT',
                                   'g01',
                                   'g12',
                                   'x01',
                                   'x12',
                                   'y01',
                                   'y12',
                                   'z01',
                                   'z12',
                                   'rx01',
                                   'rx12',
                                   'ry01',
                                   'ry12',
                                   'rz01',
                                   'rz12',
                                   'u_d',
                                   'hdm',
                                   'u_ft']


class Instruction:
    """
    The class is used to represent a gate in VM,
    Each gate can be considered as an instruction and each has effect on the final state
    """

    def __init__(self, gate_type: str,
                 n_qutrit: int, first_qutrit_set: int,
                 second_qutrit_set: int | None = 0,
                 parameter: List[float] = None,
                 inverse: bool = False,
                 custom: bool = False,
                 custom_matrix: NDArray = None) -> None:
        self._type = gate_type
        if custom is False:
            self._verify_gate()
        self.n_qutrit: int = n_qutrit
        self.qutrit_dimension: int = 3 ** self.n_qutrit
        self.parameter: List[float] = parameter
        self.first_qutrit: int = first_qutrit_set
        self.second_qutrit: int = second_qutrit_set
        self._is_two_qutrit_gate: bool = False
        self._is_inverse = inverse
        self._is_custom = custom
        if first_qutrit_set > (self.n_qutrit - 1):
            raise Exception("Acting qutrit is not defined")
        if second_qutrit_set is not None:
            self._is_two_qutrit_gate = True
            # if self._type in parameterless_set: eval(self._type)
            if self._is_inverse is False:
                self.gate_matrix = multi_matrix_form(gate_type=self._type, first_index=self.first_qutrit,
                                                     second_index=self.second_qutrit)
            else:
                self.gate_matrix = np.matrix(multi_matrix_form(gate_type=self._type, first_index=self.first_qutrit,
                                                               second_index=self.second_qutrit)).getH()
        else:
            self._is_two_qutrit_gate = False
            # Check if the gate is custom
            if self._is_custom is False:
                self.gate_matrix = single_matrix_form(gate_type=self._type, parameter=self.parameter)
            else:
                self.gate_matrix = custom_matrix
            # Check if we want the inverse of this gate
            if self._is_inverse is True:
                self.gate_matrix = np.matrix(self.gate_matrix).getH()
        self._effect_matrix = self._effect()

    def _effect(self) -> NDArray:
        """

        Returns: the matrix form effect of gate on the quantum state

        """
        if not self._is_two_qutrit_gate:
            if self.n_qutrit == 1:
                return self.gate_matrix
            else:
                if self.first_qutrit == 0:
                    effect_matrix = np.einsum('ik,jl', self.gate_matrix,
                                              np.eye(int(self.qutrit_dimension / 3))).reshape(self.qutrit_dimension,
                                                                                              self.qutrit_dimension)
                else:
                    effect_matrix = np.einsum('ik,jl', np.eye(3 ** self.first_qutrit),
                                              self.gate_matrix).reshape(3 ** (self.first_qutrit + 1),
                                                                        3 ** (self.first_qutrit + 1))
                    effect_matrix = np.einsum('ik,jl', effect_matrix,
                                              np.eye(3 ** (self.n_qutrit - self.first_qutrit - 1))).reshape(
                        self.qutrit_dimension,
                        self.qutrit_dimension)
                return effect_matrix
        else:
            left = min((self.first_qutrit, self.second_qutrit))
            right = max((self.first_qutrit, self.second_qutrit))
            if left == 0:
                effect_matrix = np.einsum('ik,jl', self.gate_matrix,
                                          np.eye(3 ** (self.n_qutrit - right - 1))).reshape(self.qutrit_dimension,
                                                                                            self.qutrit_dimension)
            else:
                effect_matrix = np.einsum('ik,jl', np.eye(3 ** left),
                                          self.gate_matrix).reshape(3 ** (self.first_qutrit + 1),
                                                                    3 ** (self.first_qutrit + 1))
                effect_matrix = np.einsum('ik,jl', effect_matrix,
                                          np.eye(3 ** (self.n_qutrit - right - 1))).reshape(
                    self.qutrit_dimension,
                    self.qutrit_dimension)
            return effect_matrix

    def _verify_gate(self) -> None:
        if self._type not in gate_set:
            raise Exception("This gate is not defined in set of gates")

    @property
    def effect_matrix(self) -> NDArray:
        return self._effect_matrix

    def type(self) -> str:
        return self._type

    def print(self):
        """
        Support printing out the instruction description
        """
        if not self._is_two_qutrit_gate:
            if self.parameter is None:
                print("Gate " + str(self._type) + ", acting qutrit: " + str(self.first_qutrit))
            else:
                print("Gate " + str(self._type) + " with parameter " + str(self.parameter) +
                      ", acting qutrit: " + str(self.first_qutrit))
        else:
            print("Gate " + str(self._type) + ", acting qutrit: "
                  + str(self.first_qutrit) + ", control qutrit: " + str(self.second_qutrit))
