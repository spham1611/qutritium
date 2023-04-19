"""
Instruction class that can be called from VM
Used to represent a gate in quantum circuit
"""
from typing import Union, Any
import numpy as np
from .QC_utility import single_matrix_form, multi_matrix_form

gate_set: list[Union[str, Any]] = ['I',
                                   'x+',
                                   'x-',
                                   'S',
                                   'T',
                                   'CNOT',
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
                                   'WH',
                                   'measure']


class Instruction:
    """
    The class is used to represent a gate in VM,
    Each gate can be considered as an instruction and each has effect on the final state
    """

    def __init__(self, gate_type: str, n_qutrit: int, first_qutrit_set: int,
                 second_qutrit_set: int = None, parameter: float = None):
        self._type = gate_type
        self._verify_gate()
        self.n_qutrit = n_qutrit
        self.qutrit_dimension = 3 ** self.n_qutrit
        self.parameter = parameter
        self.first_qutrit = first_qutrit_set
        self.second_qutrit = second_qutrit_set
        self._is_two_qutrit_gate = False
        if first_qutrit_set > (self.n_qutrit - 1):
            raise Exception("Acting qutrit is not defined")
        if second_qutrit_set is not None:
            self._is_two_qutrit_gate = True
            self.gate_matrix = multi_matrix_form(gate_type=self._type, first_index=self.first_qutrit,
                                                 second_index=self.second_qutrit)
        else:
            self._is_two_qutrit_gate = False
            self.gate_matrix = single_matrix_form(gate_type=self._type, parameter=self.parameter)
        self._effect_matrix = self._effect()

    def _effect(self):
        """
        Return the matrix form effect of gate on the quantum state
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
                                          np.eye(3**(self.n_qutrit-right-1))).reshape(self.qutrit_dimension,
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

    def return_effect(self):
        return self._effect_matrix

    def _verify_gate(self):
        if self._type not in gate_set:
            raise Exception("This gate is not defined in set of gate")

    def matrix(self):
        return self.gate_matrix

    def print(self):
        if not self._is_two_qutrit_gate:
            print("Gate " + str(self._type) + ", acting qutrit: " + str(self.first_qutrit))
        else:
            print("Gate " + str(self._type) + ", acting qutrit: "
                  + str(self.first_qutrit) + ", control qutrit: " + str(self.second_qutrit))
