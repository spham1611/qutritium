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
Support the construction of Qutrit tomography
"""
import copy

import numpy as np
from src.quantumcircuit.QC import Qutrit_circuit

Gell_man_matrices = [
    np.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 0]]),
    np.array([[0, -1j, 0],
              [1j, 0, 0],
              [0, 0, 0]]),
    np.array([[1, 0, 0],
              [0, -1, 0],
              [0, 0, 0]]),
    np.array([[0, 0, 1],
              [0, 0, 0],
              [1, 0, 0]]),
    np.array([[0, 0, -1j],
              [0, 0, 0],
              [1j, 0, 0]]),
    np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0]]),
    np.array([[0, 0, 0],
              [0, 0, -1j],
              [0, 1j, 0]]),
    1 / np.sqrt(3) * np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, -2]])]


class Tomography:
    """
    The class support the construction of Qutrit tomography given a Qutrit circuit
    """
    def __init__(self, circuit: Qutrit_circuit):
        """
        Args:
            circuit: Qutrit circuit to be used for tomography
        """
        self.circuit = circuit
        self.n_qutrit = circuit.n_qutrit
        self._exp_circuit = [self.circuit]
        if circuit.measurement_flag:
            raise Exception("The given circuit should not contains measurement")
        self.tomo_exp_op = [["rx01", "", [np.pi/2, 0]], ["ry01", "", [np.pi/2, 0]], ["rx01", "", [np.pi, 0]],
                            ["rx12", "", [np.pi/2, 0]], ["ry12", "", [np.pi/2, 0]], ["rx01", "rx12", [np.pi, np.pi/2]],
                            ["rx01", "ry12", [np.pi, np.pi/2]], ["rx01", "rx12", [np.pi, np.pi]]]

    def construct_tomography_exp(self) -> None:
        """
        Returns: Construct the qutrit circuits experiments for tomography
        """
        for i in self.tomo_exp_op:
            tmp_circuit = copy.copy(self.circuit)
            if i[1] == "":
                tmp_circuit.add_gate(gate_type=i[0], first_qutrit_set=0, parameter=i[2][0], to_all=True)
            else:
                tmp_circuit.add_gate(gate_type=i[0], first_qutrit_set=0, parameter=i[2][0], to_all=True)
                tmp_circuit.add_gate(gate_type=i[1], first_qutrit_set=0, parameter=i[2][1], to_all=True)
            tmp_circuit.measure_all()
            self._exp_circuit.append(tmp_circuit)

    def execute_tomography(self):
        return

    def tomography_exp(self):
        return self._exp_circuit
