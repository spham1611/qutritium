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
The Qutrit circuit that contains the qutrit gates and measurement
"""
import numpy as np
from typing import List, Union
from numpy.typing import NDArray
from src.quantumcircuit.qc_utility import print_statevector
from src.quantumcircuit.instruction_structure import Instruction


class Qutrit_circuit:
    """
    This class defines the wrapper for Virtual Machine which can be treated as a Quantum Circuit
    """

    def __init__(self, n_qutrit: int, initial_state: Union[NDArray, None]) -> None:
        """

        Args:
            n_qutrit: Number of qutrit
            initial_state: The initial state of the quantum circuit
        """
        self.n_qutrit = n_qutrit
        self.state = []
        self._measurement_result = None
        self._measurement_flag = False
        self._operation_set: List = []
        self._dimension = 3 ** n_qutrit
        if initial_state is not None:
            if initial_state.shape == (self._dimension, 1):
                self.initial_state = initial_state
                self.state = initial_state
            else:
                raise Exception(
                    "The initial state declared does not have correct dimension. The current shape is " + str(
                        initial_state.shape))
        else:
            self.initial_state = np.array([[0] * self._dimension]).transpose()
            self.initial_state[0][0] = 1
            self.state = self.initial_state

    def add_gate(self, gate_type: str,
                 first_qutrit_set: int,
                 second_qutrit_set: int = None,
                 parameter: List[float] = None,
                 to_all: bool = False,
                 is_dagger: bool = False) -> None:
        """
        Args:
            gate_type: the type of qutrit gate
            first_qutrit_set: the index of first qutrit that the gate has effect on
            second_qutrit_set: the index of second qutrit that the gate has effect on (for multi-qutrit gate)
            parameter: the parameter of the gate
            to_all: apply the gate to all qutrit in the circuit or not?
            is_dagger: complex conjugate of the given gate
        Returns:
        """
        if to_all is True and second_qutrit_set is None:
            for i in range(self.n_qutrit):
                ins = Instruction(gate_type=gate_type, n_qutrit=self.n_qutrit,
                                  first_qutrit_set=i, second_qutrit_set=second_qutrit_set,
                                  parameter=parameter, inverse=is_dagger)
                self.operation_set = [ins]

        else:
            ins = Instruction(gate_type=gate_type, n_qutrit=self.n_qutrit,
                              first_qutrit_set=first_qutrit_set, second_qutrit_set=second_qutrit_set,
                              parameter=parameter, inverse=is_dagger)
            self.operation_set = [ins]

    def measure_all(self):
        """
        Adding measurement in the qutrit circuit
        """
        if self._measurement_flag is True:
            raise Exception("A measurement has already been added to the circuit.")
        else:
            self._measurement_flag = True
            self.operation_set = ['measurement']

    @property
    def operation_set(self) -> List:
        return self._operation_set

    @operation_set.setter
    def operation_set(self, op: List[Union[Instruction, str]]) -> None:
        self._operation_set.extend(op)

    @property
    def measurement_flag(self) -> bool:
        return self._measurement_flag

    def reset_circuit(self) -> None:
        """
        Delete all the elements of the operation set
        """
        self._operation_set.clear()

    # TODO: Add image circuit
    def draw(self):
        """
        Representation of the quantum circuit
        """
        print("Initial state of the circuit: ")
        print_statevector(self.initial_state, self.n_qutrit)
        print("Set of gate on the circuits: ")
        for i in self._operation_set:
            if type(i) == Instruction:
                i.print()
            else:
                print(i)

    def __add__(self, object2):
        if self.n_qutrit != object2.n_qutrit:
            raise Exception("The two circuit has different number of qubits")
        elif self.measurement_flag is True:
            raise Exception("The first circuit contains measurement which is prohibited")
        else:
            self._operation_set += object2.operation_set
            self._measurement_flag = object2.measurement_flag
            return self
