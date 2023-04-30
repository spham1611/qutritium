import numpy as np
from .qc_utility import print_statevector
from .instruction_structure import Instruction


class Qutrit_circuit:
    """
    This class defines the wrapper for Virtual Machine which can be treated as a Quantum Circuit
    """

    def __init__(self, n_qutrit: int, initial_state: np.array):
        """
        :param n_qutrit: Number of qutrit
        :param initial_state: The initial state of the quantum circuit
        """
        self._measurement_result = None
        self._measurement_flag = False
        self.n_qutrit = n_qutrit
        self._operation_set = []
        self._dimension = 3 ** n_qutrit
        self.state = []
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

    def add_gate(self, gate_type: str, first_qutrit_set: int, second_qutrit_set: int = None, parameter: float = None):
        """
        :param gate_type: quantum gate type as define in gate_set
        :param first_qutrit_set: acting qubits
        :param second_qutrit_set: control qubits
        :param parameter: parameter of rotation gate (if needed)
        :return:
        """
        if gate_type != 'measure':
            ins = Instruction(gate_type, self.n_qutrit, first_qutrit_set, second_qutrit_set,
                              parameter)
            self._operation_set.append(ins)
        else:
            self._measurement_flag = True
            self._operation_set.append("measurement")

    def return_operation_set(self):
        return self._operation_set

    def return_meas_flag(self):
        return self._measurement_flag

    # TODO: Add image circuit
    def draw(self):
        """
        Representation of the quantum circuit
        """
        print("Initial state of the circuit: ")
        print_statevector(self.initial_state, self.n_qutrit)
        print("Final state of the circuit: ")
        print_statevector(self.state, self.n_qutrit)
        print("Set of gate on the circuits: ")
        for i in self._operation_set:
            if type(i) == Instruction:
                i.print()
            else:
                print(i)
