from VM_utility import matrix_form
import numpy as np
from instruction_structure import instruction


class Virtual_Machine:
    def __init__(self, n_qubits=int, n_classical=int, initial_state=[int]):
        self.n_qubits = n_qubits
        self.n_classical = n_classical
        self.operation_set = []
        if initial_state is not None:
            if initial_state.shape == (3 ** n_qubits, 3 ** n_qubits):
                self.initial_state = initial_state
                self.state = initial_state
            else:
                raise Exception(
                    "The initial state declared does not have correct dimension. The current shape is " + str(
                        initial_state.shape))
        else:
            self.state = [0] * (3 ** n_qubits)
            self.state[0] = 1
            self.initial_state = self.state

    def add_gate(self, instruct=instruction):
        gate_matrix = matrix_form(instruct.gate_type())
        self.operation_set.append(instruct)
        if self.n_qubits == 1:
            self.state = self.state @ gate_matrix
        else:
            if instruct.first_qubit_index() == 0:
                default_matrix = gate_matrix
            else:
                default_matrix = np.eye(3)
            for i in range(self.n_qubits - 1):
                if i == (instruct.first_qubit_index() + 1):
                    default_matrix = np.kron(default_matrix, gate_matrix)
                else:
                    default_matrix = np.kron(default_matrix, np.eye(3))
            self.state = self.state @ default_matrix

    def return_final_state(self):
        return self.state

    def return_operation_set(self):
        return self.operation_set

    def draw(self):
        print("Initial state of the circuit: " + str(self.initial_state))
        print("Final state of the circuit: " + str(self.state))
        print("Set of gate on the circuits: ")
        for i in self.operation_set:
            i.print()
