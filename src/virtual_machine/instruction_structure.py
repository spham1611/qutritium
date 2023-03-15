import numpy as np
from VM_utility import single_matrix_form
from VM_utility import multi_matrix_form

gate_set = ['x+',
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
    def __init__(self, gate_type, n_qutrit, state, first_qutrit_set, second_qutrit_set=None, parameter=None):
        self.type = gate_type
        self.verify_gate()
        self.n_qutrit = n_qutrit
        self.parameter = parameter
        self.first_qutrit = first_qutrit_set
        self.second_qutrit = second_qutrit_set
        self.state = state
        self.is_two_qutrit_gate = False
        if second_qutrit_set is not None:
            self.is_two_qutrit_gate = True
            self.gate_matrix = multi_matrix_form(self.type, self.first_qutrit, self.second_qutrit)
            print(self.gate_matrix)
        else:
            self.is_two_qutrit_gate = False
            self.gate_matrix = single_matrix_form(self.type)
        self.__effect()

    def __effect(self):
        """
        Perform the gate on the quantum state
        """
        if not self.is_two_qutrit_gate:
            if self.n_qutrit == 1:
                self.state = self.gate_matrix @ self.state
            else:
                if self.first_qutrit == 0:
                    default_matrix = self.gate_matrix
                else:
                    default_matrix = np.eye(3)
                for i in range(self.n_qutrit - 1):
                    if i == (self.first_qutrit - 1):
                        default_matrix = np.kron(default_matrix, self.gate_matrix)
                    else:
                        default_matrix = np.kron(default_matrix, np.eye(3))
                self.state = default_matrix @ self.state
        else:
            left = min((self.first_qutrit, self.second_qutrit))
            right = max((self.first_qutrit, self.second_qutrit))
            if left == 0:
                default_matrix = self.gate_matrix
            else:
                default_matrix = np.eye(3)
            for i in range(self.n_qutrit-1):
                if i == (left-1):
                    default_matrix = np.kron(default_matrix, self.gate_matrix)
                if i+1 < left or i+1 > right:
                    default_matrix = np.kron(default_matrix, np.eye(3))
            self.state = default_matrix @ self.state

    def verify_gate(self):
        if self.type not in gate_set:
            raise Exception("This gate is not defined in set of gate")

    def matrix(self):
        return self.gate_matrix

    def return_effect(self):
        return self.state

    def print(self):
        if self.second_qutrit is None:
            print("Gate " + str(self.type) + ", acting qutrit: " + str(self.first_qutrit))
        else:
            print("Gate " + str(self.type) + ", acting qutrit: " \
                  + str(self.first_qutrit) + ", control qutrit: " + str(self.second_qutrit))
