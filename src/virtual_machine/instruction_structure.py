from VM_utility import matrix_form

gate_set = ['x+',
            'S',
            'T',
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
    def __init__(self, gate_type, n_qutrit, qutrit_state, first_qutrit_set, second_qutrit_set=None, parameter=None):
        self.type = gate_type
        self.verify_gate()
        self.n_qutrit = n_qutrit
        self.qutrit_state = qutrit_state
        if parameter is not None:
            self.parameter = parameter
        self.first_qutrit = first_qutrit_set
        self.second_qutrit = second_qutrit_set
        self.gate_matrix = matrix_form(self.type)
        self.__effect()

    def __effect(self):
        for i in range(self.n_qutrit):
            if i == self.first_qutrit:
                self.qutrit_state["qutrit_" + str(i)] = self.gate_matrix @ self.qutrit_state["qutrit_" + str(i)]

    def verify_gate(self):
        if self.type not in gate_set:
            raise Exception("This gate is not defined in set of gate")

    def first_qutrit_index(self):
        return self.first_qutrit

    def matrix(self):
        return self.gate_matrix

    def return_effect(self):
        return self.qutrit_state

    def print(self):
        if self.second_qutrit is None:
            print("Gate " + str(self.type) + ", acting qutrit: " + str(self.first_qutrit))
        else:
            print("Gate " + str(self.type) + ", acting qutrit: " \
                  + str(self.first_qutrit) + ", control qutrit: " + str(self.second_qutrit))
