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
            'WH']


class instruction:
    def __init__(self, gate_type, first_qubit_set=None, second_qubit_set=None, parameter=None):
        self.type = gate_type
        self.verify_gate()

        if parameter is not None:
            self.parameter = parameter
        if first_qubit_set is None:
            raise Exception("The acting qubit is not defined")
        self.first_qubit = first_qubit_set
        self.second_qubit = second_qubit_set

    def verify_gate(self):
        if self.type not in gate_set:
            raise Exception("This gate is not defined in set of gate")

    def first_qubit_index(self):
        return self.first_qubit

    def gate_type(self):
        return self.type

    def print(self):
        if self.second_qubit is None:
            print("Gate " + str(self.type) + ", acting qubit: " + str(self.first_qubit))
        else:
            print("Gate " + str(self.type) + ", acting qubit: " \
                  + str(self.first_qubit) + ", control qubit: " + str(self.second_qubit))
