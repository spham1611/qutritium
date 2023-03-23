import typing
from typing import get_type_hints
import numpy as np
import matplotlib.pyplot as plt
from VM_utility import print_statevector
from VM_utility import statevector_to_state
from instruction_structure import Instruction


class Virtual_Machine:
    def __init__(self, n_qutrit=int, n_classical=int, initial_state=[int]):
        """
        :param n_qutrit: Number of qutrit
        :param n_classical: Number of classical bits
        :param initial_state: The initial state of the quantum circuit
        """
        self.measurement_result = None
        self.measurement_flag = False
        self.n_qutrit = n_qutrit
        self.n_classical = n_classical
        self.operation_set = []
        self.state = []
        self.SPAM_error = False
        self.error_meas = None
        if initial_state is not None:
            if initial_state.shape == (3 ** n_qutrit, 1):
                self.initial_state = initial_state
                self.state = initial_state
            else:
                raise Exception(
                    "The initial state declared does not have correct dimension. The current shape is " + str(
                        initial_state.shape))
        else:
            self.initial_state = np.array([[1], [0], [0]])
            for i in range(self.n_qutrit - 1):
                self.initial_state = np.kron(self.initial_state, np.array([[1], [0], [0]]))
            self.state = self.initial_state

    # def __final_state_calculation(self):
    #     """
    #     :return: Final state as tensor product of all qutrit states
    #     """
    #     for i in range(self.qutrit_state.keys()):
    #         if i == 0:
    #             self.state = self.qutrit_state["qutrit_0"]
    #         else:
    #             self.state = np.kron(self.state, self.qutrit_state["qutrit_" + str(i)])

    def add_gate(self, gate_type, first_qutrit_set, second_qutrit_set=None, parameter=None):
        """
        :param gate_type: quantum gate type as define in gate_set
        :param first_qutrit_set: acting qubits
        :param second_qutrit_set: control qubits
        :param parameter: parameter of rotation gate (if needed)
        :return:
        """
        if gate_type != 'measure':
            ins = Instruction(gate_type, self.n_qutrit, self.state, first_qutrit_set, second_qutrit_set,
                              parameter)
            self.operation_set.append(ins)
            self.state = ins.return_effect()
        else:
            self.measurement_flag = True
            self.operation_set.append("measurement")

    def run(self, num_shots=1024):
        """
        :param num_shots: Number of shots
        Performs the defined amount of shots.
        """
        if self.measurement_flag:
            if self.SPAM_error:
                probs_prep = [self.error_meas[i][1] for i in range(len(self.error_meas))]
                for i in range(self.n_qutrit):
                    choice = np.random.choice(range(len(self.error_meas)), p=probs_prep)
                    self.add_gate(self.error_meas[choice][0], i)
            state_coeff, state_construction = statevector_to_state(self.state, self.n_qutrit)
            probs = [np.abs(i) ** 2 for i in state_coeff]
            self.measurement_result = []
            for i in range(num_shots):
                measure = np.random.choice(range(len(state_construction)), p=probs)
                self.measurement_result.append(state_construction[measure])
        else:
            raise Exception("Your circuit does not contains measurement.")

    def get_counts(self):
        """
        :return: count of each state
        """
        if self.measurement_result is not None:
            return dict((x, self.measurement_result.count(x)) for x in set(self.measurement_result))
        else:
            raise Exception("You have not make measurement yet.")

    def plot(self, type):
        """
        :param type: Type of plotting
        Draw graph
        """
        result_dict = self.get_counts()
        if type == "histogram":
            plt.bar(result_dict.keys(), result_dict.values())
        elif type == "line":
            plt.plot(result_dict.keys(), result_dict.values())
        elif type == "dot":
            plt.scatter(result_dict.keys(), result_dict.values())
        plt.show()

    def return_final_state(self):
        """
        :return: Final state of the quantum circuit
        """
        return self.state

    def result(self):
        """
        :return: Measurement result
        """
        if self.measurement_result is not None:
            return self.measurement_result
        else:
            raise Exception("You have not make measurement yet.")

    def density_matrix(self):
        """
        :return: Density matrix of the current state of the quantum circuit
        """
        return self.state@np.transpose(self.state)

    def add_SPAM_noise(self, p_prep, p_meas, error_type = 'Pauli_error'):
        if error_type == 'Pauli_error':
            self.error_meas = [('x+', p_meas/2), ('x-', p_meas/2), ('I', 1-p_meas)]
            error_prep = [('x+', p_prep/2), ('x-', p_prep/2), ('I', 1-p_prep)]
            """
            Adding preparation error
            """
            probs_prep = [error_prep[i][1] for i in range(len(error_prep))]
            for i in range(self.n_qutrit):
                choice = np.random.choice(range(len(error_prep)), p=probs_prep)
                self.add_gate(error_prep[choice][0], i)
            """
            Adding measurement error
            """
            self.SPAM_error = True

    def draw(self):
        """
        Representation of the quantum circuit
        """
        print("Initial state of the circuit: ")
        print_statevector(self.initial_state, self.n_qutrit)
        print("Final state of the circuit: ")
        print_statevector(self.state, self.n_qutrit)
        print("Set of gate on the circuits: ")
        for i in self.operation_set:
            if type(i) == Instruction:
                i.print()
            else:
                print(i)
