"""
Backend of the VM that can be used to simulate the Quantum Circuit
"""
import numpy as np
import matplotlib.pyplot as plt
from src.quantumcircuit import Qutrit_circuit
from src.quantumcircuit import statevector_to_state
from src.quantumcircuit import Instruction


class QASM_simulator:
    """
    The class is used to represent a backend simulator in VM,
    A Quantum Circuit is given as input to the backend and the final result is returned.
    """
    def __init__(self, QC: Qutrit_circuit):
        self.circuit = QC
        self.n_qutrit = QC.n_qutrit
        self._measurement_flag = QC.return_meas_flag()
        self._operation_set = QC.return_operation_set()
        self._SPAM_error = None
        self._error_meas = []
        self._measurement_result = []
        self.initial_state = QC.initial_state
        self._error_meas = None
        self.state = self.initial_state

    def add_SPAM_noise(self, p_prep: float, p_meas: float, error_type: str = 'Pauli_error'):
        """
        :param p_prep:  Probability of preparation error
        :param p_meas:  Probability of measurement error
        :param error_type: The type of error
        """
        if error_type == 'Pauli_error':
            self._error_meas = [('x+', p_meas / 2), ('x-', p_meas / 2), ('I', 1 - p_meas)]
            error_prep = [('x+', p_prep / 2), ('x-', p_prep / 2), ('I', 1 - p_prep)]
            """
            Adding preparation error
            """
            probs_prep = [error_prep[i][1] for i in range(len(error_prep))]
            for i in range(self.n_qutrit):
                choice = np.random.choice(range(len(error_prep)), p=probs_prep)
                error_effect = Instruction(gate_type=error_prep[choice][0], n_qutrit=self.n_qutrit, first_qutrit_set=i,
                                           second_qutrit_set=None, parameter=None)
                self._operation_set.insert(__index=0, __object=error_effect)
            """
            Adding measurement error
            """
            self._SPAM_error = True

    def _simulation(self):
        """
        The simulation process of the backend
        """
        if self._measurement_flag:
            for i in range(len(self._operation_set)-1):
                self.state = np.einsum('ij,jk', self._operation_set[i].return_effect(), self.state)
        else:
            for i in self._operation_set:
                self.state = np.einsum('ij,jk', i.return_effect(), self.state)

    def run(self, num_shots: int = 1024):
        """
        :param num_shots: Number of shots
        Performs the defined amount of shots.
        """
        self._simulation()
        if self._measurement_flag:
            if self._SPAM_error:
                probs_prep = [self._error_meas[i][1] for i in range(len(self._error_meas))]
                for i in range(self.n_qutrit):
                    choice = np.random.choice(range(len(self._error_meas)), p=probs_prep)
                    error_effect = Instruction(gate_type=self._error_meas[choice][0], n_qutrit=self.n_qutrit,
                                               first_qutrit_set=i,
                                               second_qutrit_set=None, parameter=None)
                    self._operation_set.insert(__index=0, __object=error_effect)
            state_coeff, state_construction = statevector_to_state(self.state, self.n_qutrit)
            probs = [np.abs(i) ** 2 for i in state_coeff]
            for i in range(num_shots):
                measure = np.random.choice(range(len(state_construction)), p=probs)
                self._measurement_result.append(state_construction[measure])
        else:
            raise Exception("Your circuit does not contains measurement.")

    def get_counts(self):
        """
        :return: count of each state
        """
        if self._measurement_result is not None:
            return dict((x, self._measurement_result.count(x)) for x in set(self._measurement_result))
        else:
            raise Exception("You have not make measurement yet.")

    def return_final_state(self):
        """
        :return: Final state of the quantum circuit
        """
        self._simulation()
        return self.state

    def result(self):
        """
        :return: Measurement result
        """
        if self._measurement_result is not None:
            return self._measurement_result
        else:
            raise Exception("You have not make measurement yet.")

    def density_matrix(self):
        """
        :return: Density matrix of the current state of the quantum circuit
        """
        self._simulation()
        return self.state @ np.transpose(self.state)

    def plot(self, _type: str):
        """
        :param _type: Type of plotting
        Draw graph
        """
        result_dict = self.get_counts()
        if _type == "histogram":
            plt.bar(result_dict.keys(), result_dict.values())
        elif _type == "line":
            plt.plot(result_dict.keys(), result_dict.values())
        elif _type == "dot":
            plt.scatter(result_dict.keys(), result_dict.values())
        plt.show()
