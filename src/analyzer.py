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

"""Analyzing tools"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_provider.ibm_provider import IBMJob

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

from src.constant import QubitParameters
from src.utility import reshape_complex_vec

from scipy.optimize import minimize
from typing import List, Iterable, Union

SCALE_FACTOR = QubitParameters.SCALE_FACTOR.value


class DataAnalysis:
    """ This class provide tools to analyze and the parameters for single qutrit process
    An example of using DataAnalysis object::

        from qutritium.analyzer import DataAnalysis

        data_analysis = DataAnalysis(IBMJob_instance)
        data_analysis.retrieve_data()
        data_analysis.IQ_data

    Notes:
        * Most of our analyzing tools are modification of qiskit analyzing tools used in this reference:
            https://learn.qiskit.org/course/quantum-hardware-pulses/calibrating-qubits-using-qiskit-pulse?fbclid=IwAR3z8d7nhqtbzBnua6aaOD7DktPeW1x1_77QzbGco50o27p0_JY8L2RoH1Y#zerovone

    Here is a list of available attributes "DataAnalysis" class:
        * experiment: Job submitted to IBM Quantum Computer
        * num_shots: number of shots from given experiment
        * IQ_data: y_value of plot iq
        * gfs: hold IQ_data of 3 states: 0, 1, 2
        * retrieve_data():
        * lda():
        * count_pop():
        * error_mitiq():
        * baseline_remove():
        * iq_012_plot(): Plot Discriminator

    """

    def __init__(self, experiment: IBMJob) -> None:
        """ Takes IBMJob and collect its data

        Args:
            experiment:
        """
        self.experiment = experiment
        self.num_shots = self.experiment.backend_options().get('shots')

        # Analytical parameters
        self._IQ_data: List = []
        self._mitiq_data: List = []
        self._gfs: List = []
        self._lda_012 = LinearDiscriminantAnalysis()
        self._score_012: float = 0.
        self._raw_counted = None
        self._assign_mat = None

    @property
    def IQ_data(self) -> List:
        return self._IQ_data

    @property
    def gfs(self) -> List:
        return self._gfs

    def retrieve_data(self, average: bool) -> None:
        """ Retrieve data from experiment

        Args:
            average:

        Returns:

        """
        experiment_results = self.experiment.result(timeout=120)
        for i in range(len(experiment_results.results)):
            check_indices = experiment_results.get_memory(i)
            if len(check_indices) > 1:
                data = check_indices[:, 0] * SCALE_FACTOR
            else:
                data = check_indices[0] * SCALE_FACTOR
            if average:
                self._IQ_data.append(np.real(data))
            else:
                self._IQ_data.append(data)

        self._gfs = [self._IQ_data[0], self._IQ_data[1], self._IQ_data[2]]

    def lda(self) -> None:
        """

        Returns:

        """
        data_reshaped = []
        for i in range(0, 3):
            data_reshaped.append(reshape_complex_vec(self.gfs[i]))

        iq_012_data = list(itertools.chain.from_iterable(data_reshaped))
        state_012 = np.zeros(self.num_shots)
        state_012 = np.concatenate((state_012, np.ones(self.num_shots)))
        state_012 = np.concatenate((state_012, 2 * np.ones(self.num_shots)))

        iq_012_train, iq_012_test, state_012_train, state_012_test = train_test_split(iq_012_data, state_012,
                                                                                      test_size=0.5)
        self._lda_012.fit(iq_012_train, state_012_train)
        self._score_012 = self._lda_012.score(iq_012_test, state_012_test)

    def count_pop(self) -> None:
        """

        Returns:

        """
        iq_reshaped = []
        for i in range(len(self._IQ_data)):
            iq_reshaped.append(reshape_complex_vec(self._IQ_data[i]))
        classified_data = []
        for j in range(len(iq_reshaped)):
            classified_data.append(self._lda_012.predict(iq_reshaped[j]))
        raw_final_data = []
        for k in range(len(classified_data)):
            result = {'0': 0, '1': 0, '2': 0}
            for l in range(len(classified_data[k])):
                if classified_data[k][l] == 0.0:
                    result['0'] += 1
                elif classified_data[k][l] == 1.0:
                    result['1'] += 1
                elif classified_data[k][l] == 2.0:
                    result['2'] += 1
                else:
                    print('Unexpected behavior')
            raw_final_data.append(result)

        raw_final_data = [[raw_final_data[i]['0'] / self.num_shots, raw_final_data[i]['1'] / self.num_shots,
                           raw_final_data[i]['2'] / self.num_shots] for i in
                          range(np.shape(raw_final_data)[0])]
        self._raw_counted = raw_final_data
        self._assign_mat = self._raw_counted[0:3]

    def error_mitiq(self) -> None:
        """ Consists of two stages: first is counting; then SPAM error is mitigated
        using NN least-square optimization. For details, see __data_mitigatory()

        Returns:

        """
        for i in range(len(self._raw_counted)):
            self._mitiq_data.append(DataAnalysis._data_mitigatory(self._raw_counted[i], self._assign_mat))
        self._mitiq_data = np.array(self._mitiq_data)

    def baseline_remove(self) -> None:
        """
        Normalize IQ Data

        Returns:

        """
        self._IQ_data = np.array(self._IQ_data) - np.mean(self._IQ_data)
        self._IQ_data = np.real(self._IQ_data)

    def iq_012_plot(self, x_min: float = -20, x_max: float = 20,
                    y_min: float = -20, y_max: float = 20) -> None:
        """ Helper function for plotting IQ plane for 0, 1, 2. Limits of plot given
        as arguments.

        Args:
            x_min:
            x_max:
            y_min:
            y_max:

        Returns:

        """
        zero_data = self._gfs[0]
        one_data = self._gfs[1]
        two_data = self._gfs[2]
        # state 0 plotted in blue
        plt.scatter(np.real(zero_data), np.imag(zero_data),
                    s=5, c='blue', alpha=0.5, label=r'state_0')
        # state 1 plotted in red
        plt.scatter(np.real(one_data), np.imag(one_data),
                    s=5, c='red', alpha=0.5, label=r'state_1')
        # state 2 plotted in green
        plt.scatter(np.real(two_data), np.imag(two_data),
                    s=5, c='green', alpha=0.5, label=r'state_2')

        # Plot a large dot for the average result of the 0, 1 and 2 states.
        mean_zero = np.mean(zero_data)  # takes mean of both real and imaginary parts
        mean_one = np.mean(one_data)
        mean_two = np.mean(two_data)
        plt.scatter(np.real(mean_zero), np.imag(mean_zero),
                    s=200, cmap='viridis', c='black', alpha=1.0)
        plt.scatter(np.real(mean_one), np.imag(mean_one),
                    s=200, cmap='viridis', c='black', alpha=1.0)
        plt.scatter(np.real(mean_two), np.imag(mean_two),
                    s=200, cmap='viridis', c='black', alpha=1.0)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.ylabel('I [a.u.]', fontsize=15)
        plt.xlabel('Q [a.u.]', fontsize=15)
        plt.title("0-1-2 discrimination", fontsize=15)

    @staticmethod
    def _data_mitigatory(raw_data: Union[List, Iterable],
                         assign_matrix: Union[List, Iterable]):
        """ Normalizes matrix function
        Args:
            raw_data:
            assign_matrix:

        Returns:

        """
        cal_mat = np.transpose(assign_matrix)
        raw_data = raw_data
        num_shots = sum(raw_data)

        def fun(x):
            return sum((raw_data - np.dot(cal_mat, x)) ** 2)

        x0 = np.random.rand(len(raw_data))
        x0 = x0 / sum(x0)
        cons = {"type": "eq",
                "fun": lambda x: num_shots - sum(x)}
        bounds = tuple((0, num_shots) for _ in x0)
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bounds, tol=1e-6)
        data_mitigated = res.x

        return data_mitigated
