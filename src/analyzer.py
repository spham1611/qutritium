"""
Analyzing Tools
"""
import itertools
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_provider.ibm_provider import IBMJob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from src.constant import QUBIT_PARA
from src.utility import reshape_complex_vec
from scipy.optimize import minimize
from typing import List, Iterable, Optional, Any, Union

SCALE_FACTOR = QUBIT_PARA.SCALE_FACTOR.value
plt.rcParams['savefig.dpi'] = 300


class DataAnalysis:
    """ Use this class for data analysis"""

    def __init__(self, experiment: IBMJob, num_shots: int = 2 ** 14) -> None:
        """
        Basic protocol that is used to analyze pulse characteristics
        :param experiment: IBMJob
        :param num_shots: usually power of 2
        """
        self.experiment = experiment
        self.num_shots = num_shots

        # Analytical parameters
        self._IQ_data: List = [Any]
        self._mitiq_data: List = [Any]
        self._gfs: List = []
        self._lda_012 = LinearDiscriminantAnalysis()
        self._score_012: float = 0.
        self._raw_counted = None
        self._assign_mat = None

    @property
    def IQ_data(self) -> List:
        return self._IQ_data

    @property
    def mitiq_data(self) -> List:
        return self._mitiq_data

    def retrieve_data(self, average: bool) -> None:
        """
        Retrieve data from IBM job
        :param average:
        :return:
        """
        experiment_results = self.experiment.result(timeout=120)
        for i in range(len(experiment_results.results)):
            if average:
                self._IQ_data.append(np.real(experiment_results.get_memory(i)[0] * SCALE_FACTOR))
            else:
                self._IQ_data.append(experiment_results.get_memory(i)[:, 0] * SCALE_FACTOR)

        self._gfs = [self._IQ_data[0], self._IQ_data[1], self._IQ_data[2]]

    def lda(self) -> None:
        """

        :return:
        """
        data_reshaped = []
        for i in range(0, 3):
            data_reshaped.append(reshape_complex_vec(self._gfs[i]))

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
        Count the population

        :return:
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
        """errorMitiq

        This consists of two stages: first is counting; then SPAM error is mitigated
        using NN least-square optimization. For details, see __data_mitigatory()

        :return:
        """
        for i in range(len(self._raw_counted)):
            self._mitiq_data.append(DataAnalysis._data_mitigatory(self._raw_counted[i], self._assign_mat))
        self._mitiq_data = np.array(self._mitiq_data)

    def baseline_remove(self) -> None:
        """
        Normalize the IQ_data

        :return:
        """
        self._IQ_data = np.array(self._IQ_data) - np.mean(self._IQ_data)
        self._IQ_data = np.real(self._IQ_data)

    def iq_012_plot(self, x_min: float, x_max: float,
                    y_min: float, y_max: float) -> None:
        """iq_012_plot

        Helper function for plotting IQ plane for 0, 1, 2. Limits of plot given
        as arguments.
        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        zero_data = self._gfs[0]
        one_data = self._gfs[1]
        two_data = self._gfs[2]

        """"""
        # zero data plotted in blue
        plt.scatter(np.real(zero_data), np.imag(zero_data),
                    s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\r angle$')
        # one data plotted in red
        plt.scatter(np.real(one_data), np.imag(one_data),
                    s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\r angle$')
        # two data plotted in green
        plt.scatter(np.real(two_data), np.imag(two_data),
                    s=5, cmap='viridis', c='green', alpha=0.5, label=r'$|2\r angle$')

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
    def plot_svg(x_values: Iterable,
                 y_values: Iterable,
                 title: str = '',
                 x_label: str = '',
                 y_label: str = '',
                 plot_name: str = '',
                 **kwargs) -> None:

        """
        Plot the matplotlib and save it in output folder
        :param x_values: List or np.ndarray
        :param y_values: List or np.ndarray
        :param title: title of the plot
        :param x_label:
        :param y_label:
        :param plot_name:
        :return:
        """
        for x_list, y_list, label, color, marker, size, width, style, edge_color in zip(x_values,
                                                                                        y_values,
                                                                                        kwargs['labels'],
                                                                                        kwargs['colors'],
                                                                                        kwargs['markers'],
                                                                                        kwargs['markersizes'],
                                                                                        kwargs['widths'],
                                                                                        kwargs['styles'],
                                                                                        kwargs['edge_colors']):
            plt.scatter(x=x_list, y=y_list, label=label, color=color,
                        marker=marker, markersize=size, linewidth=width, linestyle=style,
                        markeredgecolor=edge_color)
        plt.title(title, fontsize=15)
        plt.xlabel(x_label, fontsize=15)
        plt.ylabel(y_label, fontsize=15)
        plt.legend(fontsize=15)
        plt.savefig(plot_name, format='svg')
        plt.show()

    @staticmethod
    def _data_mitigatory(raw_data: Union[List, Iterable],
                         assign_matrix: Union[List, Iterable]):
        """
        Normalize matrix function
        :param raw_data:
        :param assign_matrix:
        :return:
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
        bounds = tuple((0, num_shots) for x in x0)
        res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bounds, tol=1e-6)
        data_mitigated = res.x

        return data_mitigated
