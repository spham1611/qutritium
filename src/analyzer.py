"""

"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from src.constant import QUBIT_PARA
from src.utility import data_mitigatory, reshape_complex_vec, fit_function
from typing import List, Tuple, Callable, Any
import itertools
import numpy as np
import matplotlib.pyplot as plt

QUBIT = QUBIT_PARA.QUBIT.value
SCALE_FACTOR = QUBIT_PARA.SCALE_FACTOR.value


class DataAnalysis:
    """ Use this class for data analysis"""

    def __init__(self, experiment, num_shots=2 ** 14) -> None:
        """__init__ _summary_

        _extended_summary_

        Args:
            experiment (__type__): _description_
            num_shots (int, optional): _description_. Defaults to 2**14.
        """
        self.experiment = experiment
        self.num_shots = num_shots
        self.gfs = None
        self.IQ_data = []
        self.lda_012 = None
        self.score_012 = 0
        self.raw_counted = None
        self.mitiq_data = None
        self.assign_mat = None

    def retrieve_data(self, average: bool) -> None:
        """retrievedData

        Retrieve data from experiment

        Args:
            average ():
        Returns:
            list: IQ_data ?
        """
        experiment_results = self.experiment.result(timeout=120)
        self.IQ_data = []
        for i in range(len(experiment_results.results)):
            if average:
                self.IQ_data.append(np.real(experiment_results.get_memory(i)[0] * SCALE_FACTOR))
            else:
                self.IQ_data.append(experiment_results.get_memory(i)[:, 0] * SCALE_FACTOR)

        self.gfs = [self.IQ_data[0], self.IQ_data[1], self.IQ_data[2]]

    def lda(self) -> None:
        """lda _summary_

        _extended_summary_

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
        self.lda_012 = LinearDiscriminantAnalysis()
        self.lda_012.fit(iq_012_train, state_012_train)
        self.score_012 = self.lda_012.score(iq_012_test, state_012_test)

    def count_pop(self) -> None:
        """countPop _summary_

        _extended_summary_

        Returns:
        """
        iq_reshaped = []
        for i in range(len(self.IQ_data)):
            iq_reshaped.append(reshape_complex_vec(self.IQ_data[i]))
        classified_data = []
        for j in range(len(iq_reshaped)):
            classified_data.append(self.lda_012.predict(iq_reshaped[j]))
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
        self.raw_counted = raw_final_data
        self.assign_mat = self.raw_counted[0:3]

    def error_mitiq(self) -> None:
        """errorMitiq

        This consists of two stages: first is counting; then SPAM error is mitigated
        using NN least-square optimization. For details, see __data_mitigatory()

        Returns:
        """
        self.mitiq_data = []
        for i in range(len(self.raw_counted)):
            self.mitiq_data.append(data_mitigatory(self.raw_counted[i], self.assign_mat))
        self.mitiq_data = np.array(self.mitiq_data)

    def iq_012_plot(self, x_min, x_max, y_min, y_max) -> None:
        """iq_012_plot

        Helper function for plotting IQ plane for 0, 1, 2. Limits of plot given
        as arguments.

        Args:

        Returns:
        """
        zero_data = self.gfs[0]
        one_data = self.gfs[1]
        two_data = self.gfs[2]

        """"""
        # zero data plotted in blue
        plt.scatter(np.real(zero_data), np.imag(zero_data),
                    s=5, cmap='viridis', c='blue', alpha=0.5, label=r'$|0\rangle$')
        # one data plotted in red
        plt.scatter(np.real(one_data), np.imag(one_data),
                    s=5, cmap='viridis', c='red', alpha=0.5, label=r'$|1\rangle$')
        # two data plotted in green
        plt.scatter(np.real(two_data), np.imag(two_data),
                    s=5, cmap='viridis', c='green', alpha=0.5, label=r'$|2\rangle$')

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

    def baseline_remove(self) -> None:
        """
        Normalize the IQ_data
        :return:
        """
        self.IQ_data = np.array(self.IQ_data) - np.mean(self.IQ_data)
        self.IQ_data = np.real(self.IQ_data)

