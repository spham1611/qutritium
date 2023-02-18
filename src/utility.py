"""Utility functions to analyze data"""
import itertools
import numpy as np
import qiskit
import matplotlib.pyplot as plt

from constant import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.optimize import minimize


class DataAnalysis:
    """ Use this class for data analysis"""
    def __init__(self, experiment, average=False, num_shots=2**14) -> None:
        """__init__ _summary_

        _extended_summary_

        Args:
            experiment (__type__): _description_
            analyze (bool, optional): _description_. Defaults to True.
            average (bool, optional): _description_. Defaults to False.
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
        
    def reshape_complex_vec(self, vec: np.ndarray) -> np:
        """reshape_complex_vec 
        
        Take in complex vector vec and return 2d array w/ real, imag entries. This is needed for the learning.
        
        Args:
            vec (np): complex vector of data
        Returns:
            np: vector w/ entries given by (real(vec], imag(vec))
        """
        length = len(vec)
        vec_reshaped = np.zeros((length, 2))
        for i in range(len(vec)):
            vec_reshaped[i] = [np.real(vec[i]), np.imag(vec[i])]
        return vec_reshaped
    
    def retrieve_data(self, average: bool) -> None:
        """retrivedData
        
        Retrive data from experiment
        
        Args:
            experiment ():
            average ():
        Returns:
            list: IQ_data ?
        """ 
        experiment_results = self.experiment.result(timeout=120)
        self.IQ_data = []
        for i in range(len(experiment_results.results)):
            if average:  
                self.IQ_data.append(np.real(experiment_results.get_memory(i)[QUBIT] * SCALE_FACTOR))
            else: 
                self.IQ_data.append(experiment_results.get_memory(i)[:, QUBIT] * SCALE_FACTOR)  

        self.gfs = [self.IQ_data[0], self.IQ_data[1], self.IQ_data[2]]

    def lda(self):
        """lda _summary_

        _extended_summary_

        Args:
            discrim_data (_type_): _description_
            num_shots (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_reshaped = []
        for i in range(0,3):
            data_reshaped.append(self.reshape_complex_vec(self.gfs[i]))

        IQ_012_data = list(itertools.chain.from_iterable(data_reshaped))
        state_012 = np.zeros(self.num_shots)
        state_012 = np.concatenate((state_012, np.ones(self.num_shots)))
        state_012 = np.concatenate((state_012, 2*np.ones(self.num_shots)))

        IQ_012_train, IQ_012_test, state_012_train, state_012_test = train_test_split(IQ_012_data, state_012, test_size=0.5)
        self.lda_012 = LinearDiscriminantAnalysis()
        self.lda_012.fit(IQ_012_train, state_012_train)
        self.score_012 = self.lda_012.score(IQ_012_test, state_012_test)

    def count_pop(self):
        """countPop _summary_

        _extended_summary_

        Args:
            raw_data (_type_): _description_
            LDA (_type_): _description_
            num_shots (_type_): _description_

        Returns:
            _type_: _description_
        """
        IQreshaped = []
        for i in range(len(self.IQ_data)):
            IQreshaped.append(self.reshape_complex_vec(self.IQ_data[i]))
        classified_data = []
        for j in range(len(IQreshaped)):
            classified_data.append(self.lda_012.predict(IQreshaped[j]))
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
        
        raw_final_data = [[raw_final_data[i]['0']/self.num_shots, raw_final_data[i]['1']/self.num_shots, raw_final_data[i]['2']/self.num_shots] for i in
                    range(np.shape(raw_final_data)[0])] 
        self.raw_counted = raw_final_data
        self.assign_mat = self.raw_counted[0:3]

    def error_mitiq(self):
        """errorMitiq 
        
        This consists of two stages: first is counting; then SPAM error is mitigated
        using NN least-square optimization. For details, see __dataMitigator()

        Args:
            data (_type_): _description_
            assign_matrix (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.mitiq_data = []
        for i in range(len(self.raw_counted)):
            self.mitiq_data.append(data_mitigator(self.raw_counted[i], self.assign_mat))
        self.mitiq_data = np.array(self.mitiq_data)

    def iq_012_plot(self, x_min, x_max, y_min, y_max):
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
        mean_zero = np.mean(zero_data) # takes mean of both real and imaginary parts
        mean_one = np.mean(one_data)
        mean_two = np.mean(two_data)
        plt.scatter(np.real(mean_zero), np.imag(mean_zero), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.scatter(np.real(mean_one), np.imag(mean_one), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        plt.scatter(np.real(mean_two), np.imag(mean_two), 
                    s=200, cmap='viridis', c='black',alpha=1.0)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min,y_max)
        plt.legend()
        plt.ylabel('I [a.u.]', fontsize=15)
        plt.xlabel('Q [a.u.]', fontsize=15)
        plt.title("0-1-2 discrimination", fontsize=15)
        plt.savefig('output/iq_plane_plot.svg')
        
    def average_counter(counts, num_shots):
        all_exp = []
        for j in counts:
            zero = 0
            for i in j.keys():
                if i[-1] == "0":
                    zero += j[i]
            all_exp.append(zero)
            
        return np.array(all_exp)/num_shots


# Fitting functions
def fit_function(x_values, y_values, function, init_params):
    """fit_function _summary_

    _extended_summary_

    Args:
        x_values (_type_): _description_
        y_values (_type_): _description_
        function (_type_): _description_
        init_params (_type_): _description_

    Returns:
        _type_: _description_
    """
    fitparams, _ = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit

def baseline_remove(values):
    return np.array(values) - np.mean(values)

def average_counter(counts, num_shots):
    all_exp = []
    for j in counts:
        zero = 0
        for i in j.keys():
            if i[-1] == "0":
                zero += j[i]
        all_exp.append(zero)
    return np.array(all_exp)/num_shots

def data_mitigator(raw_data, assign_matrix):
    """data_mitigator _summary_

    _extended_summary_

    Returns:
        _type_: _description_
    """
    cal_mat = np.transpose(assign_matrix)
    raw_data = raw_data
    nshots = sum(raw_data)

    def fun(x):
        return sum((raw_data - np.dot(cal_mat, x)) ** 2)

    x0 = np.random.rand(len(raw_data))
    x0 = x0 / sum(x0)
    cons = {"type": "eq",
            "fun": lambda x: nshots - sum(x)}
    bnds = tuple((0, nshots) for x in x0)
    res = minimize(fun, x0, method="SLSQP", constraints=cons, bounds=bnds, tol=1e-6)
    data_mitigated = res.x

    return data_mitigated