"""
Contain pulse class and its example
"""
from functools import singledispatchmethod


class Pulse:
    """
    Our pulse have 5 distinct parameters which can be accessed and shown separately
    """
    def __int__(self, frequency, x_amp, sx_amp,
                beta, mode='01', duration=None):
        """

        :param frequency: Hz
        :param x_amp:
        :param sx_amp:
        :param beta:
        :param mode: There are two modes only 01 and 12
        :param duration:
        :return:
        """
        self._frequency = frequency
        self._x_amp = x_amp
        self._sx_amp = sx_amp
        self._beta = beta
        self._mode = mode
        self._duration = duration
        self._sigma = duration/4 if duration else 0

    @property
    def frequency(self):
        return self._frequency

    @property
    def x_amp(self):
        return self._x_amp

    @property
    def sx_amp(self):
        return self._sx_amp

    @property
    def beta(self):
        return self._beta

    @property
    def mode(self):
        return self._mode

    @property
    def duration(self):
        return self._duration

    @property
    def sigma(self):
        return self._sigma

    def __str__(self):
        return f"Pulse(f:{self.frequency}, x_amp:{self.x_amp}, sx_amp:{self.sx_amp}" \
               f"beta:{self.beta}, mode:{self.mode}, duration:{self.duration}, sigma:{self.sigma}"

    def show(self):
        pass

    def save(self):
        pass

    @staticmethod
    def convert_to_qiskit_pulse():
        pass

