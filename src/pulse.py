"""
Contain pulse class and its example
"""
from functools import singledispatchmethod


class Pulse:
    """
    Our pulse have 5 distinct parameters which can be accessed, shown and saved as a plot
    """
    def __int__(self, frequency, x_amp, sx_amp,
                beta, mode='01', duration=None) -> None:
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
        """
        Frequency getter
        :return: Frequency in Hz unit
        """
        return self._frequency

    @property
    def x_amp(self):
        """
        X_amp getter
        :return: x_amp
        """
        return self._x_amp

    @property
    def sx_amp(self):
        """
        SX_amp getter
        :return: sx_amp
        """
        return self._sx_amp

    @property
    def beta(self):
        """
        Beta getter
        :return: Beta
        """
        return self._beta

    @property
    def mode(self) -> str:
        """
        Mode getter
        :return: mode in string
        """
        return self._mode

    @property
    def duration(self):
        """
        Duration getter
        :return: duration in seconds
        """
        return self._duration

    @property
    def sigma(self):
        """
        Sigma getter
        :return: sigma
        """
        return self._sigma

    def __str__(self):
        """String representation"""
        return f"Pulse(f:{self.frequency}, x_amp:{self.x_amp}, sx_amp:{self.sx_amp}" \
               f"beta:{self.beta}, mode:{self.mode}, duration:{self.duration}, sigma:{self.sigma}"

    def show(self):
        """Show the plot of pulse"""
        pass

    def save(self):
        """Save pulse plot"""
        pass

    @staticmethod
    def convert_to_qiskit_pulse():
        """Convert to qiskit pulse type"""
        pass

