"""
Contain pulse class
"""


class Pulse:
    """
    Our pulse have 5 distinct parameters for us to control and modify
    """
    def __int__(self, frequency, x_amp, sx_amp,
                beta, mode='01', duration=None):
        """

        :param frequency:
        :param x_amp:
        :param sx_amp:
        :param beta:
        :param mode:
        :param duration:
        :return:
        """
        self.frequency = frequency
        self.x_map = x_amp
        self.sx_amp = sx_amp
        self.beta = beta
        self.mode = mode
        self._duration = duration
        self._sigma = duration/4 if duration else 0

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if not 5000 <= value <= 200000:
            raise ValueError("Invalid range of frequency. Should be between (5000, 200000")
        self._frequency = value

    @property
    def x_amp(self):
        return self._x_amp

    @x_amp.setter
    def x_amp(self, value):
        if not 0 <= value <= 200000:
            raise ValueError("Invalid range of x amplitude. Should be between (0, 200000)")
        self._x_amp = value

    @property
    def sx_amp(self):
        return self._sx_amp

    @sx_amp.setter
    def sx_amp(self, value):
        if not 0 <= value <= 200000:
            raise ValueError("Invalid range of sx amplitude. Should be between (0, 200000)")
        self._sx_amp = value

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if not 0 <= value <=2:
            raise ValueError("Invalid range of beta amplitude. Should be between (0, 2)")
        self._beta = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value == '01':
            self._mode = '01'
        elif value == '12':
            self._mode = '12'
        else:
            raise ValueError("Invalid mode of pulse")

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

    @property
    def sigma(self):
        return self._sigma

    def __str__(self):
        return f"Pulse(f:{self.frequency}, x_amp:{self.x_amp}, sx_amp:{self.sx_amp}" \
               f"beta:{self.beta}, mode:{self.mode}, duration:{self.duration}, sigma:{self.sigma}"

