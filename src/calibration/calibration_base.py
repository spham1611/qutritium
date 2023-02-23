"""Abstract class -> create contract for other calibration types"""

from abc import ABC, abstractmethod


class Calibration_Base(ABC):
    """"""

    def __init__(self):
        """

        """
        self._duration = None

    def __int__(self, duration):
        """

        :param duration:
        :return:
        """
        self._duration = duration

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        if value <= 0:
            raise ValueError("Invalid value of duration")
        self._duration = value

    @abstractmethod
    def t_drag(self):
        raise NotImplementedError

    @abstractmethod
    def rough_rabi(self):
        raise NotImplementedError

    @abstractmethod
    def fine_tune(self):
        raise NotImplementedError

    @abstractmethod
    def drag(self):
        raise NotImplementedError
