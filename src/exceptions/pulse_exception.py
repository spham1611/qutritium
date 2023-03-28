"""Exceptions for pulse model"""
from src.pulse import Pulse


class MissingFrequencyPulse(Exception):
    """Raise if the pulse does not have frequency"""
    pass


class MissingDurationPulse(Exception):
    """Raise if the pulse does not have duration"""
    pass


class MissingAmplitudePulse(Exception):
    """Raise if the pulse does not have x_amp"""
    pass
