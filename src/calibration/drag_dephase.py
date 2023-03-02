"""Drag techniques"""
from abc import ABC, abstractmethod


class Drag(ABC):
    """
    The class act as provider + regulator for Drag techniques
    """
    def __int__(self):
        pass

    @abstractmethod
    def drag_method(self):
        pass


class Drag01(Drag):
    def drag_method(self):
        pass


class Drag12(Drag):
    def drag_method(self):
        pass