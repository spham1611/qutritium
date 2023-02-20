"""Test pulse file"""
import unittest
from src.pulse import Pulse


class TestPulse(unittest.TestCase):
    def setUp(self) -> None:
        self.pulse = Pulse()

    def test_pulse_created(self):
        pass
