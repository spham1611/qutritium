from src.pulse import Pulse01, Pulse12
import unittest


class TestTR(unittest.TestCase):
    """Test TR"""

    def setUp(self) -> None:
        """

        :return:
        """
        self.pulse01 = Pulse01(duration=144)
        self.pulse12 = Pulse12(pulse01=self.pulse01, duration=144)

    def test_tr01_constructor(self) -> None: