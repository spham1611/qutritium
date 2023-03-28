"""Test pulse file"""
import os.path
import unittest
from src.pulse import Pulse, Pulse01, Pulse12


class TestPulse(unittest.TestCase):
    """Check pulse model"""

    def setUp(self) -> None:
        """Set up two pulse objects"""
        self.pulse01 = Pulse01(frequency=5, x_amp=5,
                               sx_amp=5, duration=5)
        self.pulse12 = Pulse12(frequency=10, x_amp=10,
                               sx_amp=10, duration=10, pulse01=self.pulse01)
        self.pulse01.pulse12 = self.pulse12

    def test_pulse01_constructor(self) -> None:
        # Test if object is created
        self.assertIsInstance(self.pulse01, Pulse01)

        # Check assigned attr
        self.assertEqual(self.pulse01.frequency, 5)
        self.assertEqual(self.pulse01.duration, 5)
        self.assertEqual(self.pulse01.x_amp, 5)
        self.assertEqual(self.pulse01.sx_amp, 5)

    def test_pulse01_inheritance(self) -> None:
        """Check Pulse inheritance"""
        self.assertIsInstance(self.pulse01, Pulse)

    def test_pulse01_eq(self) -> None:
        other_pulse = Pulse01(frequency=5, duration=5,
                              x_amp=5, sx_amp=5)
        self.assertEqual(self.pulse01, other_pulse)

    def test_pulse01_str(self) -> None:
        """Test __str__() special function"""
        test_string = f"Pulse01: id={self.pulse01.id}, frequency={self.pulse01.frequency}, " \
                      f"x_amp={self.pulse01.x_amp}, sx_amp={self.pulse01.sx_amp}, " \
                      f"beta_dephase={self.pulse01.beta_dephase}, beta_leakage={self.pulse01.beta_leakage}, " \
                      f"duration={self.pulse01.duration}, sigma={self.pulse01.sigma}"
        self.assertEqual(self.pulse01.__str__(), test_string)

    def test_pulse01_repr(self) -> None:
        """Test __repr__() special function"""
        test_string = f"Pulse01: id={self.pulse01.id}, frequency={self.pulse01.frequency}, " \
                      f"x_amp={self.pulse01.x_amp}, sx_amp={self.pulse01.sx_amp}, " \
                      f"beta_dephase={self.pulse01.beta_dephase}, beta_leakage={self.pulse01.beta_leakage}, " \
                      f"duration={self.pulse01.duration}, sigma={self.pulse01.sigma}"
        self.assertEqual(self.pulse01.__repr__(), test_string)

    def test_pulse12_constructor(self) -> None:
        # Test if object is created
        self.assertIsInstance(self.pulse12, Pulse12)

        # Check assigned attr
        self.assertEqual(self.pulse12.frequency, 10)
        self.assertEqual(self.pulse12.duration, 10)
        self.assertEqual(self.pulse12.x_amp, 10)
        self.assertEqual(self.pulse12.sx_amp, 10)

    def test_pulse12_inheritance(self) -> None:
        """Check Pulse inheritance"""
        self.assertIsInstance(self.pulse12, Pulse12)

    def test_pulse12_eq(self) -> None:
        other_pulse = Pulse12(frequency=10, duration=10,
                              x_amp=10, sx_amp=10, pulse01=self.pulse01)
        self.assertEqual(self.pulse12, other_pulse)

    def test_pulse12_str(self) -> None:
        """Test __str__() special function"""
        test_string = f"Pulse12: id={self.pulse12.id}, frequency={self.pulse12.frequency}, " \
                      f"x_amp={self.pulse12.x_amp}, sx_amp={self.pulse12.sx_amp}, " \
                      f"beta_dephase={self.pulse12.beta_dephase}, beta_leakage={self.pulse12.beta_leakage}, " \
                      f"duration={self.pulse12.duration}, sigma={self.pulse12.sigma}"
        self.assertEqual(self.pulse12.__str__(), test_string)

    def test_pulse12_repr(self) -> None:
        """Test __repr__() special function"""
        test_string = f"Pulse12: id={self.pulse12.id}, frequency={self.pulse12.frequency}, " \
                      f"x_amp={self.pulse12.x_amp}, sx_amp={self.pulse12.sx_amp}, " \
                      f"beta_dephase={self.pulse12.beta_dephase}, beta_leakage={self.pulse12.beta_leakage}, " \
                      f"duration={self.pulse12.duration}, sigma={self.pulse12.sigma}"
        self.assertEqual(self.pulse12.__repr__(), test_string)

    def test_convert_qiskit_pulse(self) -> None:
        ...


class TestPulseList(unittest.TestCase):
    def setUp(self) -> None:
        """Set up three different pulses for testing"""
        self.pulse_1 = Pulse(duration=144, frequency=16000, x_amp=50, beta_dephase=1.2)
        self.pulse_2 = Pulse01(duration=144, frequency=12000, x_amp=40, beta_leakage=4)
        self.pulse_3 = Pulse12(duration=120, frequency=10000, x_amp=4, sx_amp=20, pulse01=self.pulse_2)
        self.pulse_2.pulse12 = self.pulse_3

    def test_pulse_dictionary(self) -> None:
        """Check if the dictionary has 3 items with accurate values"""
        dict_pulses = Pulse.pulse_list.pulse_dictionary()
        self.assertTrue(all(len(dict_item) == 3 for dict_item in dict_pulses.items()))

        # Check every dictionary element
        self.assertEqual(dict_pulses['pulse id'], [self.pulse_1, self.pulse_2, self.pulse_3])
        self.assertEqual(dict_pulses['mode'], [None, "01", "12"])
        self.assertEqual(dict_pulses['duration'], [144, 144, 120])
        self.assertEqual(dict_pulses['frequency'], [16000, 12000, 10000])
        self.assertEqual(dict_pulses['x_amp'], [50, 40, 4])
        self.assertEqual(dict_pulses['sx_amp'], [0, 0, 20])
        self.assertEqual(dict_pulses['beta_dephase'], [1.2, 0, 0])
        self.assertEqual(dict_pulses['beta_leakage'], [0, 4, 0])
        self.assertEqual(dict_pulses['sigma'], [36, 30, 30])
        self.assertEqual(dict_pulses['pulse_pointer'], [None, self.pulse_3.id, self.pulse_2.id])

    def test_save_pulses(self) -> None:
        """Save in three different types and check their result"""
        test_path = "Test_Pulses"
        # Check if csv is created
        full_path_csv = Pulse.pulse_list.save_pulses(saved_type="csv", file_path=test_path)
        self.assertTrue(os.path.isfile(full_path_csv))
        with open(full_path_csv, 'r') as file:
            self.assertTrue(file.readline().startswith("pulse id, mode, duration, frequency,"
                                                       "x_amp, sx_amp, beta_dephase, beta_leakage,"
                                                       "sigma, pulse_pointer"))

        # Check if json is created
        full_path_json = Pulse.pulse_list.save_pulses(saved_type="json", file_path=test_path)
        self.assertTrue(os.path.isfile(full_path_json))
        with open(full_path_json, 'r') as file:
            self.assertTrue(file.readline().startswith("pulse id, mode, duration, frequency,"
                                                       "x_amp, sx_amp, beta_dephase, beta_leakage,"
                                                       "sigma, pulse_pointer"))

        # Check other format
        with self.assertRaises(IOError):
            Pulse.pulse_list.save_pulses(saved_type="png", file_path=test_path)
