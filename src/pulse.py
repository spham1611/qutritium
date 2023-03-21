"""
Contain pulse classes
"""
from __future__ import annotations
from abc import ABC
from typing import Dict, Optional
import os
import json
import pandas as pd


class Pulse_List(list["Pulse"]):
    """List of pulses which in turn can be saved in csv or test files"""

    def pulse_dictionary(self) -> Dict:
        """Convert list of pulse to dictionary"""
        dict_pulses = {'mode': [],
                       'duration': [],
                       'frequency': [],
                       'x_amp': [],
                       'sx_amp': [],
                       'beta_dephase': [],
                       'beta_leakage': [],
                       'sigma': [],
                       }
        for pulse in self:
            if isinstance(pulse, Pulse01):
                dict_pulses['mode'].append("01")
            elif isinstance(pulse, Pulse12):
                dict_pulses['mode'].append("12")
            else:
                raise ValueError("Invalid pulse state!")
            dict_pulses['duration'].append(pulse.duration)
            dict_pulses['frequency'].append(pulse.frequency)
            dict_pulses['x_amp'].append(pulse.x_amp)
            dict_pulses['sx_amp'].append(pulse.sx_amp)
            dict_pulses['beta_dephase'].append(pulse.beta_dephase)
            dict_pulses['beta_leakage'].append(pulse.beta_leakage)
            dict_pulses['sigma'].append(pulse.sigma)

        return dict_pulses

    def save_files(self, saved_type: str, file_path: str = "Pulses") -> None:
        """
        Save list of pulses in csv using panda.DataFrame
        :param saved_type:
        :param file_path:
        :return:
        """
        dict_pulses = self.pulse_dictionary()
        if saved_type == 'csv':
            # Save CSV
            save_pulses_df = pd.DataFrame(dict_pulses)
            full_path = os.path.join(file_path, ".csv")
            save_pulses_df.to_csv(full_path, index=False)
            if os.path.isfile(full_path):
                print("Save the csv in output folder successfully!")
            else:
                print("There is a problem with saving the file!")
        elif saved_type == "json":
            # Save JSON
            json_pulse = json.dumps(dict_pulses, indent=4)
            full_path = os.path.join(file_path, ".json")
            with open(full_path, "w") as outfile:
                outfile.write(json_pulse)
        else:
            raise IOError("Unsupported type!")


class Pulse(ABC):
    """
    Our pulse have 5 distinct parameters which can be accessed, shown and saved as a plot.
    For developers, we attempt to use these variables as inner variables only
    """
    pulse_list = Pulse_List()

    def __init__(self, frequency=0, x_amp=0, sx_amp=0,
                 beta_dephase: int = 0, beta_leakage: int = 0, duration=0) -> None:
        """

        :param frequency:
        :param x_amp:
        :param sx_amp:
        :param beta_dephase:
        :param beta_leakage:
        :param duration:
        """
        self.frequency = frequency
        self.x_amp = x_amp
        self.sx_amp = sx_amp
        self.beta_leakage = beta_leakage
        self.beta_dephase = beta_dephase
        self.duration = duration
        self.sigma = duration / 4 if duration else 0
        Pulse.pulse_list.append(self)

    @staticmethod
    def convert_to_qiskit_pulse():
        """Convert to qiskit pulse type"""
        pass


class Pulse01(Pulse):
    """Pulse of 0 -> 1 state"""

    def __init__(self, frequency=0, x_amp=0, sx_amp=0,
                 beta_dephase: int = 0, beta_leakage: int = 0, duration=0,
                 pulse12: Pulse12 = None) -> None:
        """

        :param frequency:
        :param x_amp:
        :param sx_amp:
        :param beta_dephase:
        :param beta_leakage:
        :param duration:
        :param pulse12: point to related 12 state
        """
        self.pulse12 = pulse12
        super().__init__(frequency=frequency, x_amp=x_amp, sx_amp=sx_amp,
                         beta_dephase=beta_dephase, beta_leakage=beta_leakage, duration=duration)

    def __str__(self) -> str:
        return (
            f"Pulse01: frequency={self.frequency}, x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        return (
            f"Pulse01: frequency={self.frequency}, x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: "Pulse01") -> bool:
        return (
                self.frequency == other.frequency
                and self.x_amp == other.x_amp
                and self.sx_amp == other.sx_amp
                and self.beta_leakage == other.beta_leakage
                and self.beta_dephase == other.beta_dephase
                and self.duration == other.duration
        )

    def is_pulse12_there(self) -> bool:
        """

        :return:
        """
        return self.pulse12 is not None


class Pulse12(Pulse):
    """Pulse of 1 -> 2 state"""

    def __init__(self, frequency=0, x_amp=0, sx_amp=0,
                 beta_dephase: int = 0, beta_leakage: int = 0, duration: int = 0,
                 pulse01: Pulse01 = None) -> None:
        """

        :param frequency:
        :param x_amp:
        :param sx_amp:
        :param beta_dephase:
        :param beta_leakage:
        :param duration:
        :param pulse01: Not allowed to be None
        """
        if pulse01 is None:
            raise ValueError("Pulse 12 must have its related 01 state")
        self.pulse01 = pulse01
        super().__init__(frequency=frequency, x_amp=x_amp, sx_amp=sx_amp,
                         beta_dephase=beta_dephase, beta_leakage=beta_leakage, duration=duration)

    def __str__(self) -> str:
        return (
            f"Pulse12: frequency={self.frequency}, x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        return (
            f"Pulse12: frequency={self.frequency}, x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: "Pulse12") -> bool:
        return (
                self.frequency == other.frequency
                and self.x_amp == other.x_amp
                and self.sx_amp == other.sx_amp
                and self.beta_leakage == other.beta_leakage
                and self.beta_dephase == other.beta_dephase
                and self.duration == other.duration
        )

    def is_pulse01_there(self) -> bool:
        """

        :return:
        """
        return self.pulse01 is not None
