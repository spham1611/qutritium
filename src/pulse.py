"""
Contain pulse classes
"""
from __future__ import annotations
from abc import ABC
from typing import Dict
import tkinter as tk
import os
import json
import numpy as np
import pandas as pd
import uuid


class Pulse_List(list["Pulse"]):
    """List of pulses which in turn can be saved in csv or text files

    """

    def pulse_dictionary(self) -> Dict:
        """Convert list of pulse to dictionary -> Tabulate in other formats
        :return: Dictionary contains all pulses
        """
        dict_pulses = {'pulse id': [],
                       'mode': [],
                       'duration': [],
                       'frequency': [],
                       'x_amp': [],
                       'sx_amp': [],
                       'beta_dephase': [],
                       'beta_leakage': [],
                       'sigma': [],
                       'pulse_pointer': [],
                       }
        for pulse in self:
            pulse: Pulse
            dict_pulses['pulse id'].append(pulse.id)
            if isinstance(pulse, Pulse01):
                pulse: Pulse01
                dict_pulses['mode'].append("01")
                dict_pulses['pulse_pointer'].append(pulse.pulse12.id if pulse.pulse12 else None)
            elif isinstance(pulse, Pulse12):
                pulse: Pulse12
                dict_pulses['mode'].append("12")
                dict_pulses['pulse_pointer'].append(pulse.pulse01.id)
            else:
                dict_pulses['mode'].append(None)
                dict_pulses['pulse_pointer'].append(None)
            dict_pulses['duration'].append(pulse.duration)
            dict_pulses['frequency'].append(pulse.frequency)
            dict_pulses['x_amp'].append(pulse.x_amp)
            dict_pulses['sx_amp'].append(pulse.sx_amp)
            dict_pulses['beta_dephase'].append(pulse.beta_dephase)
            dict_pulses['beta_leakage'].append(pulse.beta_leakage)
            dict_pulses['sigma'].append(pulse.sigma)

        return dict_pulses

    def save_pulses(self, saved_type: str, file_name: str = "pulses") -> None:
        """
        Save list of pulses in csv using panda.DataFrame and json using python standard library
        Save file in output path which is in qutrit/output
        :param saved_type: currently supported .txt and .csv
        :param file_name:
        :return: None
        :raises:
            IOError: Can not save the given format
        """
        dict_pulses = self.pulse_dictionary()
        # Get the current directory of the script
        file_path = os.path.abspath(__file__).split("\\")[:-2]
        file_path = "\\".join(file_path)
        file_path = os.path.join(file_path, "output")
        if saved_type == 'csv':
            # Save CSV
            save_pulses_df = pd.DataFrame(dict_pulses)
            save_pulses_df['mode'] = save_pulses_df['mode'].apply('="{}"'.format)
            print(save_pulses_df['mode'])
            full_path = file_path + f"\\{file_name}" + ".csv"
            save_pulses_df.to_csv(full_path, index=False, )
        elif saved_type == "json":
            # Save JSON
            json_pulse = json.dumps(dict_pulses, indent=4)
            full_path = file_path + f"\\{file_name}" + ".json"
            with open(full_path, "w") as outfile:
                outfile.write(json_pulse)
        else:
            raise IOError("Unsupported type!")


class Pulse(ABC):
    """
    Our pulse have 5 characteristics of a physical pulse in quantum computer.
    These 5 characteristics are: frequency, duration of the pulse, x_amp which is the max amplitude of the pulse,
    beta dephase and beta leakage which are parameters for system intrinsic noise


    """
    pulse_list = Pulse_List()

    def __init__(self, frequency: float, x_amp: float, sx_amp: float,
                 beta_dephase: float, beta_leakage: float, duration: int) -> None:
        """ Automatically add pulse to a list which can be exported -> See PulseList class for more details

        :param frequency: in Hz
        :param x_amp:
        :param sx_amp: default = x_amp / 2
        :param beta_dephase:
        :param beta_leakage:
        :param duration: in milliseconds
        :raise:
            ValueError: if the pulse does not have duration, frequency and x_amp
        """
        if 4 * 1e9 <= frequency <= 6 * 1e9:
            raise ValueError("Invalid frequency. The frequency for a typical pulse should be btw 4 to 6 GHz")
        if not duration or duration <= 0:
            raise ValueError("Time must be >= 0")
        if not x_amp:
            raise ValueError("Pulse must have amplitude")
        self.frequency = frequency
        self.x_amp = x_amp
        self.sx_amp = sx_amp if sx_amp else self.x_amp / 2
        self.beta_leakage = beta_leakage
        self.beta_dephase = beta_dephase
        self.duration = duration
        self.sigma = duration / 4 if duration else 0
        self.id = uuid.uuid4()
        Pulse.pulse_list.append(self)

    def draw(self, canvas_width: int = 600,
             canvas_height: int = 400, time_destroy: int = 5000) -> None:
        """
        Draw pulse as sin waveform using tkinter. Remember that this is for visualization only
        :param canvas_width:
        :param canvas_height:
        :param time_destroy: Time delay -> close the window and stop showing the plot
        :return: None
        """
        root = tk.Tk()
        root.title(f"Sine Wave of {self.__class__.__name__}")

        # Create a canvas widget
        canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
        canvas.pack()

        # Draw the pulse in sine wave form
        x_start = 0
        y_start = canvas_height // 2
        x_end = canvas_width
        num_points = 1000
        for i in range(num_points):
            x = x_start + (i / num_points) * (x_end - x_start)
            y = y_start - self.x_amp * np.sin(2 * np.pi * self.frequency * (i / num_points))
            canvas.create_line(x, y, x + 1, y + 1, fill='black')
        root.after(time_destroy, root.destroy)
        root.mainloop()


class Pulse01(Pulse):
    """Pulse that represents 0 -> 1 state. The pulse can stand alone or go with Pulse that represents 1 -> 2 state
    """

    def __init__(self, frequency: float = 0, x_amp: float = 0.2, sx_amp: float = 0,
                 beta_dephase: float = 0, beta_leakage: float = 0, duration: int = 144,
                 pulse12: Pulse12 = None) -> None:
        """
        It depends on types of quantum computer that the frequency may vary.
        However, we typically get the frequency to be around 5.1 to 5.2 GHz
        :param frequency: in Hz
        :param x_amp:
        :param sx_amp: default x_amp / 2
        :param beta_dephase:
        :param beta_leakage:
        :param duration: in milliseconds
        :param pulse12: Pulse12 related to this pulse
        """
        super().__init__(frequency=frequency, x_amp=x_amp, sx_amp=sx_amp,
                         beta_dephase=beta_dephase, beta_leakage=beta_leakage, duration=duration)
        self.pulse12 = pulse12

    def __str__(self) -> str:
        """
        The representation in string
        .. code-block:: python
            print(pulse01)
        :return: str
        """
        return (
            f"Pulse01: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        """

        :return: str
        """
        return (
            f"Pulse01: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: Pulse01) -> bool:
        """
        Two pulses are equal if they share all characteristics
        :param other: Pulse01
        :return: bool: True if pulses are equal
        """
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

        :return: bool: if there is a corresponding pulse12
        """
        return self.pulse12 is not None


class Pulse12(Pulse):
    """ Pulse that represents 1 -> 2 state. The pulse must go with its corresponding Pulse01
    """

    def __init__(self, pulse01: Pulse01, frequency: float = 0, x_amp: float = 0.2, sx_amp: float = 0,
                 beta_dephase: float = 0, beta_leakage: float = 0, duration: int = 144,
                 ) -> None:
        """
        It depends on types of quantum computer that the frequency may vary.
        However, we typically get the frequency to be around 4.8 to 4.9 GHz. This is because the anharmonicity of a
        typical quantum computer is around 0.3 GHz -> f1 = f2 + anharmonicity => f2 is around 4.9GHz

        :param frequency: in Hz
        :param x_amp:
        :param sx_amp: default x_amp / 2
        :param beta_dephase:
        :param beta_leakage:
        :param duration: in milliseconds
        :param pulse01: Not allowed to be None
        """
        super().__init__(frequency=frequency, x_amp=x_amp, sx_amp=sx_amp,
                         beta_dephase=beta_dephase, beta_leakage=beta_leakage, duration=duration)
        self.pulse01 = pulse01
        self.pulse01.pulse12 = self

    def __str__(self) -> str:
        """
            The representation in string
            .. code-block:: python
                print(pulse12)
        :return: str
        """
        return (
            f"Pulse12: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        """

        :return: str
        """
        return (
            f"Pulse12: id={self.id}, frequency={self.frequency},"
            f" x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"beta_dephase={self.beta_dephase}, beta_leakage={self.beta_leakage}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: Pulse12) -> bool:
        """
        Check if two pulses are the same. They must share or have the same 'pulse01'
        :param other: Pulse12
        :return: bool: If pulses are equal
        """
        return (
                self.frequency == other.frequency
                and self.x_amp == other.x_amp
                and self.sx_amp == other.sx_amp
                and self.beta_leakage == other.beta_leakage
                and self.beta_dephase == other.beta_dephase
                and self.duration == other.duration
                and self.pulse01 == other.pulse01
        )

