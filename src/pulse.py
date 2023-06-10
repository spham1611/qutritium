# MIT License
#
# Copyright (c) [2023] [son pham, tien nguyen, bach bao]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pulse model and Pulse list"""

from __future__ import annotations
import tkinter as tk
import os
import json
import numpy as np
import pandas as pd
import uuid
from abc import ABC
from typing import Dict
from src.exceptions.pulse_exception import MissingDurationPulse, MissingAmplitudePulse


class PulseList(list["Pulse"]):
    """ List of pulses which in turn can be saved in csv or text files

    Notes:
        * This class is attached to ''Pulse'' class and should not be instantiated

    Here is list of available functions in "Pulse_List" class
        * pulse_dictionary(): supplemented function to save pulses
        * save_pulses(): currently support csv + txt

    """

    def pulse_dictionary(self) -> Dict:
        """Convert list of pulse to dictionary -> Tabulate in other formats
        Returns:
            dict_pulses: Dictionary of pulses and their characteristics
            Dictionary that has the following format
            ====== ====== ======== ========= ====== ====== ========== ====== =============
            id     mode   duration frequency x_amp  sx_amp drag_coeff sigma  pulse_pointer

            ====== ====== ======== ========= ====== ====== ========== ====== =============
        """
        dict_pulses = {'pulse id': [],
                       'mode': [],
                       'duration': [],
                       'frequency': [],
                       'x_amp': [],
                       'sx_amp': [],
                       'drag_coefficient': [],
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
            dict_pulses['drag_coefficient'].append(pulse.drag_coeff)
            dict_pulses['sigma'].append(pulse.sigma)

        return dict_pulses

    def save_pulses(self, saved_type: str, file_path: str = "pulses") -> None:
        """ Save pulses info in supported formats
        Args:
            saved_type: csv, txt - will add other types
            file_path: path of t

        Raises:
            IOError: raise if invalid path
        """
        dict_pulses = self.pulse_dictionary()
        # Check if path exist:
        if not os.path.exists(file_path):
            raise ValueError(f'Invalid filepath {file_path}')
        if saved_type == 'csv':
            # Save CSV
            save_pulses_df = pd.DataFrame(dict_pulses)
            save_pulses_df['mode'] = save_pulses_df['mode'].apply('="{}"'.format)
            save_pulses_df.to_csv(file_path, index=False, )
        elif saved_type == "json":
            # Save JSON
            json_pulse = json.dumps(dict_pulses, indent=4)
            with open(file_path, "w") as outfile:
                outfile.write(json_pulse)
        else:
            raise IOError("Unsupported type!")


class Pulse(ABC):
    """ Provides base model
    Our pulse model has 5 characteristics which are inherent to realistic pulse in a quantum computer. There are also
    T1 and T2 which are quantum decoherence times, but our model does not have at the moment.
    These 5 characteristics are: frequency, duration of the pulse, x_amp which is the max amplitude of the pulse,
    beta dephase and beta leakage

    Notes:
        * This class is abstract class; users should not initiate this class. Please check the examples in Pulse01
        and Pulse12 docstring

    Here is a list of attributes available to "Pulse" class:
        * frequency:
        * x_amp:
        * sx_amp:
        * drag_coeff:
        * duration:
        * sigma: duration / 4
        * id: unique id of pulse
        * pulse_list: Pulse_List: class attr - list of pulses that have been initiated
        * draw(): draw sine waveform of the pulse (figurative only)
    """
    pulse_list = PulseList()

    def __init__(self, frequency: float,
                 x_amp: float, sx_amp: float,
                 drag_coeff: float, duration: int) -> None:
        """ Initiates pulse and add it to the list
        Args:
            frequency: in Hz and 0 in default.
                       Users should set the frequency from TR protocol or get
                        frequency from backend instead of self initializing.
            x_amp:
            sx_amp:
            drag_coeff:
            duration: in milliseconds
        Raises:
            ValueError: raise if pulse has invalid frequency, time and x_amp
        """
        if not duration or duration <= 0:
            raise MissingDurationPulse("Time must be >= 0")
        if not x_amp:
            raise MissingAmplitudePulse("Pulse must have amplitude")
        self.frequency: float = frequency
        self.x_amp: float = x_amp
        self.sx_amp: float = sx_amp if sx_amp else self.x_amp / 2
        self.drag_coeff: float = drag_coeff
        self.duration: int = duration
        self.sigma = duration / 4 if duration else 0
        self.id = uuid.uuid4()
        Pulse.pulse_list.append(self)

    def draw(self, canvas_width: int = 600,
             canvas_height: int = 400, time_destroy: int = 5000) -> None:
        """
        Draw pulse in sine waveform. Only for visualization at the moment
        Args:
            canvas_width:
            canvas_height:
            time_destroy:
        """
        root = tk.Tk()
        root.title(f"Sine Wave of {self.__class__.__name__}, id: {self.id}")

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
    """Pulse that represents 0 -> 1 state. The pulse can stand alone or go with Pulse that represents 1 -> 2 state.
    Here is an example of initializing pulse01

        from qutritium.pulse import Pulse01

        pulse01 = Pulse01(x_amp=0.2, duration=144)
        pulse01.show()

    Here is a list of attributes available in "Pulse01" class (except from the one we have with ''Pulse'' class):
        * pulse12: related Pulse12 model
        * is_pulse12_there(): check if Pulse12 exists
        * All attributes of abstract class "Pulse"
    """

    def __init__(self, frequency: float = 0.,
                 x_amp: float = 0.2, sx_amp: float = 0.,
                 drag_coeff: float = 0., duration: int = 144,
                 pulse12: Pulse12 = None) -> None:
        """
        It depends on types of quantum computer that the frequency may vary.
        However, we typically get the frequency to be around 5.1 to 5.2 GHz
        Args:
            frequency:
            x_amp:
            sx_amp:
            drag_coeff:
            duration:
            pulse12:
        """
        if duration == 0:
            raise MissingDurationPulse('Duration should not be 0!')
        if x_amp == 0:
            raise MissingAmplitudePulse('Amplitude should not be 0!')
        super().__init__(frequency=frequency, x_amp=x_amp,
                         sx_amp=sx_amp, drag_coeff=drag_coeff,
                         duration=duration)
        self.pulse12 = pulse12

    def __str__(self) -> str:
        """
        String representation
        """
        return (
            f"Pulse01: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"drag_coeff={self.drag_coeff}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        """
        Object representation
        """
        return (
            f"Pulse01: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"drag_coeff={self.drag_coeff}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: Pulse01) -> bool:
        """
        Two pulses are equal if they share all characteristics
        Args:
            other: Pulse01

        Returns: True if all characteristics are mutual
        """
        return (
                self.frequency == other.frequency
                and self.x_amp == other.x_amp
                and self.sx_amp == other.sx_amp
                and self.drag_coeff == other.drag_coeff
                and self.duration == other.duration
        )

    def is_pulse12_there(self) -> bool:
        """
        if Pulse12 exist
        """
        return self.pulse12 is not None


class Pulse12(Pulse):
    """ Pulse that represents 1 -> 2 state. The pulse must go with its corresponding Pulse01.
    Here is an example of initializing Pulse12:

        from qutritium.pulse import Pulse01, Pulse12

        pulse01 = Pulse01(x_amp=0.2, duration=144)
        pulse12 = Pulse12(x_amp=pulse01.x_amp, duration=pulse01.duration, pulse01=pulse01)
        pulse12.show()

    Notes:
        * Pulse12 should share some pulse01's parameters such as: frequency, x_amp and duration.
        However, the package does not force that restriction

    Here is list of attributes that available in "Pulse12" class:
        * pulse01: related Pulse01
        * All attributes of abstract class "Pulse"
    """

    def __init__(self, pulse01: Pulse01,
                 frequency: float = 0., x_amp: float = 0.2,
                 sx_amp: float = 0., drag_coeff: float = 0.,
                 duration: int = 144,
                 ) -> None:
        """
        It depends on types of quantum computer that the frequency may vary.
        However, we typically get the frequency to be around 4.8 to 4.9 GHz. This is because the anharmonicity of a
        typical quantum computer is around 0.3 GHz -> f1 = f2 + anharmonicity => f2 is around 4.9GHz

        Args:
            pulse01:
            frequency:
            x_amp:
            sx_amp:
            drag_coeff:
            duration:
        """
        if duration == 0:
            raise MissingDurationPulse('Duration should not be 0!')
        if x_amp == 0:
            raise MissingAmplitudePulse('Amplitude should not be 0!')
        super().__init__(frequency=frequency, x_amp=x_amp,
                         sx_amp=sx_amp, drag_coeff=drag_coeff,
                         duration=duration)
        self.pulse01 = pulse01
        self.pulse01.pulse12 = self

    def __str__(self) -> str:
        """
        String representation
        """
        return (
            f"Pulse12: id={self.id}, frequency={self.frequency}, "
            f"x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"drag_coeff={self.drag_coeff}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __repr__(self) -> str:
        """
        Object representation
        """
        return (
            f"Pulse12: id={self.id}, frequency={self.frequency},"
            f" x_amp={self.x_amp}, sx_amp={self.sx_amp}, "
            f"drag_coeff={self.drag_coeff}, "
            f"duration={self.duration}, sigma={self.sigma}"
        )

    def __eq__(self, other: Pulse12) -> bool:
        """
        Check if two pulses are the same. They must share or have the same 'pulse01'
        Returns: True if all are mutual
        """
        return (
                self.pulse01 == other.pulse01
                and self.frequency == other.frequency
                and self.x_amp == other.x_amp
                and self.sx_amp == other.sx_amp
                and self.drag_coeff == other.drag_coeff
                and self.duration == other.duration
        )
