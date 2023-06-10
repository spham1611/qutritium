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

""" Write simple IBM backend info log """

import logging
import datetime
import os

folder_log = "log_files"
folder_path = os.path.abspath(folder_log)
if not os.path.exists(folder_path):
    raise NotADirectoryError(folder_path)


def write_log(backend) -> None:
    """
    Write an info log that contains some information about the accessed quantum computer
    Args:
        backend: IBMBackend
    """
    config = backend.configuration()
    properties = backend.properties()
    defaults = backend.defaults()
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    us = 1e6
    ns = 1e9
    GHz = 1e-9

    # Configure logging with the timestamped log file name
    log_filename = os.path.join(folder_path, f"logfile_{timestamp}.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'Connecting to IBM quantum computer: {config.backend_name} ...')

    message = f"Basic Info of the backend \n" \
              f"Version: {config.backend_version},\n" \
              f"Number of qubits: {config.n_qubits},\n" \
              f"Support pulse: {config.open_pulse},\n" \
              f"Basis Gates: {config.basis_gates},\n" \
              f"dt: {config.dt},\n" \
              f"Meas_levels: {config.meas_levels}.\n" \
              f"Basic properties of qubit 0\n" \
              f"T1 time of {properties.t1(0) * us},\n" \
              f"T2 time of {properties.t2(0) * us},\n" \
              f"resonant frequency of {properties.frequency(0) * GHz}.\n" \
              f"DriveChannel(0) defaults to a modulation frequency of {defaults.qubit_freq_est[0] * GHz} GHz.\n" \
              f"MeasureChannel(0) defaults to a modulation frequency of {defaults.meas_freq_est[0] * GHz} GHz."

    logging.info(message)
    logging.info('Done!')
