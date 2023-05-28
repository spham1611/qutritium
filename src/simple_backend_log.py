"""
Write simple IBM backend info log
"""
import logging
import datetime


def write_log(backend):
    """
    Write an info log that contains some information about the accessed quantum computer
    :param backend:
    :return:
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
    log_filename = f"logfile_{timestamp}.log"
    logging.basicConfig(filename='', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
              f"U2 gate error of {properties.gate_error('u2', 0)},\n" \
              f"U2 gate duration of {properties.gate_length('u2', 0) * ns},\n" \
              f"resonant frequency of {properties.frequency(0) * GHz}.\n" \
              f"DriveChannel(0) defaults to a modulation frequency of {defaults.qubit_freq_est[0] * GHz} GHz.\n" \
              f"MeasureChannel(0) defaults to a modulation frequency of {defaults.meeas_freq_est[0] * GHz} GHz."

    logging.info(message)
    logging.info('Done')
