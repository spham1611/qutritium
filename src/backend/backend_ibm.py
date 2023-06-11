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

"""List all the available backends and assign their empirically effective qubit"""
from qiskit_ibm_provider import IBMProvider, IBMBackend
from typing import DefaultDict, Tuple, Optional
from collections import defaultdict
from src.simple_backend_log import write_log
from src.constant import QUBIT_PARA


def initiate_eff_dict() -> DefaultDict:
    """ Creates a default dictionary in which ibm_nairobi effective qubit: 6
    Returns:
        Dictionary that has the following format
        ===============  ===============
        IBMBackend Name  Effective Qubit
        'ibm_nairobi'    6
        ===============  ===============
    """
    return defaultdict(lambda: 0, {'ibm_nairobi': QUBIT_PARA.QUBIT_CHANGE_TYPE1.value})


class EffProvider(IBMProvider):
    """ Provides default attributes and functions of IBMProvider + effective qubit + other parameters for qutrit process
    An example of this flow::

        from qutritium.backend.backend_ibm import EffProvider

        eff_provider = EffProvider()
        backend, _ = eff_provider.backend('ibmq_lima')

        eff_provider.show()

    Notes:
        * This class inherits IBMProvider so the docs is analogous to IBMProvider.

    Here is a list of available attributes in class "EffProvider" class, besides IBMProvider attrs:
        * eff_dict: a dictionary that contains effective qubit number of respective quantum computer
        * backend(): return the wanted backend and its parameters
        * show(): print all available backends
    """

    def __init__(
            self,
            token: Optional[str] = None,
            url: Optional[str] = None,
            name: Optional[str] = None,
            instance: Optional[str] = None,
            proxies: Optional[dict] = None,
            verify: Optional[bool] = None,
            eff_dict: Optional[DefaultDict] = None,
    ) -> None:
        """ Refer to the IBMProvider doc
        Args:
            token:
            url:
            name:
            instance:
            proxies:
            verify:
            eff_dict: Dictionary contains name of the backend and their effective qubits. For example:
                    dict_eff = {'ibmq_lima': 2,
                                'ibm_nairobi': 6,}

        Returns:
            An instance of EffBackends
        """
        super().__init__(token, url, name, instance, proxies, verify)
        if not self.active_account():
            raise ValueError('Can not find account saved on disk. Please provide token via constructor'
                             ', or save_account() function')
        self.eff_dict: DefaultDict[str, int] = eff_dict if eff_dict else initiate_eff_dict()

    def retrieve_backend_info(self, name: str) -> Tuple[IBMBackend, DefaultDict]:
        """

        Args:
            name: name of the quantum computer which appears on API

        Returns:
            Tuple: IBMBackend and its parameters

        """
        backend = self.backends(name=name)[0]
        default_freq = backend.defaults().qubit_freq_est[self.eff_dict[name]]
        anharmonicity = backend.properties().qubits[self.eff_dict[name]][3].value * QUBIT_PARA.GHZ.value
        backend_params = defaultdict(lambda: 0,
                                     {'effective_qubit': self.eff_dict[name],
                                      'drive_frequency': default_freq,
                                      'anharmonicity': anharmonicity})
        write_log(backend)
        return backend, backend_params

    def show(self) -> None:
        """
        Show all available backends based on given provider
        """
        print(f"{'Backend name:':<30}{'# Qubit used:':<40}")
        for available_backend in self.backends():
            name = available_backend.name
            print(f"{name:<30}{self.eff_dict[name]:<40}")
