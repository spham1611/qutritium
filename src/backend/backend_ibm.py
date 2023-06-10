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

"""List all the vm_backend available and assign qubit value"""
from qiskit.providers import Provider
from qiskit_ibm_provider.exceptions import IBMBackendError
from qiskit_ibm_provider import IBMProvider, IBMBackend
from qiskit_ibm_provider.api.exceptions import RequestsApiError
from src.simple_backend_log import write_log
from typing import DefaultDict, Tuple, List, Optional
from src.constant import QUBIT_PARA


class BackEnds(DefaultDict[str, Tuple]):
    """Show the name of backends and map the qubit used for package
    This class is needed due to the fact that some quantum computers work better with qubit different from 0 qubit
    You can run experiments as the example below:

        from qutritium.backend_ibm import BackEndDict

        backends = BackEndDict("some_token")
        backends.show()

    Here is list of attributes available on the ''BackEndDict'' class:
        * provider: IBMProvider
        * show(): show the available IBM quantum computers
        * default_backend(): return the backend needed based on name of the computer. It will return nairobi in default
    """

    def __init__(self, /, token: str = '', overwrite: bool = True) -> None:
        """

        Args:
            token: string representation
            overwrite: overwrite the existed IBM account in local machine
        Raises:
            EnvironmentError: raise if token is not provided and no account present
            RequestApiError: raise if invalid token
        """
        super().__init__()
        # IBM Config -> activate account in this file
        self.provider = IBMProvider()
        if not self.provider.active_account() and not token:
            raise EnvironmentError("Can't find the account saved in this session. Please activate in the script folder"
                                   "or token input")
        elif token:
            try:
                self.provider.save_account(token=token, overwrite=overwrite)
                # Create dummy var to check if we can access IBM account
                self.provider.get_backend('ibm_nairobi')
            except RequestsApiError:
                print('Invalid token')

        self._available_backends: List[IBMBackend] = []
        self._set_up()

    def _set_up(self, qc_qubit: Optional[DefaultDict[str, Tuple]] = None) -> None:
        """
        Get the name from each quantum computer and their corresponding effective qubit number.
        Note:
            * The 'effective' is solely empirical!
        """
        if qc_qubit is None:
            self._available_backends = self.provider.backends()
            for backend in self._available_backends:
                # We set it all to 0 except nairobi vm_backend
                if "nairobi" in str(backend_name := backend.name):
                    self[backend_name] = backend, QUBIT_PARA.NUM_QUBIT_TYPE2.value
                else:
                    self[backend_name] = backend, QUBIT_PARA.NUM_QUBIT_TYPE1.value
        else:
            self.update(qc_qubit)

    def show(self) -> None:
        """
        Show the name of all available backends and their associated qubit used
        """
        print(f"{'Backend name:':<30}{'# Qubit used:':<40}")
        for name in self:
            print(f"{name:<30}{self[name][1]:<40}")

    def default_backend(self, quantum_computer: str = 'ibm_nairobi') -> Tuple[IBMBackend, int]:
        """
        Return nairobi vm_backend as the default and its qubit value
        Returns:
            Tuple[IBMBackEnd, int]: The quantum computer provider and its corresponding number of qubits
        """
        backend = self[quantum_computer][0]
        if backend is None:
            raise IBMBackendError
        return backend, self[quantum_computer][1]

    def provider(self) -> IBMProvider:
        return self.provider


class BackEndChoice:
    """
    Choose the backend from list of backends and provide computer parameters
    User can set the backend as follow::
        from ...

    Here is a list of available attributes from class ''BackEndChoice'' class:
        *
    """
    def __init__(self, backend_name: str = 'ibm_nairobi') -> None:
        """

        Args:
            backend_name:

        Raises:

        """

        self._backends = BackEnds()
        self._backend, self._qubit = self._backends.default_backend(backend_name)
        self._provider = self._backend.provider
        self._anhar = self._backend.qubit_properties(self._qubit).__getattribute__('anharmonicity')
        self._default_f01 = self._backend.qubit_properties(self._qubit).frequency
        self._default_f12 = self._default_f01 + self._anhar

        write_log(self._backend)

    @property
    def default_f01(self) -> float:
        return self._default_f01

    @property
    def default_f12(self) -> float:
        return self._default_f12

    @property
    def anharmonicity(self) -> float:
        return self._anhar

    @property
    def backend(self) -> IBMBackend:
        return self._backend

    @property
    def effective_qubit(self) -> int:
        return self._qubit

    @property
    def provider(self) -> Provider:
        return self._provider

    def set_backend(self, name: str) -> None:
        """
        If user wish to change the backend
        Args:
            name:

        Returns:

        """
        self._backend, self._qubit = self._backends.default_backend(name)
