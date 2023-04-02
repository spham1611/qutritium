"""List all the backend available and assign qubit value"""
from qiskit import *
from qiskit.providers.ibmq import IBMQBackend
from typing import DefaultDict, Tuple
from src.constant import QUBIT_PARA


class BackEndList(DefaultDict[str, int]):
    """Show the name of backends and map the qubit used for each backend"""

    def __init__(self) -> None:
        """

        """
        super().__init__()
        # IBM Config -> activate account in this file
        if not IBMQ.active_account():
            IBMQ.load_account()
        self.provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        self._set_up()

    def _set_up(self) -> None:
        """
        Get the name from each backend
        :return:
        """
        backends = self.provider.backends()
        for backend in backends:
            # We set it all to 0 except nairobi backend
            if "nairobi" in str(backend_name := backend.name()):
                self[backend_name] = QUBIT_PARA.NUM_QUBIT_TYPE2.value
            else:
                self[backend_name] = QUBIT_PARA.NUM_QUBIT_TYPE1.value

    def show(self) -> None:
        """
        Show the name of all available backends and their associated qubit used
        :return:
        """
        print(f"{'Backend name:':<30}{'# Qubit used:':<40}")
        for name in self:
            print(f"{name:<30}{self[name]:<40}")

    def default_backend(self, quantum_computer: str = 'ibm_nairobi') -> Tuple[IBMQBackend, int]:
        """
        Return nairobi backend as the default and its qubit value
        :return:
        """
        backend = self.provider.get_backend(quantum_computer)
        return backend, self[quantum_computer]
