"""
Minimal 2-qutrit example: H3 + CSUM → qutrit Bell state.
"""
from qutritium import StatevectorSimulator, QutritCircuit
from qutritium.gates import CSUM, H3


def main() -> None:
    qc = QutritCircuit(2, None)
    qc.append(H3(), first_qutrit=0)
    qc.append(CSUM(), first_qutrit=0, second_qutrit=1)
    qc.measure_all()

    sim = StatevectorSimulator(qc)
    sim.run(num_shots=10_000)
    print(sim.get_counts())


if __name__ == "__main__":
    main()
