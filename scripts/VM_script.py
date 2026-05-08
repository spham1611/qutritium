"""
Minimal example: build a 2-qutrit Hadamard + CNOT circuit, simulate it,
and plot the measurement counts.

Updated for qutritium v1.0.0: imports now go through the top-level
``qutritium`` package rather than ``src.X``.
"""
# MODIFIED (v1.0.0): import paths updated from ``src.quantumcircuit.QC`` /
# ``src.vm_backend.QASM_backend`` to the new ``qutritium.*`` package layout.
import numpy as np

from qutritium import QASM_Simulator, Qutrit_circuit


def main() -> None:
    qc = Qutrit_circuit(n_qutrit=2, initial_state=None)
    qc.add_gate("hdm", first_qutrit_set=0)
    qc.add_gate("CNOT", first_qutrit_set=1, second_qutrit_set=0)
    qc.measure_all()

    sim = QASM_Simulator(qc)
    sim.run(num_shots=10_000)
    print("Final state:")
    print(sim.return_final_state())
    print("-" * 40)
    print("Counts:", sim.get_counts())

    # Optional: requires the [plot] extra (matplotlib).
    try:
        fig = sim.plot(plot_type="histogram")
        fig.savefig("vm_script_histogram.png", bbox_inches="tight")
        print("Histogram saved to vm_script_histogram.png")
    except ImportError:
        print("matplotlib not installed; skipping plot. "
              "Install with `pip install qutritium[plot]`.")


if __name__ == "__main__":
    main()
