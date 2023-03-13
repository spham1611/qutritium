import VM
'''
Simple example to test VM 
'''
qc = VM.Virtual_Machine(3, 0, None)

qc.add_gate('WH', 0)
qc.add_gate('measure', [])
qc.draw()
qc.run()
print(qc.get_counts())
qc.plot("histogram")
# n_qutrit = 2
# state = np.array([[0], [0], [1/np.sqrt(2)], [0], [0], [0], [0], [1/np.sqrt(2)], [0]])
# print(state.shape)
# VM_utility.print_statevector(state, n_qutrit)