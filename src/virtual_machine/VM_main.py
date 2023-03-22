import VM
'''
Simple example to test VM 
'''
qc = VM.Virtual_Machine(5, 0, None)
qc.add_gate('WH', 0)
qc.add_gate('CNOT', 1, 0)
qc.add_gate('measure', [])
qc.draw()
qc.run()
print(qc.get_counts())
qc.plot("histogram")