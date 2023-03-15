import VM
'''
Simple example to test VM 
'''
qc = VM.Virtual_Machine(3, 0, None)
qc.add_gate('WH', 2)
qc.add_gate('CNOT', 0, 2)
qc.add_gate('measure', [])
qc.draw()
qc.run()
print(qc.get_counts())
qc.plot("histogram")
