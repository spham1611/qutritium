import VM

'''
Simple example to test VM 
'''

qc = VM.Virtual_Machine(2, 0, None)
instruction_1 = VM.instruction('x01', 0)
instruction_2 = VM.instruction('x01', 1)

qc.add_gate(instruction_1)
qc.add_gate(instruction_2)
qc.draw()