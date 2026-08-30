

def create_noise_circuit(p, block_size, error_gate):

    def flip(p):
        return 1 if random.random() < p else 0
    
    noise_circuit = cirq.Circuit()
    data_qubits = cirq.LineQubit.range(block_size * block_size)

    has_error = [flip(p) for _ in range(len(data_qubits))]
    error_indices = [i for i, x in enumerate(has_error) if x == 1]
    errors = []
    for error_index in error_indices:
        noise_circuit.append(error_gate(data_qubits[error_index]))
        errors.append(error_gate)

    return noise_circuit, [error_indices, errors] 
