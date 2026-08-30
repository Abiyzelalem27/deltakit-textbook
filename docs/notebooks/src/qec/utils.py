

import random
from collections import Counter
import numpy as np 
import cirq
import stimcirq
from myMWPM import MWPMDecoder1D
import mlx.core as mx 

def get_syndrome_measurement(qubits, syndrome_qubits):

    # For each pair of adjacent qubits, measure the ZiZi+1 stabilizer
    syndrome_measurement = []
        
    for i in range(len(qubits) - 1):
        # Extract the parity of qubits i and i+1 onto syndrome qubit i
        syndrome_measurement.append(cirq.CNOT(qubits[i], syndrome_qubits[i]))
        syndrome_measurement.append(cirq.CNOT(qubits[i+1], syndrome_qubits[i]))
    
    # Measure the syndrome qubits to extract the syndrome
    syndrome_measurement.append(cirq.measure(*syndrome_qubits, key='syndrome'))
    
    return syndrome_measurement  

def process_batch(batch_size, seed, syndrome_table, n_qubits, error_probability):    
    # Ensure different randomness per worker
    np.random.seed(seed)  
    # Generate error patterns for this batch
    error_patterns = np.random.random((batch_size, n_qubits)) < error_probability
    
    logical_errors = 0
    # Process each pattern
    for pattern in error_patterns:
        error_locations = tuple(np.where(pattern)[0])
        
        # Look up in syndrome table
        if syndrome_table[error_locations][2]:  # Logical error has occurred
            logical_errors += 1
    
    return logical_errors 


def convert_syndrome_table_to_GPU(syndrome_table, n_qubits):
    # Create a lookup table as a dense array for GPU -- a list containing [error_pattern: is_logical_error] pairs
    syndrome_table_GPU = mx.zeros(2**n_qubits, dtype=mx.bool_)

    for specific_error_locations, entry in syndrome_table.items():
        is_logical_error = entry[2]
      # Convert error tuple to binary index
        error_pattern_index = 0
        for qubit_index in specific_error_locations:
            error_pattern_index |= 1 << int(qubit_index)

        syndrome_table_GPU[error_pattern_index] = is_logical_error

    return syndrome_table_GPU

def simulate_with_syndrome_table_parallel_GPU_mlx(decoded_syndrome_table, n_qubits, error_probability, n_shots):
    
    # Convert syndrome table to GPU-friendly format
    syndrome_table_GPU = convert_syndrome_table_to_GPU(decoded_syndrome_table, n_qubits)

    total_logical_errors = 0
    batch_size = min(100_000_000, n_shots)  # Process 100M at a time on GPU
    
    shots_processed = 0
    while shots_processed < n_shots:
        current_batch = min(batch_size, n_shots - shots_processed)
        
        # Generate random numbers uniformly from 0 to 1 in an array of shape (current_batch, n_qubits) on GPU
        # Then convert that to an error pattern on the qubits
        error_patterns = mx.random.uniform(shape=(current_batch, n_qubits)) < error_probability
        
        # Convert patterns to indices for lookup
        # The error patterns contain 1 where there is an error, and 0 where there isn't
        # They can be indexed from 0 to 2^n-1 by converting the error pattern tuple into a decimal integer
        # By doing the dot product (2^n-1 2^n-2 2^n-3.... 2^1 2^0) . (error_pattern)
        powers = mx.array([2**i for i in range(n_qubits)])
        error_patterns_indices = mx.sum(error_patterns.astype(mx.int32) * powers, axis=1)
        
        # Lookup whether each index leads to a logical error, and add them all up
        logical_errors_batch = mx.sum(syndrome_table_GPU[error_patterns_indices])
        
        # Transfer result back to CPU
        total_logical_errors += int(logical_errors_batch)
        
        shots_processed += current_batch
            
    return total_logical_errors / n_shots 

def get_binary_representation(index, n_qubits):
    # Binary representation of index in n_qubits bits, LSB first
    # This trick avoids having to do string manipulations with Python's generic bin() function
    # TLDR: >> is a right-shift, & 1 picks out the LSBs
    return (index >> np.arange(n_qubits)) & 1 

def create_initial_state(starting_state = '+'):
    
    starting_circuit = cirq.Circuit()
    starting_qubit = cirq.LineQubit(0)

    if starting_state == '0':
        starting_circuit.append(cirq.Moment(cirq.I(starting_qubit)))
    elif starting_state == '1':
        starting_circuit.append(cirq.Moment(cirq.X(starting_qubit)))
    elif starting_state == '+':
        starting_circuit.append(cirq.Moment(cirq.H(starting_qubit)))
    elif starting_state == '-':
        starting_circuit.append(cirq.Moment(cirq.H(starting_qubit)))
        starting_circuit.append(cirq.Moment(cirq.Z(starting_qubit)))

    return starting_circuit
    

def create_shor_encoder(block_size):

    qubits = cirq.LineQubit.range(block_size * block_size)
    circuit = cirq.Circuit()
    # outer phase encoding
    for i in range(1, block_size):
        circuit.append([
            cirq.CNOT(qubits[0], qubits[block_size * i]),
    ])

    circuit.append(cirq.Moment(
                cirq.H.on_each(*[qubits[i*block_size] for i in range(block_size)])
        )
    )
    
    # inner bit-flip encoding
    for i in range(block_size):
        block_start = i * block_size
        for j in range(1, block_size):
            circuit.append([
                cirq.CNOT(qubits[block_start], qubits[block_start + j]),
            ])
    return circuit  

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

def convert_syndrome_table_to_GPU(syndrome_table, n_qubits):
     # Create a lookup table as a dense array for GPU -- a list containing [error_pattern: is_logical_error] pairs
    syndrome_table_GPU = mx.zeros(2**n_qubits, dtype=mx.bool_)

    for error_locations, entry in syndrome_table.items():
        is_logical_error = entry[2]  # Extract True or False
       # Convert error tuple to binary index
        index = 0
        for qubit_index in error_locations:
            index |= 1 << int(qubit_index)

        syndrome_table_GPU[index] = is_logical_error

    return syndrome_table_GPU

    

def simulate_with_syndrome_table_parallel_GPU_mlx(syndrome_table, n_qubits, error_probability, n_shots):
    
    # Convert syndrome table to GPU-friendly format
    syndrome_table_GPU = convert_syndrome_table_to_GPU(syndrome_table, n_qubits)

    total_logical_errors = 0
    batch_size = min(100_000_000, n_shots)  # Process 100M at a time on GPU
    
    shots_processed = 0
    while shots_processed < n_shots:
        current_batch = min(batch_size, n_shots - shots_processed)
        
        # Generate random numbers uniformly from 0 to 1 in an array of shape (current_batch, n_qubits) on GPU
        # Then convert that to an error pattern on the qubits
        error_patterns = mx.random.uniform(shape=(current_batch, n_qubits)) < error_probability
        
        # Convert patterns to indices for lookup
        # The error patterns contain 1 where there is an error, and 0 where there isn't
        # They can be indexed from 0 to 2^n-1 by converting the error pattern tuple into a decimal integer
        # By doing the dot product (2^n-1 2^n-2 2^n-3.... 2^1 2^0) . (error_pattern)
        powers = mx.array([2**i for i in range(n_qubits)])
        error_patterns_indices = mx.sum(error_patterns.astype(mx.int32) * powers, axis=1)
        
        # Lookup whether each index leads to a logical error, and add them all up
        logical_errors_batch = mx.sum(syndrome_table_GPU[error_patterns_indices])
        
        # Transfer result back to CPU
        total_logical_errors += int(logical_errors_batch)
        
        shots_processed += current_batch
            
    return total_logical_errors / n_shots