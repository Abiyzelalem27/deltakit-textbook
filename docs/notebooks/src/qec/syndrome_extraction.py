

from math import comb, ceil
import numpy as np
import matplotlib.pyplot as plt
import cirq, stimcirq
from myMWPM import MWPMDecoder1D
from qec.repetition_codes import create_full_repetition_code_circuit 
from qec.utils import get_binary_representation, process_batch 
from joblib import Parallel, delayed 

def receive_and_get_syndromes(bits):
    syndromes = []
    for i in range(len(bits)-1):
        syndromes.append(bits[i] ^ bits[i+1]) # XOR(bits[i], bits[i+1])
    return syndromes



def compute_decoded_syndrome_table(n_qubits, logical_state, error_gate, simulator):
    # step 1: build the repetition code circuit without errors
    # Build the circuit without random errors.
    base_circuit = create_full_repetition_code_circuit(
        n_qubits=n_qubits,
        error_probability=0.0,
        error_gate=error_gate,
        logical_state=logical_state,
    )

    # step 2: generate all possible errors
    all_possible_error_locations = []
    for i in range(2**n_qubits):
        error_pattern = get_binary_representation(i, n_qubits)
        error_locations = np.where(error_pattern)[0].tolist()
        all_possible_error_locations.append(error_locations)
    
    # step 3: create copies of the base_circuit with all possible errors inserted
    circuits = []
    if logical_state == '+':
        insert_index = (1 +                     # initial H gate
                       (n_qubits - 1) +         # CNOT gates to create logical +
                       + 1)                     # H gates to turn phase flips into bit flips
    elif logical_state == '-':
        insert_index = (1 +                     # initial H gate
                        (n_qubits - 1) +        # CNOT gates to create logical +
                        1 +                     # Z gate to turn logical + into logical -
                        1)                      # H gates to turn phase flips into bit flips
    
    data_qubits = cirq.LineQubit.range(n_qubits)
    for specific_error_locations in all_possible_error_locations:
        circuit = base_circuit.copy()
        error_moment = []
        for i in range(n_qubits):
            if i in specific_error_locations:
                error_moment.append(error_gate(data_qubits[i]))
        if error_moment:
            circuit.insert(insert_index, cirq.Moment(error_moment)) # insert a moment with all errors
        circuits.append(circuit)

    # step 4: run the circuits containing all possible errors inserted, and get the resulting syndromes
    print(f"Getting syndromes for all possible error locations in distance {n_qubits}, |{logical_state}>_L")
    results = simulator.run_batch(circuits, repetitions=1)
    print(f"Done getting syndromes for all possible error locations in distance {n_qubits}, |{logical_state}>_L")

    all_syndromes = [results[i][0].measurements['syndrome'].tolist()[0] for i in range(len(circuits))]
    decoder = MWPMDecoder1D(num_qubits=n_qubits)
    all_decoded_error_locations = [decoder.decode(syndromes) for syndromes in all_syndromes]
    
    # step 5: prepare a decoded syndrome table with a logical error diagnosis (True if a logical error has occurred, False if not)
    decoded_syndrome_table = {}
    for specific_error_locations, syndromes, decoded_error_locations in zip(all_possible_error_locations, all_syndromes, all_decoded_error_locations):
        is_logical_error = set(specific_error_locations) != set(decoded_error_locations)
        decoded_syndrome_table[tuple(specific_error_locations)] = [syndromes, decoded_error_locations, is_logical_error]
    return decoded_syndrome_table


def simulate_with_syndrome_table(decoded_syndrome_table, n_qubits, error_probability, n_shots):
        
    # Chunk size for memory efficiency
    chunk_size = 100_000_000  # 100M shots at a time
    n_chunks = (n_shots + chunk_size - 1) // chunk_size
    
    total_logical_errors = 0
    
    for chunk_index in range(n_chunks):
        # Calculate actual chunk size
        start_index = chunk_index * chunk_size
        end_index = min(start_index + chunk_size, n_shots)
        current_chunk_size = end_index - start_index
        
        # Generate error patterns for entire chunk at once
        # Shape: (current_chunk_size, n_qubits), contents uniform from 0 to 1
        random_values = np.random.random((current_chunk_size, n_qubits))
        error_patterns = random_values < error_probability
        
        # Convert binary patterns to error location tuples
        logical_errors_chunk = 0
        
        # Process in batches to avoid memory explosion in each batch
        batch_size = 10_000
        for i in range(0, current_chunk_size, batch_size):
            batch_end = min(i + batch_size, current_chunk_size)
            batch_patterns = error_patterns[i:batch_end]
            
            # Look up each pattern in syndrome table
            for pattern in batch_patterns:
                error_locations = tuple(np.where(pattern)[0])
                if decoded_syndrome_table[error_locations][2]:
                        logical_errors_chunk += 1
        
        total_logical_errors += logical_errors_chunk
    
    return total_logical_errors * 1. / n_shots

def simulate_with_syndrome_table_parallel(decoded_syndrome_table, n_qubits, error_probability, n_shots, n_workers):
    
    # Divide work among workers, make sure no work is left behind
    shots_per_worker = n_shots // n_workers
    remainder = n_shots % n_workers
    
    # Create work distribution
    work_distribution = []
    for i in range(n_workers):
        worker_shots = shots_per_worker + (1 if i < remainder else 0)
        if worker_shots > 0:
            work_distribution.append((worker_shots, i))  # (batch_size, seed)    
  
    # Parallel execution across n_workers jobs
    # see https://joblib.readthedocs.io/en/latest/parallel.html for examples of parallelization
    results = Parallel(n_jobs=n_workers)(
        # the work within one batch
        delayed(process_batch)(batch_size, seed, decoded_syndrome_table, n_qubits, error_probability)
        for batch_size, seed in work_distribution
    )
    
    # Sum up logical errors from all workers
    total_logical_errors = sum(results)
    return total_logical_errors / n_shots

def get_shor_bitflip_syndrome_measurement(block_size):
    
    circuit = cirq.Circuit()
    data_qubits = cirq.LineQubit.range(block_size * block_size)
    bitflip_syndrome_start_index = len(data_qubits)
    bitflip_syndrome_qubits = cirq.LineQubit.range(bitflip_syndrome_start_index, bitflip_syndrome_start_index + (block_size - 1)*block_size)
    
    for block_index in range(block_size):
        for within_block_index in range(block_size-1):
            data_qubit_index = block_index * block_size + within_block_index
            syndrome_index = block_index * (block_size-1) + within_block_index
            circuit.append([
                cirq.CNOT(data_qubits[data_qubit_index], bitflip_syndrome_qubits[syndrome_index]),
                cirq.CNOT(data_qubits[data_qubit_index + 1], bitflip_syndrome_qubits[syndrome_index]),
            ])
        circuit.append(cirq.Moment(cirq.I.on_each(*data_qubits)))
    circuit.append(cirq.measure(*bitflip_syndrome_qubits, key = 'bitflip-syndrome'))

    return circuit 

def get_shor_phaseflip_syndrome_measurement(block_size):
    
    circuit = cirq.Circuit()
    data_qubits = cirq.LineQubit.range(block_size * block_size)
    bitflip_syndrome_start_index = len(data_qubits)
    bitflip_syndrome_qubits = cirq.LineQubit.range(bitflip_syndrome_start_index, bitflip_syndrome_start_index + (block_size - 1) * block_size)
    phaseflip_syndrome_start_index = len(data_qubits) + len(bitflip_syndrome_qubits)
    phaseflip_syndrome_qubits = cirq.LineQubit.range(phaseflip_syndrome_start_index, phaseflip_syndrome_start_index + (block_size - 1))

    circuit.append(cirq.Moment(cirq.H.on_each(*data_qubits)))
    # block 0 to both all syndrome qubits. The data qubits of block 0 are indexed 0 to block_size
    for data_qubit_index in range(block_size):
        for phaseflip_syndrome_index in range(block_size - 1):
            circuit.append(
                cirq.CNOT(data_qubits[data_qubit_index], phaseflip_syndrome_qubits[phaseflip_syndrome_index])
            )
    circuit.append(cirq.Moment(cirq.I.on_each(*phaseflip_syndrome_qubits)))


    # all other blocks to one syndrome at a time each
    for block_index in range(1, block_size):
        data_qubit_start_index = block_size * block_index
        for data_qubit_index in range(data_qubit_start_index, data_qubit_start_index + block_size):
            circuit.append(
                cirq.CNOT(data_qubits[data_qubit_index], phaseflip_syndrome_qubits[block_index - 1])
            )
        circuit.append(cirq.Moment(cirq.I.on_each(*phaseflip_syndrome_qubits)))

    circuit.append(cirq.Moment(cirq.H.on_each(*data_qubits)))
    circuit.append(cirq.measure(*phaseflip_syndrome_qubits, key = 'phaseflip-syndrome'))
   
    return circuit 

