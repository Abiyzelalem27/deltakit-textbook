


import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math 
import cirq
import stimcirq 
from myMWPM import MWPMDecoder1D 
from tqdm import tqdm 
from qec.repetition_codes import encode, receive_and_interpret, send_with_random_iid_bitflip, send_with_correlated_bitflip, send, create_full_repetition_code_circuit 
from qec. utils import create_initial_state, create_shor_encoder, create_noise_circuit, get_binary_representation, simulate_with_syndrome_table_parallel_GPU_mlx, get_syndrome_measurement, convert_syndrome_table_to_GPU
from qec.syndrome_extraction import get_shor_bitflip_syndrome_measurement, get_shor_phaseflip_syndrome_measurement


def get_code_error_probability_random_iid(code_distance, p_error, n_shots):
    """
    Estimate the logical error probability under independent bit flips.

    The Monte Carlo estimator is

        p_L = N_fail / n_shots,

    where ``N_fail`` is the number of trials in which majority-vote
    decoding fails to recover the original information bit.
    """
    code_errors = 0
    for _ in range(n_shots):
        desired_message = random.choice([0,1])

        encoded_message = encode(desired_message, num_copies = code_distance)
        sent_message = send_with_random_iid_bitflip(encoded_message, p_error = p_error)
        received_message = receive_and_interpret(sent_message, majority_vote = True)

        if received_message != desired_message:
            code_errors += 1

    return code_errors / n_shots

def get_code_error_probability_correlated(code_distance, p_error, p_correlated, n_shots):
    """
    Estimate the logical error probability under correlated bit flips.

    With probability ``p_correlated``, all encoded bits flip together.
    Otherwise, every bit flips independently with probability
    ``p_error``.
    """

    code_errors = 0
    for _ in range(n_shots):
        desired_message = random.choice([0,1])

        encoded_message = encode(desired_message, num_copies = code_distance)
        sent_message = send_with_correlated_bitflip(encoded_message, p_error,  p_correlated)
        received_message = receive_and_interpret(sent_message, majority_vote = True)

        if received_message != desired_message:
            code_errors += 1

    return code_errors / n_shots 


def majority_decoder_error_probability(code_distance, p_error):
    """
    Exact logical error probability when ordinary majority voting is used.
    """
    minimum_errors_for_failure = (code_distance + 1) // 2

    return sum(
        math.comb(code_distance, number_of_errors)
        * p_error**number_of_errors
        * (1.0 - p_error)**(code_distance - number_of_errors)
        for number_of_errors in range(
            minimum_errors_for_failure,
            code_distance + 1,
        )
    )


def adaptive_decoder_error_probability(code_distance, p_error):
    """
    Exact logical error probability of the optimal decoder.

    Majority voting is used for p < 0.5.
    Minority voting is used for p > 0.5.
    """
    effective_error_probability = min(p_error, 1.0 - p_error)
    return majority_decoder_error_probability(code_distance, effective_error_probability)

def get_code_error_probability_mv(code_distance, p_error, n_shots):

    code_errors = 0
    for _ in range(n_shots):
        desired_message = random.choice([0,1])

        encoded_message = encode(desired_message, num_copies = code_distance)
        sent_message = send(encoded_message, p_error)
        received_message = receive_and_interpret(sent_message, majority_vote = True)

        if received_message != desired_message:
            code_errors += 1

    return code_errors / n_shots 

def get_logical_error_probability_for_rep_code(n_qubits, error_probability, 
                                               logical_state, error_gate, 
                                               n_shots, simulator):

    # step 1: build the repetition code circuit
    full_circuit = create_full_repetition_code_circuit(n_qubits, error_probability, error_gate, logical_state)
    
    # step 2: simulate physical errors during computation
    # print(f"Running distance {n_qubits}, bit-flip rep code, |{logical_state}>_L, error probability {error_probability}")
    # t = time.time()
    result = simulator.run(full_circuit, repetitions = n_shots)
    # elapsed = time.time() - t
    # print(f"Took {elapsed:.0f} seconds")
    
    # step 3: decode the syndrome information
    syndromes = result.measurements['syndrome']
    decoder = MWPMDecoder1D(num_qubits)
    decoded_syndromes = [decoder.decode(syndrome) for syndrome in syndromes]

    # step 4: count logical errors
    datas = result.measurements['data_qubits']
    logical_errors = 0

    initial_state = [int(logical_state)]*n_qubits
    for data, error_locations in zip(datas, decoded_syndromes):
        final_state = data.copy()
        for error_location in error_locations:
            final_state[error_location] = 1-final_state[error_location] # flip the bit at error_location
        if not np.array_equal(initial_state, final_state):
            logical_errors += 1

    return logical_errors * 1. / n_shots

def get_logical_error_probability_simulated(distances, physical_errors, n_shots, error_gate, logical_state, simulator):

    all_logical_errors = []
    for distance in distances:
        print(f"Simulating distance-{distance} repetition code circuits")
        thisdistance_logicalerrors = []
        for physical_error in physical_errors:
            logical_error = get_logical_error_probability_for_rep_code(distance, physical_error, logical_state, error_gate,  n_shots, simulator)
            thisdistance_logicalerrors.append(logical_error)
        all_logical_errors.append(thisdistance_logicalerrors)

    return all_logical_errors

def get_logical_error_probability_analytical(distances, physical_errors):

    # # method 1: small p approximation
    # all_analytical_errors = []
    # for distance in distances:
    #     t = ceil(distance / 2)
    #     analytical_errors = comb(distance, t) * physical_errors**t
    #     all_analytical_errors.append(analytical_errors)

    # method 2: full expression
    all_analytical_errors = []
    for distance in distances:
        analytical_error = 0
        for i in range(ceil(distance/2.), distance + 1):
            analytical_error += comb(distance, i) * physical_errors**i * (1-physical_errors)**(distance-i)
        all_analytical_errors.append(analytical_error)

    return all_analytical_errors


def get_logical_error_probability_for_rep_code(n_qubits, error_probability, logical_state, error_gate, n_shots, simulator):

    # step 1: build the repetition code circuit without errors
    base_circuit = create_full_repetition_code_circuit(n_qubits, error_probability, error_gate, logical_state)

    # step 2: generate all errors
    # first, create independent errors in a n_shots x n_qubits matrix
    # then, for each shot, the errors can be taken sliced out of this matrix and applied to the data qubits
    actual_errors_all_shots = []
    error_mask = np.random.random((n_shots, n_qubits)) < error_probability
    for shot in range(n_shots):
        actual_errors_all_shots.append(np.where(error_mask[shot])[0].tolist())

    # step 3: insert all errors into copies of the base_circuit
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
    for shot_errors in actual_errors_all_shots:
        circuit = base_circuit.copy()
        error_moment = []
        for i in range(n_qubits):
            if i in shot_errors:
                error_moment.append(error_gate(data_qubits[i]))
        if error_moment:
            circuit.insert(insert_index, cirq.Moment(error_moment)) # insert a moment with all errors
        circuits.append(circuit)

    # step 4: run all noise instances (circuits) in one batch
    results = simulator.run_batch(circuits, repetitions=1)

    # step 5: decode the syndrome information
    syndromes = [results[i][0].measurements['syndrome'].tolist()[0] for i in range(n_shots)]
    decoder = MWPMDecoder1D(num_qubits=n_qubits)
    decoded_syndromes = [decoder.decode(syndrome) for syndrome in syndromes]

    # step 6: count logical errors
    logical_errors = 0
     # phase-flip rep code, detecting phase flips by turning them into bit flips
    for actual_error_locations, decoded_error_locations, in zip(actual_errors_all_shots, decoded_syndromes):
        # compare decoder with knowledge of actual error locations
        if not np.array_equal(actual_error_locations, decoded_error_locations):
            logical_errors += 1
            
    return logical_errors * 1. / n_shots

def get_logical_error_probability_analytical(distances, physical_errors):

    # # method 1: small p approximation
    # all_analytical_errors = []
    # for distance in distances:
    #     t = ceil(distance / 2)
    #     analytical_errors = comb(distance, t) * physical_errors**t
    #     all_analytical_errors.append(analytical_errors)

    # method 2: full expression
    all_analytical_errors = []
    for distance in distances:
        analytical_error = 0
        for i in range(ceil(distance/2.), distance + 1):
            analytical_error += comb(distance, i) * physical_errors**i * (1-physical_errors)**(distance-i)
        all_analytical_errors.append(analytical_error)

    return all_analytical_errors 

def get_shor_code_logical_error_probability(physical_error, block_size, n_shots, simulator, starting_state, error_gate):
    starting_circuit = create_initial_state(starting_state)
    encoder_circuit = create_shor_encoder(block_size)
    bitflip_syndrome_circuit = get_shor_bitflip_syndrome_measurement(block_size)
    phaseflip_syndrome_circuit = get_shor_phaseflip_syndrome_measurement(block_size)
    decoder = MWPMDecoder1D(num_qubits=block_size)

    num_logical_errors = 0
    for i in range(n_shots):
        full_shor_circuit = cirq.Circuit()
        noise_circuit, noise_applied = create_noise_circuit(physical_error, block_size, error_gate)
        full_shor_circuit = starting_circuit + encoder_circuit + noise_circuit + bitflip_syndrome_circuit + phaseflip_syndrome_circuit
        result = simulator.run(full_shor_circuit, repetitions = 1)
        bitflip_syndromes = result.measurements['bitflip-syndrome'][0]
        phaseflip_syndromes = result.measurements['phaseflip-syndrome'][0]
    
        decoded_bitflip_errors = []
        # decoding bit flips
        # we can tell where bitflip errors happened within each block precisely
        for i in range(0, len(bitflip_syndromes), block_size-1):  # Slice list in steps of block_size
            these_syndromes = bitflip_syndromes[i:i+block_size-1]
            withinblock_error_indices = decoder.decode(these_syndromes)
            error_locations = [int(i // (block_size-1))*block_size + x for x in withinblock_error_indices]
            for error_loc in error_locations:
                decoded_bitflip_errors.append([error_loc, cirq.X])
    
        # decoding phase flips
        # we can only tell that phaseflip errors happened within a block, not precisely where
        # but that's OK because a single Z gate will undo the phase error in a block
        block_errors_0 = [0] # assume block 0 has no phase flip error
        block_errors_1 = [1] # assume block 0 has phase flip error
        for j in range(len(phaseflip_syndromes)):
            # assume block 0 has no phase flip error
            block_errors_0.append(0 ^ phaseflip_syndromes[j])
            # assume block 0 has phase flip error
            block_errors_1.append(1 ^ phaseflip_syndromes[j])
        # pick the option with minimum weight
        if sum(block_errors_0) <= sum(block_errors_1):
            block_errors_final = block_errors_0
        else:
            block_errors_final = block_errors_1
        decoded_phaseflip_error_blocks = [i for i, thisblock_error in enumerate(block_errors_final) if thisblock_error == 1]
    
        # decide whether or not a logical error occurred    
        is_logical_error = False
        qubits_with_errors, qubit_errors = noise_applied

        # bitflip errors are checked within each block. if we are unable to detect a bitflip error within a block
        # then a logical error occurs
        for qubit_with_error, qubit_error in zip(qubits_with_errors, qubit_errors):
            if qubit_error == cirq.X:
                if [qubit_with_error, cirq.X] not in decoded_bitflip_errors:
                    is_logical_error = True
                    
        # if we were unable to detect a block where a Z error happened, then a logical error occurs
        for qubit_with_error, qubit_error in zip(qubits_with_errors, qubit_errors):
            if qubit_error == cirq.Z:
                thisqubit_block = qubit_with_error // block_size
                if thisqubit_block not in decoded_phaseflip_error_blocks:
                    is_logical_error = True
                    
        if is_logical_error:
            num_logical_errors += 1
            
    return num_logical_errors * 1./ n_shots

def get_logical_error_probability_simulated(
    block_sizes, physical_errors, n_shots,
    simulator, starting_state, error_gate
):
    all_logical_errors = []

    for block_size in block_sizes:
        print(f"Simulating block-size-{block_size} Shor code circuits")
        thisblock_size_logicalerrors = []

        for physical_error in tqdm(physical_errors):
            logical_error = get_shor_code_logical_error_probability(
                physical_error=physical_error,
                block_size=block_size,
                n_shots=n_shots,
                simulator=simulator,
                starting_state=starting_state,
                error_gate=error_gate,
            )
            thisblock_size_logicalerrors.append(logical_error)

        all_logical_errors.append(thisblock_size_logicalerrors)

    return all_logical_errors 

def get_logical_error_probability_analytical(block_sizes, physical_errors):
    
    all_analytical_errors = []
    for block_size in block_sizes:
        # analytical_success = 0
        # for i in range(floor(block_size/2.)+1):
        #     analytical_success += comb(block_size**2, i) * physical_errors**i * (1-physical_errors)**(block_size**2-i)
        # analytical_errors = 1-analytical_success
        analytical_errors = 1 - (1-physical_errors)**9 - 9*physical_errors*(1-physical_errors)**8
        all_analytical_errors.append(analytical_errors)

    return all_analytical_errors 

def get_shor_code_logical_error_probability_syndrome_table(block_size, simulator, starting_state, error_gate):
    starting_circuit = create_initial_state(starting_state)
    encoder_circuit = create_shor_encoder(block_size)
    bitflip_syndrome_circuit = get_shor_bitflip_syndrome_measurement(block_size)
    phaseflip_syndrome_circuit = get_shor_phaseflip_syndrome_measurement(block_size)
    decoder = MWPMDecoder1D(num_qubits=block_size)

    print(f"Getting syndromes for all possible error locations in block-size {block_size}, |{starting_state}>_L, error gate {error_gate}")
    # first, generate the list of all possible 2^n error locations, n = # data qubits
    data_qubits = cirq.LineQubit.range(block_size**2)
    num_data_qubits = len(data_qubits)
    all_possible_error_locations = []
    for i in tqdm(range(2**num_data_qubits)):
        error_pattern = get_binary_representation(i, num_data_qubits)
        error_locations = np.where(error_pattern)[0].tolist()
        all_possible_error_locations.append(error_locations)

    # then, using these error locations, construct all possible circuits with errors
    all_bitflip_syndromes = []
    all_phaseflip_syndromes = []
    all_circuits = []
    for specific_error_locations in tqdm(all_possible_error_locations):
        full_shor_circuit = starting_circuit + encoder_circuit
        # insert errors
        error_moment = []
        for i in range(num_data_qubits):
            if i in specific_error_locations:
                error_moment.append(error_gate(data_qubits[i]))
        if error_moment:
            full_shor_circuit += cirq.Moment(error_moment) # insert a moment with all errors
        full_shor_circuit += bitflip_syndrome_circuit + phaseflip_syndrome_circuit
        all_circuits.append(full_shor_circuit)

    results = simulator.run_batch(all_circuits, repetitions = 1)
    all_bitflip_syndromes = [results[i][0].measurements['bitflip-syndrome'].tolist()[0] for i in range(len(all_circuits))]
    all_phaseflip_syndromes = [results[i][0].measurements['phaseflip-syndrome'].tolist()[0] for i in range(len(all_circuits))]

    logical_decisions = []
    for bitflip_syndromes, phaseflip_syndromes, applied_error_locations in tqdm(zip(all_bitflip_syndromes, all_phaseflip_syndromes, all_possible_error_locations)):
        decoded_bitflip_errors = []
        # decoding bit flips
        # we can tell where bitflip errors happened within each block precisely
        for i in range(0, len(bitflip_syndromes), block_size-1):  # Slice list in steps of block_size
            these_syndromes = bitflip_syndromes[i:i+block_size-1]
            withinblock_error_indices = decoder.decode(these_syndromes)
            error_locations = [int(i // (block_size-1))*block_size + x for x in withinblock_error_indices]
            for error_loc in error_locations:
                decoded_bitflip_errors.append([error_loc, cirq.X])

        # decoding phase flips
        # we can only tell that phaseflip errors happened within a block, not precisely where
        # but that's OK because a single Z gate will undo the phase error in a block
        block_errors_0 = [0] # assume block 0 has no phase flip error
        block_errors_1 = [1] # assume block 0 has phase flip error
        for j in range(len(phaseflip_syndromes)):
            # assume block 0 has no phase flip error
            block_errors_0.append(0 ^ phaseflip_syndromes[j])
            # assume block 0 has phase flip error
            block_errors_1.append(1 ^ phaseflip_syndromes[j])
        # pick the option with minimum weight
        if sum(block_errors_0) <= sum(block_errors_1):
            block_errors_final = block_errors_0
        else:
            block_errors_final = block_errors_1
        decoded_phaseflip_error_blocks = [i for i, thisblock_error in enumerate(block_errors_final) if thisblock_error == 1]
    
        # decide whether or not a logical error occurred    
        is_logical_error = False

        qubits_with_errors = applied_error_locations
        qubit_errors = [error_gate]*len(qubits_with_errors)
        # bitflip errors are checked within each block. if we are unable to detect a bitflip error within a block
        # then a logical error occurs
        for qubit_with_error, qubit_error in zip(qubits_with_errors, qubit_errors):
            if qubit_error == cirq.X:
                if [qubit_with_error, cirq.X] not in decoded_bitflip_errors:
                    is_logical_error = True
                    
        # if we were unable to detect a block where a Z error happened, then a logical error occurs
        for qubit_with_error, qubit_error in zip(qubits_with_errors, qubit_errors):
            if qubit_error == cirq.Z:
                thisqubit_block = qubit_with_error // block_size
                if thisqubit_block not in decoded_phaseflip_error_blocks:
                    is_logical_error = True
                    
        logical_decisions.append(is_logical_error)

    syndrome_table = {}
    for applied_error_locations, logical_decision in zip(all_possible_error_locations,logical_decisions):
        syndrome_table[tuple(applied_error_locations)] = logical_decision
        
    return syndrome_table

def get_logical_error_probability_simulated_GPU(block_sizes, physical_errors, n_shots, simulator, starting_state, error_gate):
    
    all_logical_errors = []
    for block_size in block_sizes:
        print(f"Simulating block-size-{block_size} Shor code circuits")
        syndrome_table = get_shor_code_logical_error_probability_syndrome_table(block_size=block_size, simulator=simulator, starting_state = starting_state, error_gate = error_gate)
        thisblock_size_logicalerrors = []
        for physical_error in tqdm(physical_errors):
            logical_error = simulate_with_syndrome_table_parallel_GPU_mlx(syndrome_table, n_qubits = block_size**2, error_probability = physical_error, n_shots = n_shots)
            thisblock_size_logicalerrors.append(logical_error)
        all_logical_errors.append(thisblock_size_logicalerrors)

    return all_logical_errors 

