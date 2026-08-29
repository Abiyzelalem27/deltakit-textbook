


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
    full_circuit = create_full_repetition_code_circuit(n_qubits, error_probability, logical_state, error_gate)
    
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
    base_circuit = create_full_repetition_code_circuit(n_qubits, error_probability, logical_state, error_gate)

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

