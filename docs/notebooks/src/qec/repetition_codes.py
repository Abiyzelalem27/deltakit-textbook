

import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import math
from math import comb, ceil
import cirq, stimcirq
from myMWPM import MWPMDecoder1D
from qec.utils import get_syndrome_measurement 
from tqdm import tqdm



def create_full_repetition_code_circuit(n_qubits, error_probability, error_gate, logical_state):

    # Create qubits: data qubits for encoding, syndrome qubits for syndrome measurement
    data_qubits = cirq.LineQubit.range(n_qubits)
    syndrome_qubits = cirq.LineQubit.range(n_qubits, 2*n_qubits - 1)
    
    circuit = cirq.Circuit()

    # Step 0: Decide what quantum state we are protecting. It's either 0 or 1. Then encode it
    encoding_circuit = create_repetition_code_encoder(n_qubits)

    # logical state |0>_L = |0000...>
    # do nothing, since all data qubits start reset at |0>.
    if logical_state == '0':
        pass
        
    # logical state |1>_L = |1111...>
    # apply X gate on all data qubits since they all start reset at |0>
    if logical_state == '1':
        circuit.append(
            cirq.Moment(cirq.X(data_qubits[0]))
                       )
    
    circuit += encoding_circuit    
    
    # Step 1: Simulate noise with a Pauli error error_type occurring with probability error_probability
    circuit.append(
        cirq.Moment([
        error_gate(qubit).with_probability(error_probability) for qubit in data_qubits
                    ])
                   )
            
    # Step 2: Measure error syndrome
    circuit += get_syndrome_measurement(data_qubits, syndrome_qubits)

    # Step 3: Measure data qubits
    # we will use it to predict the initial state by correcting the final state using the syndrome data
    # When we can't predict successfully, that's a logical error
    circuit.append(cirq.measure(*data_qubits, key='data_qubits'))
    
    return circuit

def create_repetition_code_encoder(n_qubits):

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # The first qubit holds the quantum state
    for i in range(1, n_qubits):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    return circuit

def send_with_random_iid_bitflip(bits, p_error = 0.1):
    return [1^bit if random.random() <= p_error else bit for bit in bits]

def flip_all_bits(bits):
    return [1^bit for bit in bits]

def send_with_correlated_bitflip(bits, p_error = 0.1, p_correlated = 1e-4):
    # with probability p_correlated, flip all bits
    #otherwise, flip each bit independently with probability p_error
    if random.random() <= p_correlated: # flip every bit
        return flip_all_bits(bits)
    else:
        return send_with_random_iid_bitflip(bits, p_error)
        

def receive_and_interpret(bits, majority_vote = False):
    """
    Count received binary values or decode them using majority voting.

    In a repetition code, one information bit is encoded as several
    identical physical bits. After transmission through a noisy channel,
    some of those bits may have flipped. Majority-vote decoding estimates
    the original information bit by selecting the value that occurs most
    frequently in the received sequence.

    Let the received sequence be

        y = (y_1, y_2, ..., y_n),

    where every ``y_i`` is either 0 or 1. Define the number of zeros and
    ones as

        N_0 = sum(1 - y_i),
        N_1 = sum(y_i).

    The majority-vote decoder returns

        0, if N_0 > N_1,
        1, if N_1 > N_0.

    If ``N_0 = N_1``, there is no unique majority. This can happen when
    the sequence contains an even number of bits. The function raises an
    error in that case rather than selecting a result arbitrarily.


    Notes
    -----
    For a repetition code of odd length ``n = 2t + 1``, majority voting
    can recover the original bit when no more than ``t`` bits have
    flipped:

        t = floor((n - 1) / 2).

    For example, a length-five repetition code can correct up to two
    bit-flip errors.
    """
    
    received_bits_counted = Counter(bits).most_common()
    if majority_vote:
        return received_bits_counted[0][0]
    else:
        return received_bits_counted

def encode(bit, num_copies):
    """ Encode one classical bit using a repetition code."""
    
    return [bit] * num_copies


def send(bits, p_error):
    """
    Transmit a binary sequence through a binary symmetric error channel.

    Each input bit is transmitted independently. With probability
    ``p_error``, the bit is flipped:

        0 -> 1
        1 -> 0

    With probability ``1 - p_error``, the bit remains unchanged.

    Mathematically, if ``x_i`` is the transmitted bit, the received bit is

        y_i = x_i XOR e_i,

    where the error variable ``e_i`` follows a Bernoulli distribution:

        P(e_i = 1) = p_error,
        P(e_i = 0) = 1 - p_error.

    Therefore, the channel transition probabilities are

        P(y_i != x_i) = p_error,
        P(y_i == x_i) = 1 - p_error.

    All bit-flip events are assumed to be statistically independent.


    Notes
    -----
    The operation ``bit ^ 1`` is the bitwise exclusive-OR operation.
    For binary values, it is equivalent to flipping the bit:

        0 XOR 1 = 1,
        1 XOR 1 = 0.

    This function models only independent classical bit-flip errors.
    It does not model correlated errors, erasures, measurement errors,
    phase errors, or general quantum noise channels.
    """

    return [1^bit if random.random() <= p_error else bit for bit in bits]


def get_code_error_probability(code_distance, p_error, n_shots):
    """
    Estimate the logical error probability of a repetition code.

    A random information bit is encoded, transmitted through a binary
    symmetric channel and decoded using majority voting. The experiment
    is repeated ``n_shots`` times.

    The estimated logical error probability is

        p_L = N_fail / n_shots,

    where ``N_fail`` is the number of trials for which the decoded bit
    differs from the original information bit.

    """
    code_errors = 0

    for _ in range(n_shots):
        desired_message = random.choice([0, 1])
        encoded_message = encode(desired_message, num_copies=code_distance)

        sent_message = send(encoded_message, p_error=p_error)
        received_message = receive_and_interpret(sent_message, majority_vote=True)

        if received_message != desired_message:
            code_errors += 1

    return code_errors / n_shots

def get_code_error_probability_random_iid(code_distance, p_error, n_shots):
    """
    Estimate the logical error probability under independent bit flips.

    The Monte Carlo estimator is

        p_L = N_fail / n_shots,

    where ``N_fail`` is the number of trials in which majority-vote
    decoding fails to recover the original information bit.
    """

    code_errors = 0.0
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

    code_errors = 0.0
    for _ in range(n_shots):
        desired_message = random.choice([0,1])

        encoded_message = encode(desired_message, num_copies = code_distance)
        sent_message = send_with_correlated_bitflip(encoded_message, p_error = p_error, p_correlated = p_correlated)
        received_message = receive_and_interpret(sent_message, majority_vote = True)

        if received_message != desired_message:
            code_errors += 1

    return code_errors / n_shots


def create_repetition_code_encoder(n_qubits):

    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    
    # The first qubit holds the quantum state
    for i in range(1, n_qubits):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    return circuit



def create_full_repetition_code_circuit(n_qubits, error_probability, error_gate, logical_state):

    # Create qubits: data qubits for encoding, syndrome qubits for syndrome measurement
    
    data_qubits = cirq.LineQubit.range(n_qubits)
    syndrome_qubits = cirq.LineQubit.range(n_qubits, 2*n_qubits - 1)
    
    circuit = cirq.Circuit()

    # Step 0: Decide what quantum state we are protecting. It's either 0 or 1. Then encode it
    encoding_circuit = create_repetition_code_encoder(n_qubits)
    
    # logical state |0>_L = |0000...>
    # do nothing, since all data qubits start reset at |0>.
    if logical_state == '0':
        pass
        
    # logical state |1>_L = |1111...>
    # apply X gate on all data qubits since they all start reset at |0>
    if logical_state == '1':
        circuit.append(
            cirq.Moment(cirq.X(data_qubits[0]))
                       )
        circuit += encoding_circuit
        
    # logical state |+>_L = 1/sqrt(2) * (|0>_L + |1>_L) = 1/sqrt(2) * (|0000...> + |1111...>)
    if logical_state == '+':
        circuit.append(
            cirq.Moment(cirq.H(data_qubits[0]))
                       )
        circuit += encoding_circuit

    # logical state |->_L = 1/sqrt(2) * (|0>_L - |1>_L) = 1/sqrt(2) * (|0000...> - |1111...>)
    if logical_state == '-':
        circuit.append(
            cirq.Moment(cirq.H(data_qubits[0]))
                       )
        circuit += encoding_circuit
        circuit.append(
            cirq.Moment(cirq.Z(data_qubits[0]))
                       )
    
    # Step 1: Hadamard sandwich where phase flips will be inserted
    
    circuit.append(
        cirq.Moment(cirq.H.on_each(*data_qubits))
    )
    
    ## errors go here
    
    circuit.append(
        cirq.Moment(cirq.H.on_each(*data_qubits))
    )
            
    # Step 2: Measure error syndrome
    circuit += get_syndrome_measurement(data_qubits, syndrome_qubits)

    # Step 3: Measure data qubits
    circuit.append(cirq.measure(*data_qubits, key='data_qubits'))
            
    return circuit 