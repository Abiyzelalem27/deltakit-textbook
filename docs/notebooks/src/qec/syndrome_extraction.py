

from math import comb, ceil
import numpy as np
import matplotlib.pyplot as plt
import cirq, stimcirq
from myMWPM import MWPMDecoder1D

def receive_and_get_syndromes(bits):
    syndromes = []
    for i in range(len(bits)-1):
        syndromes.append(bits[i] ^ bits[i+1]) # XOR(bits[i], bits[i+1])
    return syndromes

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