

import random
from collections import Counter
import numpy as np 
import cirq
import stimcirq
from myMWPM import MWPMDecoder1D
    

def count_from_left(syndromes, start_with_error):
    
    # if we know whether bit i has an error, and we know
    # parity[i], we can determine if bit i+1 has an error.
    #   -  -  -  -  -..  -   -..    -   -
    #  b0 b1 b2 b3 b4.. bi bi+1.. bn-1 bn
    #    p0 p1 p2 p3..    pi..      pn-1
    
    errors = []
    # Track whether current bit has an error
    current_bit_has_error = start_with_error
    if current_bit_has_error:
        errors.append(0)
    
    # Propagate through the chain
    for i in range(len(syndromes)):
        next_bit_has_error = current_bit_has_error ^ syndromes[i] 
        if next_bit_has_error:
            errors.append(i + 1)
        current_bit_has_error = next_bit_has_error
        
    return errors

def decode(syndromes):
    # If there are no errors, we can immediately return an empty list
    if not any(syndromes):
        return []
    # Count errors from the left, assuming no error on bit 1
    errors_from_left_noerror0 = count_from_left(syndromes, start_with_error=False)
    # Count errors from the left, assuming yes error on bit 1
    errors_from_left_error0 = count_from_left(syndromes, start_with_error=True)

    # Choose the count with fewer errors
    if len(errors_from_left_noerror0) <= len(errors_from_left_error0):
        return sorted(errors_from_left_noerror0)
    else:
        return sorted(errors_from_left_error0)

def apply_corrections(received_bits, error_locations):
    
    corrected_bits = received_bits.copy()
    for error_location in error_locations:
        corrected_bits[error_location] = 1 - corrected_bits[error_location]
    return corrected_bits 

def unencode(corrected_bits):

    unencoded_bit = None
    for i in range(len(corrected_bits)-1):
        if i == 0:
            unencoded_bit = corrected_bits[0] & corrected_bits[1]
        else:
            unencoded_bit &= corrected_bits[i+1]            
    return unencoded_bit 