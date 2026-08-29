### Version History
### 

class MWPMDecoder1D:

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.num_parities = num_qubits - 1
    
    def decode(self, parities):
        if len(parities) != self.num_parities:
            raise ValueError(f"Expected {self.num_parities} parities, got {len(parities)}")
        
        # Count errors from the left, assuming no error on qubit 0
        errors_from_left_noerror0 = self.count_from_left(parities, start_with_error=False)
        
        # Count errors from the left, assuming no error on qubit 1
        errors_from_left_error0 = self.count_from_left(parities, start_with_error=True)
        
        # Choose the count with fewer errors
        if len(errors_from_left_noerror0) <= len(errors_from_left_error0):
            return sorted(errors_from_left_noerror0)
        else:
            return sorted(errors_from_left_error0)
    
    def count_from_left(self, parities, start_with_error):
        
        # if we know whether qubit i has an error, and we know
        # parity[i], we can determine if qubit i+1 has an error.
        #   -  -  -  -  -..  -   -..    -   -
        #  q0 q1 q2 q3 q4.. qi qi+1.. qn-1 qn
        #    p0 p1 p2 p3..    pi..      pn-1
        
        errors = []
        # Track whether current qubit has an error
        current_qubit_has_error = start_with_error
        if current_qubit_has_error:
            errors.append(0)
        
        # Propagate through the chain
        for i in range(self.num_parities):
            next_qubit_has_error = current_qubit_has_error ^ parities[i] 
            if next_qubit_has_error:
                errors.append(i + 1)
            current_qubit_has_error = next_qubit_has_error
        
        return errors
    
    def decode_all_solutions(self, parities):
        config1 = self.count_from_left(parities, start_with_error=False)
        config2 = self.count_from_left(parities, start_with_error=True)
        
        if len(config1) <= len(config2):
            return (config1, config2)
        else:
            return (config2, config1)


def test_decoder():
    print("Testing MWPM1D Decoder")
    print("-" * 40)
    
    decoder = MWPMDecoder1D(num_qubits=7)
    
    # Test 1: Single error in middle
    # Error on qubit 3 affects parities 2 and 3
    parities = [0, 0, 1, 1, 0, 0]
    errors = decoder.decode(parities)
    print(f"Test 1 - Single error in middle")
    print(f"  Parities: {parities}")
    print(f"  Expected: [3]")
    print(f"  Decoded:  {errors}")
    print()
    
    # Test 2: Error at left boundary
    parities = [1, 0, 0, 0, 0, 0]
    errors = decoder.decode(parities)
    print(f"Test 2 - Left boundary error")
    print(f"  Parities: {parities}")
    print(f"  Expected: [0]")
    print(f"  Decoded:  {errors}")
    print()
    
    # Test 3: Error at right boundary  
    parities = [0, 0, 0, 0, 0, 1]
    errors = decoder.decode(parities)
    print(f"Test 3 - Right boundary error")
    print(f"  Parities: {parities}")
    print(f"  Expected: [6]")
    print(f"  Decoded:  {errors}")
    print()
    
    # Test 4: Two errors
    # Errors on qubits 1 and 4
    parities = [1, 1, 0, 1, 1, 0]
    errors = decoder.decode(parities)
    print(f"Test 4 - Two separate errors")
    print(f"  Parities: {parities}")
    print(f"  Expected: [1, 4]")
    print(f"  Decoded:  {errors}")
    
    # Show both solutions
    config1, config2 = decoder.decode_all_solutions(parities)
    print(f"  Both solutions: {config1} or {config2}")
    print()
    
    # Test 5: Verify error propagation
    print("Test 5 - Verify propagation logic")
    test_errors = [0, 0, 1, 0, 1, 1, 0]  # Errors on qubits 2, 4, 5
    expected_parities = []
    for i in range(6):
        expected_parities.append(test_errors[i] ^ test_errors[i+1])
    print(f"  True errors: {[i for i, e in enumerate(test_errors) if e]}")
    print(f"  Expected parities: {expected_parities}")
    decoded = decoder.decode(expected_parities)
    print(f"  Decoded errors: {decoded}")
    print()
    
    # Test with distance 3, 5, 7, 9
    print("Testing scaling with distance:")
    for dist in [3, 5, 7, 9]:
        decoder = MWPMDecoder1D(num_qubits=dist)
        # Single error in middle
        parities = [0] * (dist - 1)
        mid = dist // 2
        if mid > 0:
            parities[mid-1] = 1
        if mid < dist - 1:
            parities[mid] = 1
        errors = decoder.decode(parities)
        print(f"  Distance {dist}: parities={parities}, errors={errors}, num_errors={len(errors)}")