

import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plotter
plotter.rcParams['font.family'] = 'Monospace'
import math


def encode(bit, num_copies):
     """Encode one classical bit by repetition."""
    return [bit] * num_copies

def send(bits, p_error = 0.1):
    """Transmit bits through a binary symmetric error channel."""
    return [1^bit if random.random() <= p_error else bit for bit in bits]

def receive_and_interpret(bits, majority_vote = False):
    """Count received bits or decode using majority voting."""
    received_bits_counted = Counter(bits).most_common()
    if majority_vote:
        return received_bits_counted[0][0]
    else:
        return received_bits_counted 