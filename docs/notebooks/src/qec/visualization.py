

import numpy as np
import matplotlib.pyplot as plt
import cirq
import stimcirq
from myMWPM import MWPMDecoder1D 
from collections import Counter 
from itertools import combinations 
from matplotlib import cm 

def plot_logical_error_probabilities(
    distances,
    physical_errors,
    all_logical_errors,
    all_analytical_errors,
    ylim=(1e-5, 1.1),
):
    physical_errors = np.asarray(physical_errors)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(distances)))

    plt.loglog(
        physical_errors,
        physical_errors,
        label="Unprotected qubit",
        linewidth=2,
        linestyle="--",
        color="gray",
    )

    for distance, logical_errors, analytical_errors, color in zip(
        distances,
        all_logical_errors,
        all_analytical_errors,
        colors,
    ):
        plt.loglog(
            physical_errors,
            logical_errors,
            label=f"d = {distance} Simulated",
            marker="o",
            linewidth=2,
            markersize=8,
            color=color,
        )

        plt.loglog(
            physical_errors,
            analytical_errors,
            label=f"d = {distance} Analytical",
            linewidth=2,
            linestyle="--",
            color=color,
        )

    plt.legend()
    plt.xlim(physical_errors.min(), physical_errors.max())
    plt.ylim(ylim)
    plt.xlabel("Physical error probability")
    plt.ylabel("Logical error probability")
    plt.tight_layout()
    plt.show()

def plot_logical_error_probabilities(distances, physical_errors, all_logical_errors, all_analytical_errors, ylim=[1e-10, 1.1]):
    
    plt.figure(figsize=(10, 8))

    num_curves = 1 if distances is None else len(distances)
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_curves))

    plt.loglog(physical_errors, physical_errors, label = 'Unprotected qubit',
                          linewidth=2, linestyle = '--', color='gray',
                          )
    
    if distances is None:
        plotter.loglog(physical_errors, all_logical_errors,
                          marker='o', linewidth=2, markersize=8,
                          color=colors[0],
                          )
    else:
        if all_analytical_errors is None:
            for distance, logical_errors, color in zip(distances, all_logical_errors, colors):
                    plotter.loglog(physical_errors, logical_errors, label = f'd = {distance}',
                                  marker='o', linewidth=2, markersize=8,
                                  color=color,
                                  )
        else:
            for distance, logical_errors, analytical_errors, color in zip(distances, all_logical_errors, all_analytical_errors, colors):
                plt.loglog(physical_errors, logical_errors, label = f'd = {distance} simulated',
                              marker='o', linewidth=2, markersize=8,
                              color=color,
                              )
                plt.loglog(physical_errors, analytical_errors, label = f'd = {distance} analytical',
                              linewidth=2, linestyle = '--', color=color,
                              )
    
    plt.legend()
    plt.xlim([physical_errors.min(), physical_errors.max()])
    plt.ylim(ylim)
    plt.grid(visible=True, which='major', axis='both')
    plt.xlabel('Physical error probability')
    plt.ylabel('Logical error probability')
    plt.tight_layout()
    plt.show() 

def show_error_pattern_distribution(n_qubits, error_probability, n_shots):
    
    # Generate error patterns
    error_patterns = np.random.random((n_shots, n_qubits)) < error_probability
    
    # Count error weights (number of errors in each pattern)
    error_weights = np.sum(error_patterns, axis=1)
    weight_counts = Counter(error_weights)

    # Convert patterns to tuples for counting
    pattern_list = []
    for pattern in error_patterns:
        error_locations = tuple(np.where(pattern)[0])
        pattern_list.append(error_locations)
    pattern_counts = Counter(pattern_list)
    
    # Analyze number of unique patterns
    unique_patterns = len(pattern_counts)
    
    # Create figure and axes
    fig, ax1 = plt.subplots(1,1,figsize=(16, 10))
    
    # Generate all possible patterns for ALL weights
    all_patterns_by_weight = {}
    for weight in range(n_qubits + 1):
        all_patterns_by_weight[weight] = list(combinations(range(n_qubits), weight))
    
    # Create color gradient for weights
    colors = cm.rainbow(np.linspace(0, 1, n_qubits + 1))
    color_map = {i: colors[i] for i in range(n_qubits + 1)}
    
    # Count frequencies for each specific pattern
    pattern_freq_detailed = []
    colors_detailed = []
    
    x_position = 0
    x_positions = []
    weight_boundaries = []
    
    for weight in sorted(all_patterns_by_weight.keys()):
        weight_boundaries.append(x_position)
        for pattern in all_patterns_by_weight[weight]:
            freq = pattern_counts.get(pattern, 0)
            pattern_freq_detailed.append(freq if freq > 0 else 0.1)  # Use 0.1 for log scale visibility
            colors_detailed.append(color_map[weight])
            x_positions.append(x_position)
            x_position += 1
    
    # Calculate bar width based on total number of patterns
    total_patterns = x_position
    bar_width = max(0.1, min(1.0, 800.0 / total_patterns))  # Adaptive width
    
    # Plot bars with thin width
    bars = ax1.bar(x_positions, pattern_freq_detailed, 
                   width=bar_width,
                   color=colors_detailed, 
                   edgecolor='black',  # Remove edges for cleaner look
                   linewidth=0.1)
    
    # Add weight group separators
    for boundary in weight_boundaries[1:]:
        ax1.axvline(x=boundary - 0.5, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

    ax1.set_xlabel('All possible error patterns (grouped by weight)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Frequency of all {2**n_qubits} possible error patterns in {n_shots} shots')
    ax1.set_yscale('log')
    ax1.set_xlim(-0.5, x_position - 0.5)
    ax1.set_xticks([])
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[i], label=f'Weight {i}') for i in range(0, n_qubits+1)]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add grid for easier reading
    ax1.grid(True, alpha=0.2, axis='y')

def plot_logical_error_probabilities(block_sizes, physical_errors, all_logical_errors, all_analytical_errors, ylim = [1e-3, 1.1]):
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(block_sizes)))
    
    plt.loglog(physical_errors, physical_errors, label = 'Unprotected qubit',
                      linewidth=2, linestyle = '--', color='gray',
                      )
    for block_size, logical_errors, analytical_errors, color in zip(block_sizes, all_logical_errors, all_analytical_errors, colors):
        plt.loglog(physical_errors, logical_errors, label = f'Shor code (block-size = {block_size}) simulated',
                      marker='o', linewidth=2, markersize=8,
                      color=color,
                      )
        plt.loglog(physical_errors, analytical_errors, label = f'Shor code(block-size = {block_size}) analytical',
                      linewidth=2, linestyle = '--', color=color,
                      )
    
    plt.legend()
    plt.xlim([physical_errors.min(), physical_errors.max()])
    plt.ylim(ylim)
    plt.xlabel('Physical error probability')
    plt.ylabel('Logical error probability')
    plt.tight_layout()
    plt.show() 


