

import numpy as np
import matplotlib.pyplot as plt
import cirq
import stimcirq
from myMWPM import MWPMDecoder1D 

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
    
    plotter.figure(figsize=(10, 8))

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
