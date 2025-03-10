"""
Example: HeH+ with Positron
===========================

This example demonstrates a basic calculation on the HeH+ system with a positron.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Try different import approaches to make it more robust
try:
    # First try direct import (assuming package is installed)
    from antimatter_qc.antimatter_core import AntimatterQuantumChemistry, Molecule
    print("Using installed package import")
except ImportError:
    # If that fails, add parent directory to path and try direct import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from antimatter_qc.antimatter_core import AntimatterQuantumChemistry, Molecule
        print("Using parent directory import")
    except ImportError:
        # If that still fails, try importing directly from the module files
        try:
            from antimatter_core import AntimatterQuantumChemistry, Molecule
            print("Using direct module import")
        except ImportError:
            print("ERROR: Could not import AntimatterQuantumChemistry. Make sure the modules are properly installed.")
            sys.exit(1)

# Create a molecule (HeH+)
print("Creating HeH+ molecule...")
molecule = Molecule()
molecule.add_atom("He", [0.0, 0.0, 0.0])
molecule.add_atom("H", [0.0, 0.0, 1.5])

# Create the quantum chemistry calculator
print("Setting up quantum chemistry calculator...")
qc = AntimatterQuantumChemistry(
    molecule=molecule,
    basis_type='positron-minimal',  # Use minimal basis for speed
    include_relativistic=False,
    include_annihilation=True,
    scf_max_iterations=20,
    scf_convergence=1e-5
)

# Initialize components - this will use mock objects if actual modules aren't available
print("Initializing components...")
qc.initialize_components()

# Calculate integrals
print("Calculating integrals...")
qc.calculate_integrals()

# Run SCF calculation
print("Running SCF calculation...")
results = qc.run_scf()

# Print results
print("\nSCF Results:")
print(f"Energy: {results['energy']:.10f} Hartree")
print(f"Converged: {results['converged']}")
print(f"Iterations: {results['iterations']}")

# Analyze the results
print("\nAnalyzing results...")
analysis = qc.analyze_results()

# Create visualization (save to file rather than display)
print("Creating visualizations...")
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)
qc.visualize_results(plot_type='density', save_dir=save_dir, show=False)

# Print full summary
print("\nFull calculation summary:")
print(qc.get_summary())

print("\nExample completed successfully!")