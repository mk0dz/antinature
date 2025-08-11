#!/usr/bin/env python
"""
Example 1: Basic Positronium Calculation
=========================================
This example demonstrates the simplest way to calculate the ground state
energy of positronium using the Antinature framework.

Positronium is an exotic atom consisting of an electron and a positron
bound together. It's the simplest pure antimatter system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from antinature import quick_test, calculate_positronium

def main():
    print("=" * 60)
    print("EXAMPLE 1: BASIC POSITRONIUM CALCULATION")
    print("=" * 60)
    
    # Method 1: Quick test
    print("\n1. Quick test of positronium:")
    result = quick_test()
    if result:
        print("✓ Quick test passed!")
    
    # Method 2: Simple positronium calculation
    print("\n2. Simple positronium calculation:")
    ps_result = calculate_positronium(accuracy='medium')
    
    print(f"\nResults:")
    print(f"  Energy: {ps_result['energy']:.6f} Hartree")
    print(f"  Converged: {ps_result['converged']}")
    print(f"  Iterations: {ps_result.get('iterations', 'N/A')}")
    
    # Theoretical comparison
    theoretical_energy = -0.25  # Hartree
    error = abs(ps_result['energy'] - theoretical_energy)
    print(f"\nAccuracy Analysis:")
    print(f"  Theoretical: {theoretical_energy:.6f} Hartree")
    print(f"  Error: {error:.6e} Hartree")
    print(f"  Relative Error: {100 * error / abs(theoretical_energy):.2f}%")

if __name__ == "__main__":
    main()