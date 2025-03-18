# Antinature Module: Use Case Examples

This directory contains a collection of example scripts demonstrating various functionalities and applications of the `antinature` module for quantum chemistry calculations with antimatter systems.

## Overview

The examples demonstrate a wide range of applications, from simple positronium calculations to complex molecular systems involving both matter and antimatter. Each example focuses on specific tasks and aspects of the module.

## Running the Examples

To run any example, make sure you have installed the `antinature` module:

```bash
pip install antinature
```

For examples that use quantum computing capabilities:

```bash
pip install antinature[qiskit]
```

Then execute an example script:

```bash
python use_cases/01_basic_positronium.py
```

## List of Examples

### Basic Systems

1. **[Basic Positronium](01_basic_positronium.py)**
   - Create a positronium system (electron-positron bound state)
   - Calculate its ground state energy
   - Calculate annihilation rate and lifetime

2. **[Anti-Hydrogen](02_anti_hydrogen.py)**
   - Create both hydrogen and anti-hydrogen systems
   - Calculate and compare their ground state energies
   - Study the differences in electron vs. positron density distribution

3. **[Positronium-Molecule Interaction](03_positronium_molecule_interaction.py)**
   - Create a combined system of positronium and H₂
   - Calculate the interaction energy
   - Study the effect of varying distance on the interaction

### Advanced Systems

4. **[Positronium Excited States](04_positronium_excited_states.py)**
   - Create a positronium system with an extended basis
   - Calculate multiple excited states
   - Analyze and visualize the state energies

5. **[Anti-Helium Ion](05_anti_helium_ion.py)**
   - Create an anti-helium ion system
   - Calculate the ground state energy
   - Calculate the positron density distribution

6. **[Relativistic Positronium](06_relativistic_positronium.py)**
   - Calculate positronium properties without relativistic effects
   - Include relativistic corrections and recalculate
   - Compare the results and visualize the differences

### Scattering and Interactions

7. **[Positron-Hydrogen Scattering](07_positron_hydrogen_scattering.py)**
   - Set up a positron-hydrogen scattering system
   - Calculate scattering phase shifts at different energies
   - Compute and visualize the scattering cross-section

8. **[Quantum Positronium](08_quantum_positronium.py)**
   - Set up a positronium system for quantum simulation
   - Map the problem to a quantum circuit
   - Run the simulation on a quantum simulator

### Complex Systems

9. **[Multi-Positron Complex](09_multi_positron_complex.py)**
   - Create a water molecule with multiple positrons
   - Calculate the energy and positron distributions
   - Visualize the electron and positron density distributions

10. **[Anti-Helium Atom](10_anti_helium_atom.py)**
    - Create an anti-helium atom system
    - Calculate its ground state energy and compare with regular helium
    - Analyze the positron spatial distribution

### Advanced Phenomena

11. **[Electron-Positron Annihilation](11_electron_positron_annihilation.py)**
    - Calculate annihilation rates for different systems
    - Study positron penetration into electron clouds
    - Compare annihilation rates across different molecules

12. **[Anti-Molecules](12_anti_molecules.py)**
    - Create and analyze an anti-hydrogen molecule (anti-H₂)
    - Study a simple matter-antimatter molecular complex
    - Visualize the electron and positron distributions

## Results

The examples save their results (plots, data, etc.) in the `use_cases/results/` directory, which is automatically created when running the scripts.

## References

For more information about the scientific background and theoretical methods used in these examples, please refer to:

1. Armour, E. A. G., & Humberston, J. W. (2010). *Positron Physics*. Cambridge University Press.
2. Charlton, M., & Humberston, J. W. (2000). *Positron Physics*. Cambridge University Press.
3. Surko, C. M., Gribakin, G. F., & Buckman, S. J. (2005). "Low-energy positron interactions with atoms and molecules." *Journal of Physics B: Atomic, Molecular and Optical Physics*, 38(6), R57. 