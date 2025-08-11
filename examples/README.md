# Antinature Framework Examples

This directory contains comprehensive examples demonstrating the capabilities of the Antinature quantum chemistry framework for antimatter simulations.

## Examples Overview

### Basic Examples (01-03)
- **01_positronium_basic.py** - Introduction to positronium calculations
- **02_antihydrogen.py** - Anti-hydrogen atom simulation
- **03_positronium_molecule.py** - Positronium molecule (Ps₂) calculations

### Intermediate Examples (04-07)
- **04_positronium_hydride.py** - PsH molecule (hydrogen + positronium)
- **05_muonium.py** - Muonium and antimuonium systems
- **06_antihelium.py** - Anti-helium ion calculations
- **07_protonium.py** - Proton-antiproton bound system

### Advanced Examples (08-11)
- **08_dipositronium.py** - Multi-positron systems and stability
- **09_relativistic_effects.py** - Relativistic corrections in antimatter
- **10_annihilation_dynamics.py** - Detailed annihilation physics
- **11_antimatter_molecules.py** - Complex antimatter molecules

## Running the Examples

Each example is self-contained and can be run directly:

```bash
python examples/01_positronium_basic.py
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Antinature package installed

## Physical Systems Covered

### Pure Antimatter Systems
- Positronium (e⁺e⁻)
- Anti-hydrogen (p̄e⁺)
- Anti-helium ions
- Anti-water and other antimolecules

### Mixed Matter-Antimatter Systems
- Positronium hydride (PsH)
- Helium-positronium (HePs)
- Positronic molecules (e⁺ bound to molecules)

### Exotic Systems
- Muonium (μ⁺e⁻)
- Protonium (pp̄)
- Multi-positron systems

## Key Features Demonstrated

1. **Energy Calculations** - Ground state energies using SCF and VQE methods
2. **Basis Set Optimization** - Custom basis sets for antimatter
3. **Annihilation Physics** - Lifetime calculations and decay channels
4. **Relativistic Effects** - Fine structure and QED corrections
5. **Molecular Geometry** - Complex antimatter molecular structures
6. **CPT Symmetry** - Verification of charge-parity-time symmetry

## Expected Results

### Benchmark Energies (Hartree)
- Positronium: -0.25
- Anti-hydrogen: -0.5
- Positronium molecule (Ps₂): ~-0.516
- PsH: ~-0.789

### Typical Lifetimes
- Para-positronium: 125 ps
- Ortho-positronium: 142 ns
- Positronium in matter: 1-3 ns (pick-off)

## Scientific References

1. Positronium physics: Wheeler, J.A. (1946)
2. Antimatter spectroscopy: ALPHA Collaboration (2016)
3. Exotic atoms: Korobov, V.I. (2008)
4. QED in bound systems: Bethe & Salpeter (1951)

## Notes

- Calculations use atomic units (Hartree, Bohr) unless specified
- Some examples require Qiskit for quantum computing features
- Visualization outputs are saved as PNG files
- Runtime varies from seconds (basic) to minutes (advanced)

## Contributing

To add new examples:
1. Follow the naming convention: `XX_system_name.py`
2. Include comprehensive documentation
3. Add physical interpretation of results
4. Compare with theoretical/experimental values when available