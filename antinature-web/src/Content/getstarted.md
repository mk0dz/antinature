# Getting Started with Antinature

This guide will help you quickly begin working with the Antinature framework for antimatter quantum chemistry simulations.

## Installation

Antinature can be installed using pip:

```bash
# Basic installation
pip install antinature

# Installation with quantum computing features
pip install antinature[qiskit]

# Full installation with all features and development tools
pip install antinature[all]
```

For development purposes, you can install directly from the source:

```bash
git clone https://github.com/mk0dz/antinature.git
cd antinature
pip install -e .[all]
```

## Basic Usage

### Creating a Simple Positronium Model

Here's a simple example to calculate the ground state of positronium:

```python
import numpy as np
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis

# Create a positronium system
positronium = MolecularData.positronium()
print(f"Created positronium system")

# Create a basis set for positronium
basis = MixedMatterBasis()
basis.create_positronium_basis()
print(f"Created and configured basis set for positronium")

# Import SCF solver and other components
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF

# Create integral engine
integral_engine = AntinatureIntegralEngine()

# Create and build Hamiltonian
hamiltonian = AntinatureHamiltonian(
    positronium,
    basis,
    integral_engine,
    include_annihilation=True
)
hamiltonian_dict = hamiltonian.build_hamiltonian()

# Run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis,
    molecular_data=positronium
)

# Solve the SCF equations
result = scf.solve_scf()
print(f"Ground state energy: {result['energy']:.6f} Hartree")
```

### Working with Mixed Matter-Antimatter Systems

For systems involving both matter and antimatter:

```python
import numpy as np
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.core.correlation import AntinatureCorrelation

# Define atoms and their positions
atoms = [('H', np.array([0.0, 0.0, 0.0]))]

# Create a system with a hydrogen atom and a positron
h_plus_positron = MolecularData(
    atoms=atoms,                                # (symbol, position) tuples
    n_electrons=1,                              # Number of electrons
    n_positrons=1,                              # Number of positrons 
    charge=0,                                   # Overall charge
    multiplicity=2,                             # Doublet state
    name="H_Positron",                          # Name for the system
    description="Hydrogen atom with a positron" # Description
)

# Create a basis set for the mixed system
basis = MixedMatterBasis()
# Setup the basis for this mixed system - pass atoms list, not the MolecularData object
basis.create_for_molecule(atoms)

# Create an integral engine
integral_engine = AntinatureIntegralEngine()

# Create a Hamiltonian and build it
hamiltonian = AntinatureHamiltonian(
    h_plus_positron, 
    basis, 
    integral_engine, 
    include_annihilation=True
)
hamiltonian_dict = hamiltonian.build_hamiltonian()

# Run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis,
    molecular_data=h_plus_positron
)
scf_result = scf.solve_scf()
print(f"SCF energy: {scf_result['energy']:.6f} Hartree")

# Run MP2 correlation calculation
correlation = AntinatureCorrelation(
    scf_result=scf_result,
    hamiltonian=hamiltonian_dict,
    basis_set=basis
)
mp2_result = correlation.calculate_correlation_energy(method='mp2', include_electron_positron=True)
print(f"MP2 correlation energy: {mp2_result['correlation_energy']:.6f} Hartree")
print(f"Total MP2 energy: {scf_result['energy'] + mp2_result['correlation_energy']:.6f} Hartree")
```

## Setting Up Calculation Parameters

### Configuring Hamiltonian Options

Customize the physics included in your calculation:

```python
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.scf import AntinatureSCF

# Create a molecule first
molecule = MolecularData.positronium()
print(f"Created positronium system")

# Create a basis set - for positronium we can use the dedicated method
basis = MixedMatterBasis()
basis.create_positronium_basis()
print(f"Created and configured basis set for positronium")

# Create an integral engine
integral_engine = AntinatureIntegralEngine()
print(f"Created integral engine")

# Create a specialized Hamiltonian
hamiltonian = AntinatureHamiltonian(
    molecule,
    basis,
    integral_engine,
    include_annihilation=True,
    include_relativistic=True  # Include relativistic corrections
)
print(f"Created Hamiltonian with specialized options")

# Build and get the Hamiltonian dictionary
hamiltonian_dict = hamiltonian.build_hamiltonian()
print(f"Built Hamiltonian matrices")

# Use in calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict, 
    basis_set=basis, 
    molecular_data=molecule
)
result = scf.solve_scf()
print(f"Energy: {result['energy']:.6f} Hartree")
```

## Visualization and Analysis

Antinature includes tools for analyzing and visualizing results:

```python
import matplotlib.pyplot as plt
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.scf import AntinatureSCF
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.specialized.visualization import AntinatureVisualizer

# Create molecule
molecule = MolecularData.positronium()
print(f"Created positronium system")

# Create a basis set - for positronium we use the dedicated method
basis = MixedMatterBasis()
basis.create_positronium_basis()
print(f"Created and configured basis set for positronium")

# Create integral engine and Hamiltonian
integral_engine = AntinatureIntegralEngine()
hamiltonian = AntinatureHamiltonian(molecule, basis, integral_engine)
print(f"Created integral engine and Hamiltonian")

# Build and get the Hamiltonian dictionary
hamiltonian_dict = hamiltonian.build_hamiltonian()
print(f"Built Hamiltonian matrices")

# Run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis, 
    molecular_data=molecule
)
result = scf.solve_scf()
print(f"Energy: {result['energy']:.6f} Hartree")

# Create visualizer with molecule and result
visualizer = AntinatureVisualizer(molecule, result)
print(f"Created visualizer")

# Plot electron density
visualizer.plot_density(particle_type='electron', plane='xy', resolution=100)
plt.savefig('electron_density.png')
print(f"Electron density plot saved to 'electron_density.png'")
```

**Note:** values maybe comeup zero because of partial setup visit and explore core examples [Examples](/docs/examples/01_anti_heh)

## Next Steps

- Check the [Examples](/docs/examples/01_anti_heh) for more complex use cases
- Review the [Tutorials](tutorials/01_intro_to_antimatter) for step-by-step guides
- Explore the [How-To Guides](howtos) for specific tasks

For more detailed information on the theory behind Antinature, see the [Theory](theory.md) section. 

