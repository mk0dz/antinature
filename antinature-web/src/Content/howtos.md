# How-To Guides for Antinature

This section provides practical guides for accomplishing specific tasks with the Antinature framework. Each guide focuses on a particular functionality and provides code snippets and explanations.

## Basic Operations

### How to Create a Molecular System

```python
import numpy as np
from antinature.core.molecular_data import MolecularData

# Create a hydrogen molecule
h2 = MolecularData(
    atoms=[('H', np.array([0.0, 0.0, 0.0])), ('H', np.array([0.0, 0.0, 0.74]))],
    n_electrons=2,
    n_positrons=0,
    charge=0,
    multiplicity=1,
    name="H2",
    description="Hydrogen molecule"
)

# Create a positronium system using the built-in method
positronium = MolecularData.positronium()

# Create a hydrogen atom with a positron
h_positron = MolecularData(
    atoms=[('H', np.array([0.0, 0.0, 0.0]))],
    n_electrons=1,
    n_positrons=1,
    charge=0,
    multiplicity=2,
    name="H_Positron",
    description="Hydrogen atom with a positron"
)
```

**Implementation Details:**
- **Key Classes/Functions:**
  - `MolecularData`: Main class for representing molecular systems.
  - `MolecularData.positronium()`: Factory method for creating positronium.

- **Important Parameters:**
  - `atoms`: List of tuples containing (element, position) pairs.
  - `n_electrons`: Number of electrons in the system.
  - `n_positrons`: Number of positrons in the system.
  - `charge`: Total charge of the system.
  - `multiplicity`: Spin multiplicity (2S+1).

### How to Set Up Basis Sets

```python
from antinature.core.basis import MixedMatterBasis
import numpy as np

# Define atoms and their positions
atoms = [('H', np.array([0.0, 0.0, 0.0]))]

# Create a standard mixed matter basis
basis = MixedMatterBasis()

# For a normal molecule or atom
basis.create_for_molecule(atoms)

# For positronium, use the specialized method
positronium_basis = MixedMatterBasis()
positronium_basis.create_positronium_basis()
```

**Implementation Details:**
- **Key Classes/Functions:**
  - `MixedMatterBasis`: Main class for creating basis sets for mixed matter-antimatter systems.
  - `create_for_molecule()`: Creates basis functions for a molecular system.
  - `create_positronium_basis()`: Specialized method for positronium.

- **Important Parameters:**
  - For `create_for_molecule()`: Pass atoms list directly, not the MolecularData object.
  - For `create_positronium_basis()`: No parameters needed for default setup.

## Running Calculations

### How to Run an SCF Calculation

```python
import numpy as np
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF

# Create a system
atoms = [('H', np.array([0.0, 0.0, 0.0]))]
molecule = MolecularData(
    atoms=atoms,
    n_electrons=1,
    n_positrons=0,
    charge=0,
    multiplicity=2,
    name="H",
    description="Hydrogen atom"
)

# Create basis set
basis = MixedMatterBasis()
basis.create_for_molecule(atoms)

# Create integral engine and hamiltonian
integral_engine = AntinatureIntegralEngine()
hamiltonian = AntinatureHamiltonian(molecule, basis, integral_engine)
hamiltonian_dict = hamiltonian.build_hamiltonian()

# Run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis,
    molecular_data=molecule
)
result = scf.solve_scf()

# Print results
print(f"SCF Energy: {result['energy']:.6f} Hartree")
print(f"Converged: {result['converged']}")
print(f"Number of iterations: {result['iterations']}")
```

**Implementation Details:**
- **Key Steps:**
  1. Create a molecular system with `MolecularData`
  2. Create a basis set with `MixedMatterBasis`
  3. Initialize the integral engine with `AntinatureIntegralEngine()`
  4. Create a Hamiltonian and build it with `hamiltonian.build_hamiltonian()`
  5. Set up and run the SCF calculation
  6. Access results from the returned dictionary

- **Important Parameters:**
  - For `AntinatureSCF`: Pass the Hamiltonian dictionary, not the Hamiltonian object
  - The result is a dictionary - access properties with `result['energy']`, etc.

### How to Run a Correlation Calculation

```python
import numpy as np
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.core.correlation import AntinatureCorrelation

# Create a system (same as in the SCF example)
atoms = [('H', np.array([0.0, 0.0, 0.0]))]
molecule = MolecularData(
    atoms=atoms,
    n_electrons=1,
    n_positrons=0,
    charge=0,
    multiplicity=2,
    name="H",
    description="Hydrogen atom"
)

# Set up basis, integral engine and hamiltonian
basis = MixedMatterBasis()
basis.create_for_molecule(atoms)
integral_engine = AntinatureIntegralEngine()
hamiltonian = AntinatureHamiltonian(molecule, basis, integral_engine)
hamiltonian_dict = hamiltonian.build_hamiltonian()

# First run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis,
    molecular_data=molecule
)
scf_result = scf.solve_scf()

# Then run correlation calculation
correlation = AntinatureCorrelation(
    scf_result=scf_result,
    hamiltonian=hamiltonian_dict,
    basis_set=basis
)

# Calculate MP2 correlation energy
mp2_result = correlation.calculate_correlation_energy(
    method='mp2',
    include_electron_positron=True
)

# Print results
print(f"SCF Energy: {scf_result['energy']:.6f} Hartree")
print(f"MP2 correlation energy: {mp2_result['correlation_energy']:.6f} Hartree")
print(f"Total MP2 energy: {scf_result['energy'] + mp2_result['correlation_energy']:.6f} Hartree")
```

**Implementation Details:**
- **Key Steps:**
  1. Run an SCF calculation first to get the reference wavefunction
  2. Create a correlation object with the SCF result, Hamiltonian dictionary, and basis set
  3. Calculate correlation energy with your chosen method
  4. Access the correlation energy and other properties from the result dictionary

- **Important Parameters:**
  - For `AntinatureCorrelation`: Must provide `scf_result`, `hamiltonian`, and `basis_set`
  - The `mp2_result` is a dictionary containing detailed energy components
  - Access components with `mp2_result['correlation_energy']`, `mp2_result['electron_energy']`, etc.

## Specialized Features

### How to Calculate Annihilation Rates

```python
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.specialized.annihilation import AnnihilationOperator

# Create a positronium system
positronium = MolecularData.positronium()

# Set up basis and calculation
basis = MixedMatterBasis()
basis.create_positronium_basis()
integral_engine = AntinatureIntegralEngine()
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
result = scf.solve_scf()

# Calculate annihilation properties
annihilation = AnnihilationOperator(positronium, result)
rate = annihilation.calculate_rate()
lifetime = annihilation.calculate_lifetime()

print(f"Annihilation rate: {rate:.6e} s^-1")
print(f"Lifetime: {lifetime:.6e} s")
```

### How to Visualize Electron and Positron Densities

```python
import matplotlib.pyplot as plt
from antinature.core.molecular_data import MolecularData
from antinature.core.basis import MixedMatterBasis
from antinature.core.integral_engine import AntinatureIntegralEngine
from antinature.core.hamiltonian import AntinatureHamiltonian
from antinature.core.scf import AntinatureSCF
from antinature.specialized.visualization import AntinatureVisualizer

# Create a positronium system
positronium = MolecularData.positronium()

# Set up calculation
basis = MixedMatterBasis()
basis.create_positronium_basis()
integral_engine = AntinatureIntegralEngine()
hamiltonian = AntinatureHamiltonian(positronium, basis, integral_engine)
hamiltonian_dict = hamiltonian.build_hamiltonian()

# Run SCF calculation
scf = AntinatureSCF(
    hamiltonian=hamiltonian_dict,
    basis_set=basis,
    molecular_data=positronium
)
result = scf.solve_scf()

# Create visualizer and plot densities
visualizer = AntinatureVisualizer(positronium, result)

# Plot electron density
plt.figure(figsize=(10, 8))
visualizer.plot_density(particle_type='electron', plane='xy', resolution=100)
plt.title('Electron Density')
plt.savefig('electron_density.png')

# Plot positron density
plt.figure(figsize=(10, 8))
visualizer.plot_density(particle_type='positron', plane='xy', resolution=100)
plt.title('Positron Density')
plt.savefig('positron_density.png')
```

These how-to guides provide a detailed reference for accomplishing common tasks with the Antinature framework. For more complex operations or detailed explanations, please refer to the API documentation. 