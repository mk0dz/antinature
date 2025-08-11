# Antimatter Quantum Chemistry (antinature)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15079747.svg)](https://doi.org/10.5281/zenodo.15079747)
[![Tests](https://img.shields.io/badge/tests-96%25%20passing-brightgreen)](https://github.com/mk0dz/antinature)

A high-performance quantum chemistry framework designed specifically for simulating antimatter systems, including positronium, anti-hydrogen, and other exotic matter-antimatter configurations.

## 🌟 Key Features

### Core Capabilities
- **Specialized Antimatter Physics**: Dedicated algorithms for positrons and positron-electron interactions
- **Mixed Matter-Antimatter Systems**: Support for complex systems containing both matter and antimatter
- **Relativistic Corrections**: Implementation of relativistic effects critical for accurate antimatter modeling
- **Annihilation Processes**: Comprehensive modeling of electron-positron annihilation dynamics
- **Quantum Computing Integration**: Built-in Qiskit integration for quantum simulations of antimatter systems

### Supported Systems
- **Positronium (e⁺e⁻)**: Complete support with optimized basis sets
- **Anti-hydrogen (p̄e⁺)**: Full CPT-symmetric calculations
- **Positronium Molecule (Ps₂)**: Multi-positron bound states
- **Positronium Hydride (PsH)**: Mixed matter-antimatter molecules
- **Muonium (μ⁺e⁻)** and **Antimuonium (μ⁻e⁺)**: Exotic leptonic atoms
- **Custom Systems**: Build any antimatter configuration

## 📦 Installation

### Basic Installation

```bash
pip install antinature
```

### Installation with Quantum Computing Support

```bash
pip install antinature[qiskit]
```

### Development Installation

For development purposes with testing tools:

```bash
# Clone the repository
git clone https://github.com/mk0dz/antinature.git
cd antinature

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e .[all]

# Run tests
pytest tests/
```

### Dependencies

The package has the following optional dependency groups:

- `qiskit`: Quantum computing features (Qiskit ≥1.0.0, Qiskit-Algorithms)
- `dev`: Development tools (pytest, black, isort)
- `all`: Installs all optional dependencies

## 🚀 Quick Start

### Basic Positronium Calculation

```python
from antinature.utils import calculate_positronium

# Calculate positronium ground state
result = calculate_positronium()
print(f"Positronium energy: {result['energy']:.6f} Hartree")
print(f"Theoretical: -0.250000 Hartree")
print(f"Error: {result['error']:.2e} Hartree")
```

### Custom Antimatter System

```python
import numpy as np
from antinature import MolecularData, AntinatureCalculator

# Create anti-hydrogen atom
anti_h = MolecularData(
    atoms=[('H', np.array([0.0, 0.0, 0.0]))],
    n_electrons=0,    # No electrons
    n_positrons=1,    # One positron
    charge=-1,        # Anti-proton has -1 charge
    name="Anti-hydrogen"
)

# Calculate energy
calc = AntinatureCalculator()
result = calc.calculate_custom_system(anti_h, accuracy='medium')
print(f"Anti-hydrogen energy: {result['energy']:.6f} Hartree")
```

### Advanced Example: Positronium Hydride (PsH)

```python
from antinature.specialized.systems import AntinatureSystems

# Get predefined PsH system
psh = AntinatureSystems.get_system('PsH')

# Calculate with high accuracy
calc = AntinatureCalculator(print_level=1)
result = calc.calculate_custom_system(psh, accuracy='high')

# Analyze results
print(f"PsH total energy: {result['energy']:.6f} Hartree")
print(f"Binding energy: {result.get('binding_energy', 0):.6f} Hartree")
print(f"Annihilation lifetime: {result.get('lifetime', 0):.2e} seconds")
```

## 📊 Examples

The `examples/` directory contains comprehensive demonstrations:

1. **01_positronium_basic.py** - Basic positronium calculations
2. **02_antihydrogen.py** - Anti-hydrogen atom simulations
3. **03_positronium_molecule.py** - Ps₂ molecule calculations
4. **04_positronium_hydride.py** - PsH system analysis
5. **05_muonium.py** - Muonium and antimuonium
6. **06_antihelium.py** - Anti-helium ion calculations
7. **07_protonium.py** - Proton-antiproton bound states
8. **08_dipositronium.py** - Multi-positron systems
9. **09_relativistic_effects.py** - Relativistic corrections
10. **10_annihilation_dynamics.py** - Annihilation physics
11. **11_antimatter_molecules.py** - Complex antimatter molecules

Run all examples:
```bash
cd examples
for script in *.py; do python "$script"; done
```

## 🔬 Technical Details

### Architecture

The framework consists of several key modules:

- **Core Components**:
  - `MolecularData`: Molecular structure representation
  - `BasisSet`: Gaussian basis functions with positron support
  - `AntinatureHamiltonian`: Hamiltonian construction
  - `AntinatureSCF`: Self-consistent field solver
  - `AntinatureIntegralEngine`: Integral computations

- **Specialized Modules**:
  - `PositroniumSCF`: Optimized positronium solver
  - `AnnihilationOperator`: Annihilation dynamics
  - `RelativisticCorrection`: Relativistic effects
  - `AntinatureVisualizer`: Visualization tools

- **Quantum Computing**:
  - `AntinatureAnsatz`: Quantum circuit templates
  - `PositroniumCircuit`: Specialized circuits
  - `VQESolver`: Variational quantum eigensolver

### Theoretical Background

The framework implements:

1. **Hartree-Fock for Mixed Systems**: Extended HF formalism for electron-positron systems
2. **Specialized Basis Sets**: Optimized Gaussian basis for positrons with diffuse functions
3. **Annihilation Terms**: Contact density calculations for annihilation rates
4. **Relativistic Corrections**: Mass-velocity, Darwin, and spin-orbit terms
5. **CPT Symmetry**: Verification of charge-parity-time symmetry

## 📈 Performance

The framework has been optimized for:
- **Vectorized Operations**: NumPy-based vectorization for integral calculations
- **Caching**: Smart caching of expensive integrals
- **Convergence Acceleration**: DIIS and level-shifting for SCF
- **Basis Optimization**: Automatic removal of linear dependencies

Typical calculation times:
- Positronium: < 0.1 seconds
- Anti-hydrogen: < 0.5 seconds  
- Complex molecules: 1-10 seconds

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=antinature

# Run specific test category
pytest tests/test_positronium.py -v
```

Current test coverage: **96%** (28/29 tests passing)

## 📚 Documentation

Detailed documentation for each module:

- [Core Modules](docs/core.md) - Basis sets, Hamiltonians, SCF
- [Specialized Systems](docs/specialized.md) - Positronium, annihilation
- [Quantum Computing](docs/quantum.md) - Qiskit integration
- [Examples](docs/examples.md) - Detailed example walkthroughs

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:
- Setting up a development environment
- Code style and standards
- Testing requirements
- Documentation guidelines

## 📖 Citing This Work

If you use this package in your research, please cite:

```bibtex
@software{antinature2025,
  author = {Mukul},
  title = {Antimatter Quantum Chemistry Framework},
  url = {https://github.com/mk0dz/antinature},
  version = {0.1.2},
  year = {2025},
  doi = {10.5281/zenodo.15079747}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Qiskit team for quantum computing infrastructure
- NumPy and SciPy communities for numerical libraries
- Theoretical antimatter physics community for validation data

## 📮 Contact

For questions, bug reports, or collaboration:
- GitHub Issues: [github.com/mk0dz/antinature/issues](https://github.com/mk0dz/antinature/issues)
- Email: [Contact via GitHub profile](https://github.com/mk0dz)

---

**Note**: This framework is under active development. While core functionality is stable, some advanced features may still be experimental. Please report any issues you encounter!