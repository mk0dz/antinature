# Antimatter Quantum Chemistry (antinature)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance quantum chemistry framework designed specifically for simulating antimatter systems, including positronium, anti-hydrogen, and other exotic matter-antinature configurations.

## Features

- **Specialized antinature Physics**: Dedicated algorithms for positrons and positron-electron interactions
- **Relativistic Corrections**: Implementation of relativistic effects critical for accurate antinature modeling
- **Annihilation Processes**: Modeling of electron-positron annihilation dynamics
- **Quantum Computing Integration**: Built-in Qiskit integration for quantum simulations of antinature systems
- **Validation Tools**: Framework for verifying results against known theoretical benchmarks

## Installation

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

# Install in development mode with all dependencies
pip install -e .[all]

# Run tests
pytest
```

### Dependencies

The package has the following optional dependency groups:

- `qiskit`: Required for quantum computing features (Qiskit, Qiskit-Nature, Qiskit-Aer)
- `dev`: Development tools (pytest, black, isort)
- `all`: Installs all optional dependencies

If you encounter any test failures related to missing dependencies, please ensure you've installed the appropriate dependency group:

```bash
# For quantum computing features
pip install -e .[qiskit]

# For development tools
pip install -e .[dev]

# For all dependencies
pip install -e .[all]
```

## Quick Start

```python
import numpy as np
from antinature.core.molecular_data import MolecularData
from antinature.utils import create_antinature_calculation

anti_heh_data = MolecularData(
        atoms=[
            ('He', np.array([0.0, 0.0, 0.0])),
            ('H', np.array([0.0, 0.0, 1.46]))  # ~1.46 Bohr ≈ 0.77 Å bond distance
        ],
        n_electrons=0,       # No electrons in antimatter system
        n_positrons=2,       # 2 positrons (equivalent to 2 electrons in normal HeH+)
        charge=0,            # Overall neutral (2 positrons balance -2 from anti-He, anti-H)
        name="Anti-HeH+",
        description="Anti-helium hydride ion (anti-HeH+) with exotic antimatter composition"
    )

print(f"Molecule: {anti_heh_data.name}")
print(f"Description: {anti_heh_data.description}")
print(f"Formula: {anti_heh_data.get_formula()}")
print(f"Number of positrons: {anti_heh_data.n_positrons}")
print(f"Nuclear repulsion energy: {anti_heh_data.get_nuclear_repulsion_energy():.8f} Hartree")
```

## Citing This Work

If you use this package in your research, please cite:

```
@software{antinature,
  author = {Mukul},
  title = {Antimatter Quantum Chemistry},
  url = {https://github.com/mk0dz/antinature},
  version = {0.1.0},
  year = {2025},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on how to set up a development environment and contribute to this project.