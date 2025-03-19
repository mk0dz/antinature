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

# Create a positronium system (electron-positron bound state)
positronium = MolecularData.positronium()

# Configure and run the calculation
result = create_antinature_calculation(
    positronium,
    basis_options={'quality': 'positronium'},
    calculation_options={
        'include_annihilation': True,
        'include_relativistic': True
    }
)

# Print key results
print(f"Ground state energy: {result['energy']:.6f} Hartree")
print(f"Annihilation rate: {result['annihilation_rate']:.6e} s^-1")
print(f"Lifetime: {result['lifetime_ns']:.4f} ns")
```

## Examples

The package includes several example scripts for common antinature research scenarios:

- `examples/positronium_example.py`: Basic positronium energy calculation
- `examples/complex_molecule.py`: Multi-particle antimatter system simulations
- `examples/anti_heh.py`: Anti-hydrogen-helium molecule calculations
- `examples/lih_ion.py`: Lithium hydride ion with positron calculations

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