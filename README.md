# quantimatter Quantum Chemistry (quantimatter)

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A high-performance quantum chemistry framework designed specifically for simulating antimatter systems, including positronium, anti-hydrogen, and other exotic matter-quantimatter configurations.

## Features

- **Specialized quantimatter Physics**: Dedicated algorithms for positrons and positron-electron interactions
- **Relativistic Corrections**: Implementation of relativistic effects critical for accurate quantimatter modeling
- **Annihilation Processes**: Modeling of electron-positron annihilation dynamics
- **Quantum Computing Integration**: Built-in Qiskit integration for quantum simulations of quantimatter systems
- **Validation Tools**: Framework for verifying results against known theoretical benchmarks

## Installation

### Basic Installation

```bash
pip install quantimatter
```

### Installation with Quantum Computing Support

```bash
pip install quantimatter[qiskit]
```

## Quick Start

```python
import numpy as np
from quantimatter.core.molecular_data import MolecularData
from quantimatter.utils import create_quantimatter_calculation

# Create a positronium system (electron-positron bound state)
positronium = MolecularData.positronium()

# Configure and run the calculation
result = create_quantimatter_calculation(
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

The package includes several example scripts for common quantimatter research scenarios:

- `examples/positronium_simulation.py`: Basic positronium energy calculation
- `examples/quantum_quantimatter.py`: Quantum computing simulation of positronium
- `examples/anti_hydrogen.py`: Anti-hydrogen atom calculations

## Citing This Work

If you use this package in your research, please cite:

```
@software{quantimatter,
  author = {mk0dz},
  title = {quantimatter Quantum Chemistry},
  url = {https://github.com/mk0dz/quantimatter},
  version = {0.1.0},
  year = {2023},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.