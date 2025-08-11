# Release Notes

## Current Version: 0.1.0 (2023)

Antinature's initial release provides a foundational framework for antimatter quantum chemistry simulations. This version focuses on establishing core functionality while ensuring a stable and extensible codebase.

### Key Features

- **Core Framework Implementation**
  - Complete electronic and positronic Hamiltonian construction
  - Self-consistent field methods for mixed matter-antimatter systems
  - Specialized basis sets optimized for positronic systems
  - Integral evaluation engine for antimatter interactions

- **Positronium Simulation**
  - Ground and excited state energy calculations
  - Annihilation rate prediction
  - Wavefunction and density visualization

- **Small Mixed Systems**
  - Support for hydrogen and helium with positrons
  - Basic geometry optimization for mixed systems
  - Positron binding energy calculations

- **Analysis Tools**
  - Calculation of annihilation rates and lifetimes
  - Basic visualization of electron and positron densities
  - Property extraction and analysis

### Known Limitations

- Limited support for systems with more than one positron
- Performance optimization needed for larger molecular systems
- Relativistic corrections are first-order only
- Limited correlation methods for mixed systems

## Upcoming Features

### Version 0.2.0 (Planned)

- **Enhanced Correlation Methods**
  - Explicitly correlated MP2 for electron-positron systems
  - Configuration interaction with annihilation channels
  - Coupled-cluster methods adapted for antimatter

- **Expanded Basis Set Library**
  - Additional positron-optimized basis sets
  - Auto-generated explicitly correlated basis functions
  - Basis set optimization tools

- **Advanced Relativistic Treatments**
  - Full two-component relativistic implementation
  - Breit interaction for relativistic electron-positron systems
  - QED corrections for high-precision calculations

- **Performance Improvements**
  - Integral screening for faster calculations
  - Parallel implementation of key computational bottlenecks
  - GPU acceleration for integral evaluation

### Version 0.3.0 (Roadmap)

- **Expanded Particle Types**
  - Support for antiprotons and antihydrogen
  - Muonic and muon-antimuon systems
  - Antiprotonic molecules and atoms

- **Dynamics and Time Evolution**
  - Time-dependent calculations for positronic systems
  - Annihilation dynamics simulations
  - Non-adiabatic coupling between electronic and positronic states

- **Advanced Quantum Computing Integration**
  - Specialized quantum algorithms for antimatter simulation
  - Hybrid classical-quantum approaches for larger systems
  - Error mitigation techniques for realistic quantum hardware

- **Machine Learning Acceleration**
  - ML potentials for antimatter-matter interactions
  - Neural network wavefunctions for positronic systems
  - Acceleration of property prediction with ML models

## Developer Notes

- The API is still considered experimental and may change in future versions
- We welcome community contributions, especially for specialized basis sets and validation test cases
- Performance profiling and optimization is an ongoing effort

For a detailed changelog of all versions, please see the [CHANGELOG.md](../CHANGELOG.md) file in the repository root.