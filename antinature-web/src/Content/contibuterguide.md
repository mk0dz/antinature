# Contributor Guide

Thank you for your interest in contributing to the Antinature framework! This guide provides information on how to get started as a contributor and outlines our development practices.

## Getting Started

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**

   ```bash
   git clone https://github.com/YOUR-USERNAME/antinature.git
   cd antinature
   ```

2. **Install Development Dependencies**

   ```bash
   pip install -e .[dev,all]
   ```

3. **Set Up Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

### Development Workflow

1. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Write code that follows our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Tests Locally**

   ```bash
   pytest
   ```

4. **Submit a Pull Request**
   - Push your changes to your fork
   - Open a pull request against the main repository
   - Fill out the PR template with details about your changes

## Coding Standards

### Code Style

We follow PEP 8 with a few modifications:

- Line length limit is 100 characters
- Use 4 spaces for indentation (no tabs)
- Use docstrings for all public functions, classes, and methods

Our pre-commit hooks will automatically check and fix many style issues. Additionally, we use:

- Black for code formatting
- isort for import sorting
- flake8 for linting

### Documentation

All code should be well-documented:

- Every module, class, method, and function should have a docstring
- Use NumPy-style docstrings
- Include examples where appropriate
- Update the relevant documentation files when adding new features

### Testing

We use pytest for testing. All new code should include appropriate tests:

- Unit tests for individual functions and methods
- Integration tests for more complex interactions
- Benchmarks for performance-critical code

## Areas for Contribution

We welcome contributions in several areas:

### Code Contributions

- **Core Framework Improvements**
  - Optimizing performance of integral calculations
  - Implementing new correlation methods
  - Enhancing relativistic treatments

- **New Features**
  - Additional antimatter particle types
  - Novel visualization tools
  - Integration with other quantum chemistry packages

- **Bug Fixes**
  - Addressing issues in the issue tracker
  - Improving error handling
  - Fixing edge cases

### Documentation Contributions

- Improving existing documentation
- Adding new examples and tutorials
- Creating educational content

### Community Contributions

- Answering questions in discussions
- Reviewing pull requests
- Helping with issue triage

## Specialized Areas

### Basis Set Development

We particularly welcome contributions of specialized basis sets for antimatter systems:

1. Create a directory in `antinature/data/basis_sets/positron/`
2. Follow the format structure used in existing basis sets
3. Include a README.md with:
   - Description of the basis set
   - References to publications
   - Benchmarks showing performance

### Validation Test Cases

Help us verify the accuracy of Antinature by contributing validation cases:

1. Find published data on antimatter systems (e.g., positronium, positron-molecule binding)
2. Create a test in `tests/validation/` that compares Antinature results to these benchmarks
3. Document the source of the reference data

## Release Process

Our release cycle follows these steps:

1. Feature freeze and creation of a release branch
2. Final testing and bug fixing
3. Documentation updates
4. Version bump and CHANGELOG update
5. Release to PyPI
6. Post-release announcement

## Getting Help

If you need assistance with your contribution:

- Open a discussion on GitHub
- Ask in our community chat
- Contact the maintainers directly

## Code of Conduct

All contributors are expected to adhere to our [Code of Conduct](../CODE_OF_CONDUCT.md). We aim to foster an inclusive and welcoming community.

Thank you for contributing to Antinature and helping advance the field of antimatter quantum chemistry!