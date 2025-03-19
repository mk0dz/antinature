# Contributing to Antinature

Thank you for your interest in contributing to Antinature! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/antinature.git
   cd antinature
   ```

3. Create a virtual environment and install dev dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

4. Set up pre-commit hooks (optional):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

Run the tests with pytest:

```bash
pytest
```

For test coverage:

```bash
pytest --cov=antinature
```

## Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- Flake8 for linting

You can run these tools locally:

```bash
black antinature tests
isort antinature tests
```

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with meaningful commit messages.

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request to the main repository.

5. Ensure all tests pass in the CI/CD pipeline.

## Release Process

The maintainers will handle the release process, which includes:

1. Updating the version number in:
   - `antinature/__init__.py`
   - `setup.py`
   - `CHANGELOG.md`
   - `README.md` citation

2. Creating a new GitHub release with release notes

3. Publishing to PyPI

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. 