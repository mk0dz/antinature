from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="antinature",
    version="0.1.0",  # Increment version since you're making changes
    packages=find_packages(),
    py_modules=['antinature'],  # Removed duplicate 'antinature'
    install_requires=[
        "numpy>=1.20.0,<3.0.0",      # Wider range, avoiding breaking changes
        "scipy>=1.5.2,<3.0.0",       # Relaxed upper bound
        "matplotlib>=3.4.0,<4.0.0",  # More flexible but still stable
        "typing-extensions>=4.0.0,<5.0.0",  # Added upper bound for safety
    ],
    extras_require={
        "qiskit": [
            "qiskit>=0.42.0,<2.0.0",  # Broader range for better compatibility
            "qiskit-algorithms>=0.2.0,<1.0.0",
            "qiskit-nature>=0.6.0,<1.0.0",
            "qiskit-aer>=0.11.0,<1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0,<9.0.0",
            "pytest-cov>=4.0.0,<6.0.0",
            "black>=23.0.0,<25.0.0",
            "isort>=5.12.0,<7.0.0",
        ],
        # Allow installing both dev and qiskit extras together
        "all": [
            "qiskit>=0.42.0,<2.0.0",  # Broader range for better compatibility
            "qiskit-algorithms>=0.2.0,<1.0.0",
            "qiskit-nature>=0.6.0,<1.0.0",
            "qiskit-aer>=0.11.0,<1.0.0",
            "pytest>=7.0.0,<9.0.0",
            "pytest-cov>=4.0.0,<6.0.0",
            "black>=23.0.0,<25.0.0",
            "isort>=5.12.0,<7.0.0",
        ],
    },
    author="Mukul Kumar",
    author_email="Mukulpal108@hotmail.com",
    description="Quantum chemistry package for antimatter simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mk0dz/antinature",
    project_urls={
        "Bug Tracker": "https://github.com/mk0dz/antinature/issues",
        "Documentation": "https://github.com/mk0dz/antinature",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8,<3.14",  # Updated upper bound for Python version
)
