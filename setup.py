from setuptools import setup, find_packages

setup(
    name="antimatter-beta",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib"
    ],
    extras_require={
        "qiskit": [
            "qiskit>=1.0.0",
            "qiskit-algorithms>=0.3.0",
            "qiskit-nature>=0.7.0",
        ],
    },
    author="mk0dz",
    description="Quantum chemistry package for antimatter simulations",
    python_requires=">=3.8",
) 