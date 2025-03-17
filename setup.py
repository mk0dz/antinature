from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="antinature",
    version="0.1.0",
    packages=find_packages(),
    py_modules=['antinature', 'antinature'],
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.5.2,<2.0.0",
        "matplotlib>=3.4.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "qiskit": [
            "qiskit>=1.0.0",
            "qiskit-algorithms>=0.3.0",
            "qiskit-nature>=0.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    author="mk0dz",
    author_email="Mukulpal108@hotmail.com",  # Update with your email
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 4 - Beta",
    ],
    python_requires=">=3.8",
) 