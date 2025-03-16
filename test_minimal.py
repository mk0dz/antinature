#!/usr/bin/env python3
"""
Minimal Test Script for Antimatter-Beta Package
===============================================

This script directly imports specific core modules to test basic functionality
without going through the package's __init__.py which has Qiskit dependencies.
"""

import sys
import os
import numpy as np

def main():
    try:
        # Get the site-packages directory
        import site
        site_packages = site.getsitepackages()[0]
        
        # Check if the package is installed
        package_dir = os.path.join(site_packages, 'antimatter_qchem')
        if not os.path.exists(package_dir):
            print(f"❌ Package directory not found at {package_dir}")
            return 1
        
        print(f"✅ Package directory found at {package_dir}")
        
        # Import core modules directly using importlib
        import importlib.util
        
        # Import basis module
        basis_path = os.path.join(package_dir, 'core', 'basis.py')
        spec = importlib.util.spec_from_file_location('basis', basis_path)
        basis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(basis)
        
        print("✅ Successfully imported basis module")
        
        # Create a simple basis function with corrected parameters
        basis_func = basis.GaussianBasisFunction(
            center=np.array([0.0, 0.0, 0.0]),
            exponent=1.0,
            angular_momentum=(0, 0, 0)
            # normalization is optional
        )
        
        print(f"✅ Created basis function: {basis_func}")
        
        # Import molecular_data module
        mol_data_path = os.path.join(package_dir, 'core', 'molecular_data.py')
        spec = importlib.util.spec_from_file_location('molecular_data', mol_data_path)
        molecular_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(molecular_data)
        
        print("✅ Successfully imported molecular_data module")
        
        # Create a simple molecular system with corrected parameters
        # For a positron-electron pair, we'll create a simple system
        atoms = [
            ('e+', np.array([0.0, 0.0, 0.0])),  # Positron at origin
            ('e-', np.array([0.0, 0.0, 1.0]))   # Electron at 1 bohr distance
        ]
        
        mol = molecular_data.MolecularData(
            atoms=atoms,
            n_electrons=1,
            n_positrons=1,
            charge=0,
            multiplicity=1,
            description="Test Positronium System"
        )
        
        print(f"✅ Created molecular data object: {mol}")
        
        print("\n🎉 BASIC TEST SUCCESSFUL: Core components of antimatter-beta can be used! 🎉")
        print("\nNote: This test bypassed the package's automatic imports to avoid Qiskit dependencies.")
        print("For full functionality including Qiskit integration, install optional dependencies:")
        print("pip install 'antimatter-beta[qiskit]'")
        
        return 0
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 