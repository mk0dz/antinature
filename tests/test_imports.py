#!/usr/bin/env python3
"""
Simple script to verify imports from antinature package
"""

import sys

try:
    # Basic imports
    import antinature
    from antinature.core.molecular_data import MolecularData
    from antinature.core.basis import MixedMatterBasis
    
    print("✅ Imports working correctly!")
    print(f"Package version: {antinature.__version__}")
    
    # Create a simple positronium system
    pos = MolecularData.positronium()
    print(f"✅ Created positronium system: {pos}")
    
    # Create a mixed basis
    basis = MixedMatterBasis()
    print(f"✅ Created basis object: {basis}")
    
    print("\n🎉 SUCCESS: Package imports are working correctly!")
    
except ImportError as e:
    print(f"❌ ERROR: Import failed: {e}")
    print("Check package installation and import paths.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 