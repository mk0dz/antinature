#!/usr/bin/env python3
"""
Comprehensive test of antinature framework status.
Tests all major systems and reports results.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings('ignore')

from antinature import MolecularData, AntinatureCalculator
from antinature.specialized.systems import AntinatureSystems
import numpy as np


def test_system_helper(name, mol_data, expected_energy, tolerance=0.5):
    """Test a single system and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        calc = AntinatureCalculator(print_level=0)
        result = calc.calculate_custom_system(mol_data, accuracy='medium')
        
        energy = result['energy']
        converged = result.get('converged', False)
        error = abs(energy - expected_energy)
        passed = error < tolerance and converged
        
        print(f"Energy: {energy:.6f} Hartree")
        print(f"Expected: {expected_energy:.6f} Hartree")
        print(f"Error: {error:.6f} Hartree")
        print(f"Converged: {converged}")
        print(f"Status: {'✓ PASSED' if passed else '✗ FAILED'}")
        
        return passed, energy, error
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False, None, None


def main():
    """Run comprehensive tests."""
    print("ANTINATURE FRAMEWORK STATUS CHECK")
    print("="*60)
    
    results = {}
    
    # Test 1: Positronium (should work perfectly)
    ps = AntinatureSystems.positronium()
    passed, energy, error = test_system_helper("Positronium", ps, -0.25, tolerance=0.01)
    results['positronium'] = (passed, energy, error)
    
    # Test 2: Anti-hydrogen
    anti_h = AntinatureSystems.anti_hydrogen()
    passed, energy, error = test_system_helper("Anti-hydrogen", anti_h, -0.5, tolerance=0.5)
    results['anti_hydrogen'] = (passed, energy, error)
    
    # Test 3: Positronium molecule (Ps2)
    ps2 = AntinatureSystems.positronium_molecule()
    passed, energy, error = test_system_helper("Positronium Molecule (Ps₂)", ps2, -0.516, tolerance=0.1)
    results['ps2'] = (passed, energy, error)
    
    # Test 4: Positronium hydride (PsH)
    psh = AntinatureSystems.positronium_hydride()
    passed, energy, error = test_system_helper("Positronium Hydride (PsH)", psh, -0.789, tolerance=0.5)
    results['psh'] = (passed, energy, error)
    
    # Test 5: Muonium
    mu = AntinatureSystems.muonium()
    passed, energy, error = test_system_helper("Muonium", mu, -0.4975, tolerance=0.5)
    results['muonium'] = (passed, energy, error)
    
    # Test 6: Antimuonium
    antimu = AntinatureSystems.antimuonium()
    passed, energy, error = test_system_helper("Antimuonium", antimu, -0.4975, tolerance=0.5)
    results['antimuonium'] = (passed, energy, error)
    
    # Test 7: Anti-He+
    anti_he = AntinatureSystems.anti_helium_ion()
    passed, energy, error = test_system_helper("Anti-He⁺", anti_he, -2.0, tolerance=1.0)
    results['anti_he'] = (passed, energy, error)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total = len(results)
    passed_count = sum(1 for p, _, _ in results.values() if p)
    
    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed_count}/{total} ({100*passed_count/total:.1f}%)")
    print(f"Failed: {total-passed_count}/{total} ({100*(total-passed_count)/total:.1f}%)")
    
    print("\nDetailed Results:")
    print("-"*40)
    for name, (passed, energy, error) in results.items():
        status = "✓" if passed else "✗"
        energy_str = f"{energy:.6f}" if energy is not None else "ERROR"
        print(f"{status} {name:20s}: E = {energy_str:12s} Ha")
    
    # Overall assessment
    print("\n" + "="*60)
    if passed_count >= total * 0.8:
        print("✓ FRAMEWORK STATUS: GOOD (≥80% passing)")
    elif passed_count >= total * 0.6:
        print("○ FRAMEWORK STATUS: FAIR (≥60% passing)")
    else:
        print("✗ FRAMEWORK STATUS: NEEDS WORK (<60% passing)")
    
    return passed_count / total


def test_system():
    """Pytest test function"""
    success_rate = main()
    assert success_rate >= 0.6, f"Framework success rate {success_rate:.1%} is below 60%"

if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 0.6 else 1)