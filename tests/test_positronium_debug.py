import sys
sys.path.insert(0, '/home/mk/dirac/antinature')

from antinature import MolecularData, AntinatureCalculator

print("Testing Positronium with debugging...")
calc = AntinatureCalculator(print_level=2)
result = calc.calculate_positronium(accuracy='high')

print(f"\nFinal energy: {result['energy']:.6f} Ha")
print(f"Expected: -0.250000 Ha")
print(f"Error: {abs(result['energy'] - (-0.25)):.6f} Ha")
