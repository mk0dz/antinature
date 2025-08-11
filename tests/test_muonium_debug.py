import sys
sys.path.insert(0, '/home/mk/dirac/antinature')

from antinature.specialized.systems import AntinatureSystems
from antinature import AntinatureCalculator

print("Testing Muonium with debugging...")
mu = AntinatureSystems.muonium()
print(f"System: {mu.name}")
print(f"Atoms: {mu.atoms}")
print(f"n_electrons: {mu.n_electrons}")
print(f"n_positrons: {mu.n_positrons}")
print(f"charge: {mu.charge}")
print(f"nuclei: {mu.nuclei}")
print(f"nuclear_charges: {mu.nuclear_charges}")

calc = AntinatureCalculator(print_level=2)
result = calc.calculate_custom_system(mu, accuracy='low')
print(f"\nFinal energy: {result['energy']:.6f} Ha")