import sys
sys.path.insert(0, '/home/mk/dirac/antinature')

from antinature.specialized.systems import AntinatureSystems
from antinature import AntinatureCalculator

print("Testing PsH with debugging...")
psh = AntinatureSystems.positronium_hydride()
print(f"System: {psh.name}")
print(f"Atoms: {psh.atoms}")
print(f"n_electrons: {psh.n_electrons}")
print(f"n_positrons: {psh.n_positrons}")
print(f"charge: {psh.charge}")
print(f"nuclei: {psh.nuclei}")
print(f"nuclear_charges: {psh.nuclear_charges}")

calc = AntinatureCalculator(print_level=1)
result = calc.calculate_custom_system(psh, accuracy='low')
print(f"\nFinal energy: {result['energy']:.6f} Ha")
