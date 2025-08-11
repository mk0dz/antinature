import sys
sys.path.insert(0, '/home/mk/dirac/antinature')

from antinature.specialized.systems import AntinatureSystems

ps = AntinatureSystems.positronium()
print(f"n_electrons: {ps.n_electrons}")
print(f"n_positrons: {ps.n_positrons}")
print(f"is_positronium: {ps.is_positronium}")
print(f"atoms: {ps.atoms}")
