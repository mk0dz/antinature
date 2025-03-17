# test_core_modules.py

import numpy as np
import matplotlib.pyplot as plt

from antinature.core.molecular_data import MolecularData
from antinature.core.basis import GaussianBasisFunction, BasisSet, PositronBasis, MixedMatterBasis
from antinature.core.integral_engine import antinatureIntegralEngine
from antinature.core.hamiltonian import antinatureHamiltonian
from antinature.core.scf import antinatureSCF
from antinature.core.correlation import antinatureCorrelation

def test_molecular_data():
    print("\n=== Testing MolecularData ===")
    
    # Create a water molecule
    atoms = [
        ('O', np.array([0.0, 0.0, 0.0])),
        ('H', np.array([0.0, 1.43, -1.1])),
        ('H', np.array([0.0, -1.43, -1.1]))
    ]
    
    water = MolecularData(
        atoms=atoms,
        n_electrons=10,
        n_positrons=0,
        charge=0,
        name="Water",
        description="H2O molecule",
        units='bohr'
    )
    
    print(water)
    print(f"Center of mass: {water.get_center_of_mass()}")
    print(f"Formula: {water.get_formula()}")
    print(f"Interatomic distances:")
    print(water.get_interatomic_distances())
    print(f"Bonds: {water.get_bonds()}")
    
    # Create a positronium
    ps = MolecularData.positronium()
    print("\nPositronium:")
    print(ps)
    
    # Visualize
    water.visualize(show_bonds=True)
    ps.visualize()
    
    return water, ps
def test_basis_set():
    print("\n=== Testing Basis Set ===")
    
    # Create a simple basis function
    center = np.array([0.0, 0.0, 0.0])
    bf = GaussianBasisFunction(center, 1.0, (0, 0, 0))
    
    # Test evaluation
    point = np.array([0.5, 0.0, 0.0])
    value = bf.evaluate(point)
    print(f"Basis function value at {point}: {value}")
    
    # Test basis set creation
    basis = BasisSet()
    basis.add_function(bf)
    basis.create_for_atom('H', center, quality='standard')
    
    print(f"Basis set size: {basis.n_basis}")
    print(f"Basis functions: {len(basis.basis_functions)}")
    
    # Test positron basis
    p_basis = PositronBasis()
    p_basis.create_for_atom('H', center, quality='standard')
    
    print(f"Positron basis set size: {p_basis.n_basis}")
    
    # Test mixed basis
    mixed_basis = MixedMatterBasis(basis, p_basis)
    print(f"Mixed basis total size: {mixed_basis.n_total_basis}")
    print(f"Electron basis size: {mixed_basis.n_electron_basis}")
    print(f"Positron basis size: {mixed_basis.n_positron_basis}")
    
    # Create basis for a molecule
    water, _ = test_molecular_data()
    mixed_basis = MixedMatterBasis()
    try:
        # Use 'standard' instead of 'minimal'
        mixed_basis.create_for_molecule(water.atoms, 'standard', 'standard')
        print(f"Water molecule basis:")
        print(f"  Total size: {mixed_basis.n_total_basis}")
        print(f"  Electron basis: {mixed_basis.n_electron_basis}")
        print(f"  Positron basis: {mixed_basis.n_positron_basis}")
    except Exception as e:
        print(f"Error creating basis for water molecule: {str(e)}")
    
    return mixed_basis, water

def test_integral_engine():
    print("\n=== Testing Integral Engine ===")
    
    # Create integral engine
    integral_engine = antinatureIntegralEngine(use_analytical=True, cache_size=1000)
    
    # Get basis functions
    center1 = np.array([0.0, 0.0, 0.0])
    center2 = np.array([1.0, 0.0, 0.0])
    bf1 = GaussianBasisFunction(center1, 1.0, (0, 0, 0))
    bf2 = GaussianBasisFunction(center2, 1.0, (0, 0, 0))
    
    # Calculate overlap
    overlap = integral_engine._overlap_integral_impl(id(bf1), id(bf2), bf1, bf2)
    print(f"Overlap integral: {overlap}")
    
    # Calculate kinetic
    kinetic = integral_engine._kinetic_integral_impl(id(bf1), id(bf2), bf1, bf2)
    print(f"Kinetic integral: {kinetic}")
    
    # Calculate nuclear attraction
    nuclear_pos = np.array([0.5, 0.0, 0.0])
    nuclear = integral_engine._nuclear_attraction_integral_impl(id(bf1), id(bf2), bf1, bf2, nuclear_pos)
    print(f"Nuclear attraction integral: {nuclear}")
    
    return integral_engine

def test_hamiltonian():
    print("\n=== Testing Hamiltonian ===")
    
    # Get basis and molecule
    mixed_basis, water = test_basis_set()
    
    # Create integral engine
    integral_engine = test_integral_engine()
    
    # Create Hamiltonian
    hamiltonian = antinatureHamiltonian(
        molecular_data=water,
        basis_set=mixed_basis,
        integral_engine=integral_engine,
        include_annihilation=True,
        include_relativistic=False
    )
    
    # Build Hamiltonian components
    matrices = hamiltonian.build_hamiltonian()
    
    print(f"Hamiltonian components:")
    for key, matrix in matrices.items():
        if isinstance(matrix, np.ndarray):
            shape_str = f"shape={matrix.shape}"
        else:
            shape_str = "None"
        print(f"  {key}: {shape_str}")
    
    # Get overlap matrix
    S = matrices.get('overlap')
    if S is not None:
        print(f"Overlap matrix first few elements:")
        print(S[:3, :3])
    
    # Get core Hamiltonian
    H_core = matrices.get('H_core_electron')
    if H_core is not None:
        print(f"Core Hamiltonian first few elements:")
        print(H_core[:3, :3])
    
    return hamiltonian, matrices

def test_scf():
    print("\n=== Testing SCF ===")
    
    # Create a hydrogen molecule instead of using water
    h2 = MolecularData(
        atoms=[
            ('H', np.array([0.0, 0.0, 0.0])),
            ('H', np.array([0.0, 0.0, 1.4]))  # 1.4 Bohr H-H distance
        ],
        n_electrons=2,
        n_positrons=0,
        charge=0,
        name="Hydrogen",
        description="H2 molecule"
    )
    
    # Create basis set directly without relying on test_basis_set
    mixed_basis = MixedMatterBasis()
    mixed_basis.create_for_molecule(h2.atoms, 'standard', 'standard')
    
    # Create integral engine
    integral_engine = antinatureIntegralEngine(use_analytical=True, cache_size=1000)
    
    # Create Hamiltonian
    hamiltonian = antinatureHamiltonian(
        molecular_data=h2,
        basis_set=mixed_basis,
        integral_engine=integral_engine,
        include_annihilation=True,
        include_relativistic=False
    )
    
    # Build Hamiltonian components
    matrices = hamiltonian.build_hamiltonian()
    
    # Check if matrices are valid before proceeding
    if matrices['overlap'].shape[0] == 0 or matrices['H_core_electron'].shape[0] == 0:
        print("Warning: Empty matrices generated. Check basis set parameters.")
        print("Skipping SCF calculation.")
        return
    
    # Create SCF solver
    scf = antinatureSCF(
        hamiltonian=matrices,
        basis_set=mixed_basis,
        molecular_data=h2,
        max_iterations=10,
        convergence_threshold=1e-4,
        use_diis=True,
        damping_factor=0.5
    )
    
    # Run SCF calculation
    print("Running SCF calculation...")
    try:
        results = scf.solve_scf()
        
        print("\nSCF Results:")
        print(f"  Energy: {results.get('energy', 0.0):.10f} Hartree")
        print(f"  Converged: {results.get('converged', False)}")
        print(f"  Iterations: {results.get('iterations', 0)}")
        
        # Additional diagnostic information
        if 'orbital_energies' in results:
            print("\nOrbital Energies:")
            for i, energy in enumerate(results['orbital_energies']):
                print(f"  ε{i}: {energy:.6f} Hartree")
    except Exception as e:
        print(f"Error in SCF calculation: {str(e)}")
    
    return scf, results

def run_all_tests():
    test_molecular_data()
    test_basis_set()
    test_integral_engine()
    test_hamiltonian()
    test_scf()
    # test_correlation() - Not implemented yet

if __name__ == "__main__":
    run_all_tests()