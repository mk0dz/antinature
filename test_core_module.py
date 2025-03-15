# test_core_modules.py

import numpy as np
import matplotlib.pyplot as plt

from antimatter_qchem.core.molecular_data import MolecularData
from antimatter_qchem.core.basis import GaussianBasisFunction, BasisSet, PositronBasis, MixedMatterBasis
from antimatter_qchem.core.integral_engine import AntimatterIntegralEngine
from antimatter_qchem.core.hamiltonian import AntimatterHamiltonian
from antimatter_qchem.core.scf import AntimatterSCF
from antimatter_qchem.core.correlation import AntimatterCorrelation

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
    mixed_basis.create_for_molecule(water.atoms, 'minimal', 'minimal')
    
    print(f"Water molecule basis:")
    print(f"  Total size: {mixed_basis.n_total_basis}")
    print(f"  Electron basis: {mixed_basis.n_electron_basis}")
    print(f"  Positron basis: {mixed_basis.n_positron_basis}")
    
    return mixed_basis, water

def test_integral_engine():
    print("\n=== Testing Integral Engine ===")
    
    # Create integral engine
    integral_engine = AntimatterIntegralEngine(use_analytical=True, cache_size=1000)
    
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
    hamiltonian = AntimatterHamiltonian(
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
    
    # Get Hamiltonian
    hamiltonian, matrices = test_hamiltonian()
    
    # Create SCF solver
    scf = AntimatterSCF(
        hamiltonian=matrices,
        basis_set=hamiltonian.basis_set,
        molecular_data=hamiltonian.molecular_data,
        max_iterations=10,
        convergence_threshold=1e-4,
        use_diis=True,
        damping_factor=0.5
    )
    
    # Run SCF calculation
    print("Running SCF calculation...")
    results = scf.solve_scf()
    
    print("\nSCF Results:")
    print(f"  Energy: {results.get('energy', 0.0):.10f} Hartree")
    print(f"  Converged: {results.get('converged', False)}")
    print(f"  Iterations: {results.get('iterations', 0)}")
    print(f"  Computation time: {results.get('computation_time', 0.0):.2f} seconds")
    
    # Get orbital energies
    E_e = results.get('E_electron')
    if E_e is not None:
        print("\nElectron orbital energies:")
        print(E_e[:5])  # First 5 orbitals
    
    return scf, results

def test_correlation():
    print("\n=== Testing Correlation ===")
    
    # Get SCF results
    scf, results = test_scf()
    
    # Create correlation calculator
    correlation = AntimatterCorrelation(
        scf_result=results,
        hamiltonian=scf.hamiltonian,
        basis=scf.basis_set
    )
    
    # Calculate MP2 energy
    try:
        mp2_energy = correlation.mp2_energy()
        print(f"MP2 correlation energy: {mp2_energy:.10f} Hartree")
        print(f"Total energy (SCF + MP2): {results.get('energy', 0.0) + mp2_energy:.10f} Hartree")
    except Exception as e:
        print(f"MP2 calculation failed: {str(e)}")
    
    # Calculate annihilation rate
    try:
        ann_rate = correlation.calculate_annihilation_rate()
        print(f"Annihilation rate: {ann_rate:.10e} a.u.")
    except Exception as e:
        print(f"Annihilation rate calculation failed: {str(e)}")
    
    return correlation

def run_all_tests():
    test_molecular_data()
    test_basis_set()
    test_integral_engine()
    test_hamiltonian()
    test_scf()
    test_correlation()

if __name__ == "__main__":
    run_all_tests()