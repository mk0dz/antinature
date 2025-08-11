#!/usr/bin/env python
"""
Comprehensive test suite for all Antinature functionality.
This tests every major function and class in the codebase.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoreModules:
    """Test all core module functionality."""
    
    def test_molecular_data_creation(self):
        """Test MolecularData class creation and methods."""
        from antinature.core.molecular_data import MolecularData
        
        # Test basic molecular data
        h2_data = MolecularData(
            atoms=[('H', np.array([0, 0, 0])), ('H', np.array([0, 0, 1.4]))],
            n_electrons=2,
            n_positrons=0,
            charge=0,
            name='H2'
        )
        assert h2_data.n_electrons == 2
        assert h2_data.n_positrons == 0
        assert h2_data.get_formula() == 'H2'
        assert h2_data.get_nuclear_repulsion_energy() > 0
        
        # Test positronium
        ps_data = MolecularData.positronium()
        assert ps_data.is_positronium == True
        assert ps_data.n_electrons == 1
        assert ps_data.n_positrons == 1
        
        # Test anti-hydrogen
        ah_data = MolecularData.anti_hydrogen()
        assert ah_data.n_positrons == 1
        assert 'anti' in ah_data.name.lower()
        
    def test_basis_functions(self):
        """Test GaussianBasisFunction and BasisSet classes."""
        from antinature.core.basis import GaussianBasisFunction, BasisSet, PositronBasis, MixedMatterBasis
        
        # Test Gaussian basis function
        basis_func = GaussianBasisFunction(
            center=np.array([0, 0, 0]),
            exponent=1.0,
            angular_momentum=(0, 0, 0)
        )
        assert basis_func.get_type() == 's'
        value = basis_func.evaluate(np.array([0.5, 0, 0]))
        assert isinstance(value, float)
        
        # Test BasisSet
        basis_set = BasisSet(name='test')
        basis_set.add_function(basis_func)
        assert len(basis_set) == 1
        assert basis_set.n_basis == 1
        
        # Test PositronBasis
        pos_basis = PositronBasis(name='positron_test')
        pos_basis.create_positron_orbital_basis(np.array([0, 0, 0]), quality='minimal')
        assert pos_basis.n_basis > 0
        
        # Test MixedMatterBasis
        mixed_basis = MixedMatterBasis()
        mixed_basis.create_optimized_positronium_basis(target_accuracy='low')
        assert mixed_basis.n_electron_basis > 0
        assert mixed_basis.n_positron_basis > 0
        
    def test_integral_engine(self):
        """Test integral calculation engine."""
        from antinature.core.integral_engine import AntinatureIntegralEngine
        from antinature.core.basis import GaussianBasisFunction
        
        engine = AntinatureIntegralEngine()
        
        # Create test basis functions
        basis1 = GaussianBasisFunction(
            center=np.array([0, 0, 0]),
            exponent=1.0,
            angular_momentum=(0, 0, 0)
        )
        basis2 = GaussianBasisFunction(
            center=np.array([0, 0, 1.0]),
            exponent=0.8,
            angular_momentum=(0, 0, 0)
        )
        
        # Test overlap integral
        overlap = engine.overlap_integral(basis1, basis2)
        assert 0 <= overlap <= 1
        
        # Test kinetic integral
        kinetic = engine.kinetic_integral(basis1, basis2)
        assert isinstance(kinetic, float)
        
        # Test nuclear attraction
        nuclear = engine.nuclear_attraction_integral(
            basis1, basis2, np.array([0, 0, 0.5])
        )
        assert nuclear < 0  # Attraction should be negative
        
    def test_hamiltonian(self):
        """Test Hamiltonian construction."""
        from antinature.core.hamiltonian import AntinatureHamiltonian
        from antinature.core.molecular_data import MolecularData
        from antinature.core.basis import MixedMatterBasis
        
        # Create positronium system
        ps_data = MolecularData.positronium()
        basis = MixedMatterBasis()
        basis.create_positronium_basis(quality='low')
        
        # Build Hamiltonian
        from antinature.core.integral_engine import AntinatureIntegralEngine
        engine = AntinatureIntegralEngine()
        ham = AntinatureHamiltonian(ps_data, basis, engine)
        matrices = ham.build_hamiltonian()
        
        assert 'overlap' in matrices
        assert 'H_core_electron' in matrices
        assert 'H_core_positron' in matrices
        
    def test_scf_solver(self):
        """Test SCF solver functionality."""
        from antinature.core.scf import AntinatureSCF
        from antinature.core.molecular_data import MolecularData
        from antinature.core.basis import MixedMatterBasis
        from antinature.core.hamiltonian import AntinatureHamiltonian
        
        # Create simple system
        ps_data = MolecularData.positronium()
        basis = MixedMatterBasis()
        basis.create_positronium_basis(quality='low')
        
        # Build Hamiltonian
        from antinature.core.integral_engine import AntinatureIntegralEngine
        engine = AntinatureIntegralEngine()
        ham_obj = AntinatureHamiltonian(ps_data, basis, engine)
        ham_matrices = ham_obj.build_hamiltonian()
        
        # Run SCF
        scf = AntinatureSCF(
            hamiltonian=ham_matrices,
            basis_set=basis,
            molecular_data=ps_data,
            max_iterations=10
        )
        results = scf.run()
        
        assert results['converged'] == True
        assert 'energy' in results
        assert results['energy'] < 0
        

class TestSpecializedModules:
    """Test specialized physics modules."""
    
    def test_annihilation_operator(self):
        """Test annihilation operator functionality."""
        from antinature.specialized.annihilation import AnnihilationOperator
        
        ann_op = AnnihilationOperator(dimension=10)
        
        # Test rate calculation
        rate = ann_op.calculate_annihilation_rate(
            electron_density=1.0,
            positron_density=1.0,
            overlap=0.5
        )
        assert rate > 0
        
        # Test cross section
        cross_section = ann_op.calculate_cross_section(energy=0.511)  # MeV
        assert cross_section > 0
        
    def test_positronium_scf(self):
        """Test specialized positronium SCF."""
        from antinature.specialized.positronium import PositroniumSCF
        
        ps_scf = PositroniumSCF()
        results = ps_scf.solve(state='para')
        
        assert 'energy' in results
        assert results['converged'] == True
        # Positronium ground state should be around -0.25 Hartree
        assert -0.3 < results['energy'] < -0.2
        
    def test_relativistic_corrections(self):
        """Test relativistic correction calculations."""
        from antinature.specialized.relativistic import RelativisticCorrection
        
        rel_corr = RelativisticCorrection(fine_structure_constant=1/137.036)
        
        # Test Darwin term
        darwin = rel_corr.calculate_darwin_term(
            wavefunction=np.ones((10, 10)),
            nuclear_positions=[np.array([0, 0, 0])]
        )
        assert isinstance(darwin, float)
        
        # Test spin-orbit coupling
        so_coupling = rel_corr.calculate_spin_orbit_coupling(
            orbital_angular_momentum=1,
            spin=0.5
        )
        assert isinstance(so_coupling, float)
        

class TestUtilities:
    """Test utility functions and calculators."""
    
    def test_antinature_calculator(self):
        """Test high-level calculator interface."""
        from antinature.utils import AntinatureCalculator
        
        calc = AntinatureCalculator(print_level=0)
        
        # Test positronium calculation
        ps_results = calc.calculate_positronium(accuracy='low')
        assert 'energy' in ps_results
        assert ps_results['converged'] == True
        
    def test_convenience_functions(self):
        """Test convenience functions."""
        from antinature.utils import calculate_positronium, calculate_antihydrogen, quick_test
        
        # Test quick test
        assert quick_test() == True
        
        # Test positronium calculation
        ps_result = calculate_positronium(accuracy='low')
        assert 'energy' in ps_result
        
        # Test antihydrogen calculation
        ah_result = calculate_antihydrogen(accuracy='low')
        assert 'energy' in ah_result
        
    def test_create_antinature_calculation(self):
        """Test calculator factory function."""
        from antinature.utils import create_antinature_calculation
        
        # Create positronium calculator
        calc = create_antinature_calculation('positronium', accuracy='low')
        assert calc is not None
        
        # Create custom calculator
        calc_custom = create_antinature_calculation('custom', n_positrons=2)
        assert calc_custom is not None
        

class TestVisualization:
    """Test visualization components."""
    
    def test_visualizer_creation(self):
        """Test AntinatureVisualizer class."""
        from antinature.specialized.visualization import AntinatureVisualizer
        
        viz = AntinatureVisualizer(style='default')
        assert viz is not None
        assert hasattr(viz, 'plot_energy_convergence')
        assert hasattr(viz, 'plot_density')
        

@pytest.mark.requires_qiskit
class TestQiskitIntegration:
    """Test Qiskit integration modules (requires Qiskit)."""
    
    def test_ansatz_creation(self):
        """Test ansatz creation for quantum circuits."""
        try:
            from antinature.qiskit_integration.ansatze import AntinatureAnsatz, create_positronium_circuit
            
            # Test AntinatureAnsatz class
            ansatz = AntinatureAnsatz(num_qubits=2, reps=1)
            circuit = ansatz.create_positronium_ansatz()
            assert circuit.num_qubits == 2
            
            # Test standalone function
            circuit2 = create_positronium_circuit(reps=1)
            assert circuit2.num_qubits == 2
        except ImportError:
            pytest.skip("Qiskit not available")
    
    def test_quantum_systems(self):
        """Test quantum system definitions."""
        try:
            from antinature.qiskit_integration.systems import AntinatureQuantumSystems
            
            systems = AntinatureQuantumSystems()
            
            # Test positronium Hamiltonian
            ps_ham = systems.positronium()
            assert ps_ham is not None
        except ImportError:
            pytest.skip("Qiskit not available")
    
    def test_vqe_solver(self):
        """Test VQE solver functionality."""
        try:
            from antinature.qiskit_integration.vqe_solver import AntinatureVQESolver
            
            solver = AntinatureVQESolver()
            # Basic creation test
            assert solver is not None
        except ImportError:
            pytest.skip("Qiskit not available")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("COMPREHENSIVE ANTINATURE TEST SUITE")
    print("=" * 60)
    
    # Run pytest with verbose output
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    exit(run_all_tests())