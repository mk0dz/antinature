import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import time

class TestSuite:
    """
    Comprehensive test suite for antimatter quantum chemistry.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the test suite.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed output
        """
        self.verbose = verbose
        self.test_results = {}
        self.timing = {}
    
    def test_hartree_fock(self, 
                        molecule_generator,
                        scf_solver,
                        reference_energy: float,
                        tolerance: float = 0.1):
        """
        Test Hartree-Fock SCF calculation.
        
        Parameters:
        -----------
        molecule_generator : function
            Function that creates a molecule
        scf_solver : function
            Function that performs SCF calculation
        reference_energy : float
            Reference energy for validation
        tolerance : float
            Tolerance for energy comparison
            
        Returns:
        --------
        Dict
            Test results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Testing Hartree-Fock calculation:")
        
        # Create molecule
        molecule = molecule_generator()
        
        # Run SCF calculation
        scf_result = scf_solver(molecule)
        
        # Extract energy
        energy = scf_result.get('energy', 0.0)
        
        # Check convergence
        converged = scf_result.get('converged', False)
        
        # Validate against reference
        energy_diff = abs(energy - reference_energy)
        energy_valid = energy_diff <= tolerance
        
        if self.verbose:
            print(f"  SCF Energy: {energy:.6f}, Reference: {reference_energy:.6f}")
            print(f"  Difference: {energy_diff:.6f}, Tolerance: {tolerance:.6f}")
            print(f"  Converged: {converged}, Energy valid: {energy_valid}")
        
        result = {
            'energy': energy,
            'reference': reference_energy,
            'difference': energy_diff,
            'converged': converged,
            'energy_valid': energy_valid,
            'passed': converged and energy_valid,
            'test_name': 'hartree_fock'
        }
        
        self.test_results['hartree_fock'] = result
        
        end_time = time.time()
        self.timing['test_hartree_fock'] = end_time - start_time
        
        return result
    
    def test_positronium(self, 
                       positronium_generator,
                       solver,
                       properties_calculator):
        """
        Test positronium properties against theoretical values.
        
        Parameters:
        -----------
        positronium_generator : function
            Function that creates a positronium system
        solver : function
            Function that solves the system
        properties_calculator : function
            Function that calculates properties from the solution
            
        Returns:
        --------
        Dict
            Test results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Testing positronium properties:")
        
        # Known theoretical values for positronium
        references = {
            'energy': -0.25,  # Hartree
            'annihilation_rate': 0.5 * np.pi * (1/137.036)**2 * 137.036,  # a.u.
            'binding_energy_eV': 6.8  # eV
        }
        
        # Create positronium system
        positronium = positronium_generator()
        
        # Solve system
        solution = solver(positronium)
        
        # Calculate properties
        properties = properties_calculator(solution)
        
        # Extract key properties
        energy = properties.get('energy', 0.0)
        ann_rate = properties.get('annihilation_rate', 0.0)
        
        # Validate properties
        energy_diff = abs(energy - references['energy'])
        energy_valid = energy_diff <= 0.05  # 5% tolerance
        
        ann_diff = abs(ann_rate - references['annihilation_rate'])
        ann_valid = ann_diff <= 0.2 * references['annihilation_rate']  # 20% tolerance
        
        if self.verbose:
            print(f"  Energy: {energy:.6f}, Reference: {references['energy']:.6f}")
            print(f"  Energy difference: {energy_diff:.6f}, Valid: {energy_valid}")
            print(f"  Annihilation rate: {ann_rate:.6e}, Reference: {references['annihilation_rate']:.6e}")
            print(f"  Rate difference: {ann_diff:.6e}, Valid: {ann_valid}")
        
        result = {
            'energy': {
                'value': energy,
                'reference': references['energy'],
                'difference': energy_diff,
                'valid': energy_valid
            },
            'annihilation_rate': {
                'value': ann_rate,
                'reference': references['annihilation_rate'],
                'difference': ann_diff,
                'valid': ann_valid
            },
            'passed': energy_valid and ann_valid,
            'test_name': 'positronium'
        }
        
        self.test_results['positronium'] = result
        
        end_time = time.time()
        self.timing['test_positronium'] = end_time - start_time
        
        return result
    
    def test_relativistic_effects(self, 
                               system_generator,
                               relativistic_solver,
                               non_relativistic_solver):
        """
        Test relativistic corrections implementation.
        
        Parameters:
        -----------
        system_generator : function
            Function that creates a molecular system
        relativistic_solver : function
            Function that solves with relativistic corrections
        non_relativistic_solver : function
            Function that solves without relativistic corrections
            
        Returns:
        --------
        Dict
            Test results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Testing relativistic effects:")
        
        # Create molecular system
        system = system_generator()
        
        # Solve with and without relativistic corrections
        rel_result = relativistic_solver(system)
        non_rel_result = non_relativistic_solver(system)
        
        # Extract energies
        rel_energy = rel_result.get('energy', 0.0)
        non_rel_energy = non_rel_result.get('energy', 0.0)
        
        # Calculate energy difference
        energy_diff = rel_energy - non_rel_energy
        relative_diff = abs(energy_diff / non_rel_energy)
        
        # Check if difference is significant
        significant = relative_diff > 0.001  # 0.1% difference
        
        if self.verbose:
            print(f"  Relativistic energy: {rel_energy:.8f}")
            print(f"  Non-relativistic energy: {non_rel_energy:.8f}")
            print(f"  Difference: {energy_diff:.8f} ({relative_diff*100:.4f}%)")
            print(f"  Significant difference: {significant}")
        
        result = {
            'relativistic_energy': rel_energy,
            'non_relativistic_energy': non_rel_energy,
            'energy_difference': energy_diff,
            'relative_difference': relative_diff,
            'significant': significant,
            'passed': significant,  # Pass if relativistic effects are significant
            'test_name': 'relativistic_effects'
        }
        
        self.test_results['relativistic_effects'] = result
        
        end_time = time.time()
        self.timing['test_relativistic_effects'] = end_time - start_time
        
        return result
    
    def test_annihilation_physics(self,
                                annihilation_calculator,
                                distance_range: List[float],
                                expected_trend: str = 'decreasing'):
        """
        Test annihilation physics by varying distance.
        
        Parameters:
        -----------
        annihilation_calculator : function
            Function that calculates annihilation rate for a given distance
        distance_range : List[float]
            Range of distances to test
        expected_trend : str
            Expected trend of annihilation rate with distance
            
        Returns:
        --------
        Dict
            Test results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Testing annihilation physics with distance:")
        
        # Calculate annihilation rates at different distances
        distances = []
        rates = []
        
        for distance in distance_range:
            # Calculate rate
            rate = annihilation_calculator(distance)
            
            distances.append(distance)
            rates.append(rate)
            
            if self.verbose:
                print(f"  Distance: {distance:.2f}, Rate: {rate:.6e}")
        
        # Convert to numpy arrays
        distances = np.array(distances)
        rates = np.array(rates)
        
        # Calculate correlation to check trend
        correlation = np.corrcoef(distances, rates)[0, 1]
        
        # Determine if trend matches expectation
        if expected_trend == 'decreasing':
            trend_valid = correlation < -0.7  # Strong negative correlation
        elif expected_trend == 'increasing':
            trend_valid = correlation > 0.7  # Strong positive correlation
        else:
            trend_valid = abs(correlation) > 0.7  # Strong correlation either way
        
        if self.verbose:
            print(f"  Correlation: {correlation:.4f}")
            print(f"  Expected trend: {expected_trend}")
            print(f"  Trend valid: {trend_valid}")
        
        result = {
            'distances': distances.tolist(),
            'rates': rates.tolist(),
            'correlation': correlation,
            'expected_trend': expected_trend,
            'trend_valid': trend_valid,
            'passed': trend_valid,
            'test_name': 'annihilation_physics'
        }
        
        self.test_results['annihilation_physics'] = result
        
        end_time = time.time()
        self.timing['test_annihilation_physics'] = end_time - start_time
        
        return result
    
    def test_qiskit_integration(self,
                              system_generator,
                              classical_solver,
                              quantum_solver,
                              energy_tolerance: float = 0.1):
        """
        Test Qiskit integration by comparing with classical results.
        
        Parameters:
        -----------
        system_generator : function
            Function that creates a molecular system
        classical_solver : function
            Function that solves using classical methods
        quantum_solver : function
            Function that solves using quantum computing
        energy_tolerance : float
            Tolerance for energy comparison
            
        Returns:
        --------
        Dict
            Test results
        """
        start_time = time.time()
        
        if self.verbose:
            print("Testing Qiskit integration:")
        
        # Create system
        system = system_generator()
        
        # Solve using classical and quantum methods
        classical_result = classical_solver(system)
        quantum_result = quantum_solver(system)
        
        # Extract energies
        classical_energy = classical_result.get('energy', 0.0)
        quantum_energy = quantum_result.get('energy', 0.0)
        
        # Calculate difference
        energy_diff = abs(quantum_energy - classical_energy)
        relative_diff = energy_diff / abs(classical_energy) if classical_energy != 0 else float('inf')
        
        # Check if within tolerance
        energy_valid = energy_diff <= energy_tolerance
        
        if self.verbose:
            print(f"  Classical energy: {classical_energy:.8f}")
            print(f"  Quantum energy: {quantum_energy:.8f}")
            print(f"  Difference: {energy_diff:.8f} ({relative_diff*100:.4f}%)")
            print(f"  Tolerance: {energy_tolerance:.8f}")
            print(f"  Energy valid: {energy_valid}")
        
        result = {
            'classical_energy': classical_energy,
            'quantum_energy': quantum_energy,
            'energy_difference': energy_diff,
            'relative_difference': relative_diff,
            'energy_valid': energy_valid,
            'passed': energy_valid,
            'test_name': 'qiskit_integration'
        }
        
        self.test_results['qiskit_integration'] = result
        
        end_time = time.time()
        self.timing['test_qiskit_integration'] = end_time - start_time
        
        return result
    
    def run_all_tests(self, test_config: Dict):
        """
        Run all tests in the suite.
        
        Parameters:
        -----------
        test_config : Dict
            Configuration for all tests, including:
            - molecule_generators
            - solvers
            - calculators
            - reference values
            
        Returns:
        --------
        Dict
            Results of all tests
        """
        start_time = time.time()
        
        print("Running antimatter quantum chemistry test suite...")
        
        # Keep track of passed tests
        passed_tests = 0
        total_tests = 0
        
        # Run Hartree-Fock test if configured
        if 'hartree_fock' in test_config:
            print("\n1. Testing Hartree-Fock calculation...")
            config = test_config['hartree_fock']
            result = self.test_hartree_fock(
                config['molecule_generator'],
                config['solver'],
                config['reference_energy'],
                config.get('tolerance', 0.1)
            )
            if result['passed']:
                passed_tests += 1
            total_tests += 1
        
        # Run positronium test if configured
        if 'positronium' in test_config:
            print("\n2. Testing positronium properties...")
            config = test_config['positronium']
            result = self.test_positronium(
                config['positronium_generator'],
                config['solver'],
                config['properties_calculator']
            )
            if result['passed']:
                passed_tests += 1
            total_tests += 1
        
        # Run relativistic effects test if configured
        if 'relativistic_effects' in test_config:
            print("\n3. Testing relativistic effects...")
            config = test_config['relativistic_effects']
            result = self.test_relativistic_effects(
                config['system_generator'],
                config['relativistic_solver'],
                config['non_relativistic_solver']
            )
            if result['passed']:
                passed_tests += 1
            total_tests += 1
        
        # Run annihilation physics test if configured
        if 'annihilation_physics' in test_config:
            print("\n4. Testing annihilation physics...")
            config = test_config['annihilation_physics']
            result = self.test_annihilation_physics(
                config['annihilation_calculator'],
                config['distance_range'],
                config.get('expected_trend', 'decreasing')
            )
            if result['passed']:
                passed_tests += 1
            total_tests += 1
        
        # Run Qiskit integration test if configured
        if 'qiskit_integration' in test_config:
            print("\n5. Testing Qiskit integration...")
            config = test_config['qiskit_integration']
            result = self.test_qiskit_integration(
                config['system_generator'],
                config['classical_solver'],
                config['quantum_solver'],
                config.get('energy_tolerance', 0.1)
            )
            if result['passed']:
                passed_tests += 1
            total_tests += 1
        
        # Summarize results
        print(f"\nTest suite completed: {passed_tests}/{total_tests} tests passed.")
        
        all_passed = passed_tests == total_tests
        
        end_time = time.time()
        self.timing['run_all_tests'] = end_time - start_time
        
        # Return comprehensive results
        return {
            'results': self.test_results,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'all_passed': all_passed,
            'timing': self.timing
        }
    
    def plot_test_results(self, save_path=None):
        """
        Create visualizations of test results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plots
        """
        if not self.test_results:
            print("No test results to plot.")
            return
        
        # Plot annihilation physics results if available
        if 'annihilation_physics' in self.test_results:
            result = self.test_results['annihilation_physics']
            
            plt.figure(figsize=(10, 6))
            plt.plot(result['distances'], result['rates'], 'o-', lw=2)
            plt.xlabel('Distance (a.u.)')
            plt.ylabel('Annihilation Rate (a.u.)')
            plt.title(f"Annihilation Rate vs Distance\nCorrelation: {result['correlation']:.4f}")
            plt.grid(True)
            
            if save_path:
                plt.savefig(f"{save_path}/annihilation_physics.png")
            plt.show()
        
        # Plot relativistic effects if available
        if 'relativistic_effects' in self.test_results:
            result = self.test_results['relativistic_effects']
            
            plt.figure(figsize=(8, 6))
            labels = ['Non-Relativistic', 'Relativistic']
            energies = [result['non_relativistic_energy'], result['relativistic_energy']]
            
            plt.bar(labels, energies)
            plt.ylabel('Energy (Hartree)')
            plt.title(f"Relativistic Effects\nDifference: {result['energy_difference']:.8f} Hartree")
            plt.grid(True, axis='y')
            
            if save_path:
                plt.savefig(f"{save_path}/relativistic_effects.png")
            plt.show()
        
        # Summary of all test results
        if len(self.test_results) > 1:
            plt.figure(figsize=(12, 6))
            
            test_names = []
            passed = []
            
            for name, result in self.test_results.items():
                test_names.append(name)
                passed.append(1 if result.get('passed', False) else 0)
            
            plt.bar(test_names, passed, color=['green' if p else 'red' for p in passed])
            plt.ylabel('Passed (1) / Failed (0)')
            plt.title('Test Results Summary')
            plt.ylim(0, 1.2)
            
            for i, p in enumerate(passed):
                plt.text(i, p + 0.1, 'Pass' if p else 'Fail', ha='center')
            
            if save_path:
                plt.savefig(f"{save_path}/test_summary.png")
            plt.show()