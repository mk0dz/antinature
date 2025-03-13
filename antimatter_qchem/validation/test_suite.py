import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from .validator import AntimatterValidator

class TestSuite:
    """Comprehensive tests for antimatter quantum chemistry."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the test suite.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed output
        """
        self.verbose = verbose
        self.validator = AntimatterValidator()
        self.test_results = {}
    
    def test_energy_conservation(self, 
                               system_factory, 
                               solver,
                               bond_distances: List[float]):
        """
        Test energy conservation in various processes.
        
        Parameters:
        -----------
        system_factory : function
            Function that creates a system for a given bond distance
        solver : function
            Function that solves the system
        bond_distances : List[float]
            Bond distances to test
            
        Returns:
        --------
        Dict
            Test results
        """
        energies = []
        
        if self.verbose:
            print("Testing energy conservation across bond distances:")
        
        # Calculate energies at different geometries
        for distance in bond_distances:
            # Create system
            system = system_factory(distance)
            
            # Solve system
            result = solver(system)
            
            # Extract energy
            energy = result['energy']
            energies.append(energy)
            
            if self.verbose:
                print(f"  Distance: {distance:.2f} a.u., Energy: {energy:.6f} a.u.")
        
        # Check for energy conservation properties
        # For a conservative system, energy should vary smoothly
        energies = np.array(energies)
        distances = np.array(bond_distances)
        
        # Calculate energy differences
        energy_diffs = np.diff(energies)
        distance_diffs = np.diff(distances)
        energy_derivatives = energy_diffs / distance_diffs
        
        # Check for smoothness - no sharp discontinuities
        max_derivative = np.max(np.abs(energy_derivatives))
        smooth_energy = max_derivative < 1.0  # This threshold can be adjusted
        
        # Find equilibrium distance (minimum energy)
        min_idx = np.argmin(energies)
        equilibrium_distance = distances[min_idx]
        min_energy = energies[min_idx]
        
        # Check if system dissociates properly
        if min_idx < len(distances) - 1:
            dissociation_trend = (energies[-1] - min_energy) > 0
        else:
            dissociation_trend = False
            
        # Prepare result
        result = {
            'distances': distances.tolist(),
            'energies': energies.tolist(),
            'smooth_energy': smooth_energy,
            'max_derivative': max_derivative,
            'equilibrium_distance': equilibrium_distance,
            'min_energy': min_energy,
            'dissociation_trend': dissociation_trend,
            'passed': smooth_energy and dissociation_trend
        }
        
        self.test_results['energy_conservation'] = result
        return result
    
    def test_annihilation_physics(self, 
                                annihilation_calculator,
                                varying_parameter,
                                parameter_range: List[float]):
        """
        Test that annihilation physics follows expected behavior.
        
        Parameters:
        -----------
        annihilation_calculator : function
            Function that calculates annihilation rate for a parameter value
        varying_parameter : str
            Description of the parameter being varied
        parameter_range : List[float]
            Values of the parameter to test
            
        Returns:
        --------
        Dict
            Test results
        """
        parameters = []
        rates = []
        
        if self.verbose:
            print(f"Testing annihilation physics by varying {varying_parameter}:")
        
        # Calculate annihilation rates for different parameter values
        for param_value in parameter_range:
            # Calculate annihilation rate
            rate = annihilation_calculator(param_value)
            
            parameters.append(param_value)
            rates.append(rate)
            
            if self.verbose:
                print(f"  {varying_parameter}: {param_value:.2f}, Rate: {rate:.6e}")
        
        # Analyze annihilation behavior
        parameters = np.array(parameters)
        rates = np.array(rates)
        
        # Check for expected trends based on the parameter being varied
        if varying_parameter in ['distance', 'bond_length']:
            # Annihilation rate should decrease with increasing distance
            correlation = np.corrcoef(parameters, rates)[0, 1]
            expected_trend = correlation < 0
            trend_strength = abs(correlation)
        
        elif varying_parameter in ['density', 'overlap']:
            # Annihilation rate should increase with density/overlap
            correlation = np.corrcoef(parameters, rates)[0, 1]
            expected_trend = correlation > 0
            trend_strength = abs(correlation)
        
        else:
            # Default: just check for a strong correlation either way
            correlation = np.corrcoef(parameters, rates)[0, 1]
            expected_trend = abs(correlation) > 0.7
            trend_strength = abs(correlation)
        
        # Prepare result
        result = {
            'parameter': varying_parameter,
            'parameter_values': parameters.tolist(),
            'annihilation_rates': rates.tolist(),
            'correlation': correlation,
            'expected_trend': expected_trend,
            'trend_strength': trend_strength,
            'passed': expected_trend and (trend_strength > 0.7)
        }
        
        self.test_results['annihilation_physics'] = result
        return result
    
    def test_positronium_properties(self, positronium_calculator):
        """
        Test positronium properties against known values.
        
        Parameters:
        -----------
        positronium_calculator : function
            Function that calculates positronium properties
            
        Returns:
        --------
        Dict
            Test results
        """
        if self.verbose:
            print("Testing positronium properties:")
        
        # Calculate positronium properties
        properties = positronium_calculator()
        
        # Extract properties
        energy = properties.get('energy', 0.0)
        annihilation_rate = properties.get('annihilation_rate', 0.0)
        
        # Validate against known values
        validation = self.validator.validate_against_positronium(
            energy, annihilation_rate
        )
        
        if self.verbose:
            print(f"  Energy: Calculated={energy:.6f}, Reference={validation['energy']['reference']:.6f}")
            if 'annihilation' in validation:
                print(f"  Annihilation Rate: Calculated={annihilation_rate:.6e}, "
                     f"Reference={validation['annihilation']['reference']:.6e}")
            print(f"  Overall valid: {validation['overall_valid']}")
        
        self.test_results['positronium_properties'] = validation
        return validation
    
    def test_relativistic_effects(self, 
                                with_relativistic_calculator,
                                without_relativistic_calculator):
        """
        Test the impact of relativistic effects.
        
        Parameters:
        -----------
        with_relativistic_calculator : function
            Function that calculates properties with relativistic effects
        without_relativistic_calculator : function
            Function that calculates properties without relativistic effects
            
        Returns:
        --------
        Dict
            Test results
        """
        if self.verbose:
            print("Testing relativistic effects:")
        
        # Calculate properties with and without relativistic effects
        with_rel = with_relativistic_calculator()
        without_rel = without_relativistic_calculator()
        
        # Extract energies
        energy_with = with_rel.get('energy', 0.0)
        energy_without = without_rel.get('energy', 0.0)
        
        # Calculate difference
        energy_difference = energy_with - energy_without
        relative_difference = abs(energy_difference / energy_without)
        
        # Check if the difference is significant
        significant_difference = relative_difference > 0.01  # >1% difference
        
        if self.verbose:
            print(f"  Energy with relativistic effects: {energy_with:.6f}")
            print(f"  Energy without relativistic effects: {energy_without:.6f}")
            print(f"  Difference: {energy_difference:.6f} ({relative_difference*100:.2f}%)")
            print(f"  Significant difference: {significant_difference}")
        
        # Prepare result
        result = {
            'energy_with_relativistic': energy_with,
            'energy_without_relativistic': energy_without,
            'energy_difference': energy_difference,
            'relative_difference': relative_difference,
            'significant_difference': significant_difference,
            'passed': significant_difference
        }
        
        self.test_results['relativistic_effects'] = result
        return result
    
    def run_all_tests(self, 
                   system_factory=None, 
                   solver=None,
                   annihilation_calculator=None,
                   positronium_calculator=None,
                   relativistic_calculators=None):
        """
        Run all tests in the suite.
        
        Parameters:
        -----------
        system_factory : function, optional
            Function to create systems
        solver : function, optional
            Function to solve systems
        annihilation_calculator : function, optional
            Function to calculate annihilation rates
        positronium_calculator : function, optional
            Function to calculate positronium properties
        relativistic_calculators : Tuple[function, function], optional
            Functions to calculate with and without relativistic effects
            
        Returns:
        --------
        Dict
            All test results
        """
        print("Running antimatter quantum chemistry test suite...")
        
        # Run tests that have all required inputs
        if system_factory is not None and solver is not None:
            print("\n1. Testing energy conservation...")
            bond_distances = np.linspace(0.5, 5.0, 10)
            self.test_energy_conservation(system_factory, solver, bond_distances)
        else:
            print("\n1. Skipping energy conservation test (missing inputs).")
        
        if annihilation_calculator is not None:
            print("\n2. Testing annihilation physics...")
            parameter_range = np.linspace(0.5, 3.0, 8)
            self.test_annihilation_physics(annihilation_calculator, 'distance', parameter_range)
        else:
            print("\n2. Skipping annihilation physics test (missing inputs).")
        
        if positronium_calculator is not None:
            print("\n3. Testing positronium properties...")
            self.test_positronium_properties(positronium_calculator)
        else:
            print("\n3. Skipping positronium properties test (missing inputs).")
        
        if relativistic_calculators is not None:
            print("\n4. Testing relativistic effects...")
            with_rel, without_rel = relativistic_calculators
            self.test_relativistic_effects(with_rel, without_rel)
        else:
            print("\n4. Skipping relativistic effects test (missing inputs).")
        
        # Count passed tests
        passed = sum(1 for result in self.test_results.values() 
                    if result.get('passed', False) or result.get('overall_valid', False))
        total = len(self.test_results)
        
        print(f"\nTest suite completed: {passed}/{total} tests passed.")
        
        return self.test_results