import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class AntimatterValidator:
    """Validation tools for antimatter simulations."""
    
    def __init__(self):
        """Initialize the validator."""
        # Known theoretical values for validation
        self.theoretical_data = {
            # Positronium ground state energy in Hartree
            'positronium_energy': -0.25,
            
            # Positronium annihilation rates in a.u.
            'positronium_2gamma_rate': 0.5 * np.pi * (1/137.036)**2 * 137.036,
            
            # Positronium binding energy in eV
            'positronium_binding_eV': 6.8,
            
            # Positron affinities for atoms in eV
            'positron_affinities': {
                'H': -1.1,   # Negative indicates no bound state
                'Li': 0.01,
                'Na': 0.02,
                'K': 0.1
            },
            
            # Positron-Hydrogen elastic scattering cross-sections
            'positron_hydrogen_cross_section': {
                'energy_eV': [1.0, 3.0, 5.0, 10.0, 15.0],
                'cross_section_a0squared': [6.0, 4.5, 3.8, 2.5, 1.8]
            }
        }
    
    def compare_with_theory(self, 
                          result: Dict, 
                          reference_key: str,
                          tolerance: float = 0.1) -> Dict:
        """
        Compare results with theoretical predictions.
        
        Parameters:
        -----------
        result : Dict
            Calculated results
        reference_key : str
            Key in theoretical_data for comparison
        tolerance : float
            Relative tolerance for agreement
            
        Returns:
        --------
        Dict
            Comparison results
        """
        if reference_key not in self.theoretical_data:
            return {
                'success': False,
                'message': f"Reference data for '{reference_key}' not found"
            }
        
        reference_value = self.theoretical_data[reference_key]
        
        # Different handling based on reference type
        if isinstance(reference_value, dict):
            # For dictionaries, compare each key
            comparison = {}
            overall_success = True
            
            for key, ref_value in reference_value.items():
                if key in result:
                    calc_value = result[key]
                    rel_error = abs((calc_value - ref_value) / ref_value)
                    success = rel_error <= tolerance
                    
                    comparison[key] = {
                        'calculated': calc_value,
                        'reference': ref_value,
                        'relative_error': rel_error,
                        'success': success
                    }
                    
                    if not success:
                        overall_success = False
                else:
                    comparison[key] = {
                        'success': False,
                        'message': f"Key '{key}' not found in result"
                    }
                    overall_success = False
            
            return {
                'success': overall_success,
                'comparisons': comparison
            }
        
        else:
            # For scalar values, direct comparison
            if 'value' not in result:
                return {
                    'success': False,
                    'message': "Result must contain a 'value' key"
                }
            
            calc_value = result['value']
            rel_error = abs((calc_value - reference_value) / reference_value)
            success = rel_error <= tolerance
            
            return {
                'success': success,
                'calculated': calc_value,
                'reference': reference_value,
                'relative_error': rel_error
            }
    
    def validate_against_positronium(self, 
                                   energy: float,
                                   annihilation_rate: Optional[float] = None) -> Dict:
        """
        Validate using known positronium properties.
        
        Parameters:
        -----------
        energy : float
            Calculated energy in Hartree
        annihilation_rate : float, optional
            Calculated annihilation rate in atomic units
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Reference values for positronium
        ref_energy = self.theoretical_data['positronium_energy']
        ref_annihilation = self.theoretical_data['positronium_2gamma_rate']
        
        # Energy validation
        energy_error = abs((energy - ref_energy) / ref_energy)
        energy_valid = energy_error <= 0.1  # 10% tolerance
        
        results = {
            'energy': {
                'calculated': energy,
                'reference': ref_energy,
                'relative_error': energy_error,
                'valid': energy_valid
            }
        }
        
        # Annihilation validation (if provided)
        if annihilation_rate is not None:
            ann_error = abs((annihilation_rate - ref_annihilation) / ref_annihilation)
            ann_valid = ann_error <= 0.2  # 20% tolerance (annihilation is harder to calculate)
            
            results['annihilation'] = {
                'calculated': annihilation_rate,
                'reference': ref_annihilation,
                'relative_error': ann_error,
                'valid': ann_valid
            }
        
        # Overall assessment
        results['overall_valid'] = results['energy']['valid']
        if 'annihilation' in results:
            results['overall_valid'] = results['overall_valid'] and results['annihilation']['valid']
        
        return results
    
    def validate_potential_energy_curve(self, 
                                      distances: List[float],
                                      energies: List[float],
                                      system_type: str) -> Dict:
        """
        Validate potential energy curve shape.
        
        Parameters:
        -----------
        distances : List[float]
            Internuclear distances
        energies : List[float]
            Corresponding energies
        system_type : str
            Type of system ('positronium', 'positronic_molecule', etc)
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Convert to numpy arrays
        distances = np.array(distances)
        energies = np.array(energies)
        
        # Ensure arrays are sorted by distance
        sort_idx = np.argsort(distances)
        distances = distances[sort_idx]
        energies = energies[sort_idx]
        
        # Validate curve properties based on system type
        results = {
            'has_minimum': False,
            'valid_dissociation': False,
            'valid_short_range': False,
            'overall_valid': False
        }
        
        # Check if curve has a minimum
        min_idx = np.argmin(energies)
        results['has_minimum'] = min_idx > 0 and min_idx < len(energies) - 1
        results['minimum_distance'] = distances[min_idx] if results['has_minimum'] else None
        results['minimum_energy'] = energies[min_idx] if results['has_minimum'] else None
        
        # Check dissociation behavior
        if len(distances) >= 3:
            # Expect energy to increase as atoms move far apart for bound systems
            long_range_trend = (energies[-1] - energies[-3]) / (distances[-1] - distances[-3])
            results['valid_dissociation'] = long_range_trend > 0
        
        # Check short-range behavior
        if len(distances) >= 3 and distances[0] < 0.5:
            # Expect repulsive behavior at very short distances
            short_range_trend = (energies[2] - energies[0]) / (distances[2] - distances[0])
            results['valid_short_range'] = short_range_trend < 0
        
        # Overall assessment depends on system type
        if system_type == 'positronium':
            # Positronium has a simple 1/r potential
            # Energy should monotonically increase with distance
            results['expected_shape'] = 'monotonic'
            results['shape_valid'] = not results['has_minimum']
            results['overall_valid'] = results['shape_valid']
            
        elif system_type in ['positronic_molecule', 'positronic_atom']:
            # Molecules and positronic atoms should have a minimum
            results['expected_shape'] = 'has_minimum'
            results['shape_valid'] = results['has_minimum']
            results['overall_valid'] = (results['shape_valid'] and 
                                      results['valid_dissociation'] and 
                                      results['valid_short_range'])
        
        return results
    
    def plot_validation(self, 
                      result: Dict, 
                      reference: Dict, 
                      title: str = 'Validation Results'):
        """
        Plot validation results.
        
        Parameters:
        -----------
        result : Dict
            Calculated results
        reference : Dict
            Reference data
        title : str
            Plot title
        """
        # Determine what kind of data to plot
        if 'distances' in result and 'energies' in result:
            # Plot potential energy curve
            plt.figure(figsize=(10, 6))
            
            # Plot calculated data
            plt.plot(result['distances'], result['energies'], 
                    'o-', label='Calculated', color='blue')
            
            # Plot reference data if available
            if 'distances' in reference and 'energies' in reference:
                plt.plot(reference['distances'], reference['energies'], 
                        '--', label='Reference', color='red')
            
            plt.xlabel('Distance (a.u.)')
            plt.ylabel('Energy (a.u.)')
            plt.title(title)
            plt.legend()
            plt.grid(True)
            
        elif 'energies' in result and isinstance(result['energies'], list):
            # Plot energy levels
            plt.figure(figsize=(8, 6))
            
            # Plot calculated energy levels
            for i, energy in enumerate(result['energies']):
                plt.axhline(y=energy, linestyle='-', color='blue', alpha=0.7)
                plt.text(0.1, energy, f'E{i} = {energy:.4f}', color='blue')
            
            # Plot reference levels if available
            if 'energies' in reference and isinstance(reference['energies'], list):
                for i, energy in enumerate(reference['energies']):
                    plt.axhline(y=energy, linestyle='--', color='red', alpha=0.7)
                    plt.text(0.8, energy, f'Ref{i} = {energy:.4f}', color='red')
            
            plt.ylabel('Energy (a.u.)')
            plt.title(title)
            plt.grid(True)
            
        else:
            # Generic bar chart comparison
            common_keys = []
            for key in result:
                if key in reference and isinstance(result[key], (int, float)):
                    common_keys.append(key)
            
            if common_keys:
                plt.figure(figsize=(10, 6))
                
                x = np.arange(len(common_keys))
                width = 0.35
                
                # Calculate values
                calc_vals = [result[key] for key in common_keys]
                ref_vals = [reference[key] for key in common_keys]
                
                # Create bars
                plt.bar(x - width/2, calc_vals, width, label='Calculated', color='blue')
                plt.bar(x + width/2, ref_vals, width, label='Reference', color='red')
                
                plt.xlabel('Properties')
                plt.xticks(x, common_keys)
                plt.title(title)
                plt.legend()
                plt.grid(True)
        
        plt.savefig('validation_plot.png')
        plt.show()

