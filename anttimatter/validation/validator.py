import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import json
import os
import matplotlib.pyplot as plt

class AntimatterValidator:
    """
    Validation tools for antimatter simulations against theoretical data.
    """
    
    def __init__(self, reference_data_path=None):
        """
        Initialize the validator with reference data.
        
        Parameters:
        -----------
        reference_data_path : str, optional
            Path to reference data file
        """
        # Default reference data
        self.reference_data = {
            # Positronium
            'positronium': {
                'energy': -0.25,  # Hartree
                'annihilation_rate': 0.5 * np.pi * (1/137.036)**2 * 137.036,  # a.u.
                'lifetime_ns': 0.125,  # ns
                'binding_energy_eV': 6.8  # eV
            },
            
            # Positron affinities
            'positron_affinities_eV': {
                'H': -1.1,   # Negative means unbound
                'Li': 0.01,
                'Na': 0.02,
                'K': 0.1,
                'LiH': 0.9,
                'BeO': 2.4
            },
            
            # Annihilation rates (Zeff)
            'annihilation_zeff': {
                'H': 8.0,
                'He': 3.9,
                'Li': 25.8,
                'Be': 38.5,
                'H2': 14.6,
                'H2O': 326.0
            },
            
            # Relativistic corrections
            'relativistic_corrections': {
                'H': -3e-4,  # Hartree
                'He': -1.2e-3,
                'Li': -3e-3,
                'C': -1.8e-2
            }
        }
        
        # Load external reference data if provided
        if reference_data_path and os.path.exists(reference_data_path):
            try:
                with open(reference_data_path, 'r') as f:
                    external_data = json.load(f)
                
                # Update reference data with external values
                for key, value in external_data.items():
                    self.reference_data[key] = value
            except Exception as e:
                print(f"Error loading reference data: {e}")
    
    def validate_energy(self, 
                      system_name: str, 
                      calculated_energy: float, 
                      system_type: str = 'atom',
                      tolerance: float = 0.1) -> Dict:
        """
        Validate energy against reference data.
        
        Parameters:
        -----------
        system_name : str
            Name of the system (element symbol or molecule)
        calculated_energy : float
            Calculated energy in Hartree
        system_type : str
            Type of system ('atom', 'molecule', 'positronium')
        tolerance : float
            Relative tolerance for validation
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Find reference energy
        if system_type == 'positronium':
            reference = self.reference_data.get('positronium', {}).get('energy')
        elif system_name in self.reference_data.get('atoms', {}):
            reference = self.reference_data['atoms'][system_name].get('energy')
        elif system_name in self.reference_data.get('molecules', {}):
            reference = self.reference_data['molecules'][system_name].get('energy')
        else:
            reference = None
        
        if reference is None:
            return {
                'valid': False,
                'reason': 'No reference energy available',
                'calculated': calculated_energy
            }
        
        # Calculate relative error
        error = abs(calculated_energy - reference)
        relative_error = error / abs(reference)
        
        # Check if within tolerance
        valid = relative_error <= tolerance
        
        return {
            'valid': valid,
            'reference': reference,
            'calculated': calculated_energy,
            'error': error,
            'relative_error': relative_error,
            'tolerance': tolerance
        }
    
    def validate_annihilation_rate(self,
                                 system_name: str,
                                 calculated_rate: float,
                                 tolerance: float = 0.2) -> Dict:
        """
        Validate annihilation rate against reference data.
        
        Parameters:
        -----------
        system_name : str
            Name of the system
        calculated_rate : float
            Calculated annihilation rate in atomic units
        tolerance : float
            Relative tolerance for validation
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Find reference rate
        if system_name == 'positronium':
            reference = self.reference_data.get('positronium', {}).get('annihilation_rate')
        elif system_name in self.reference_data.get('annihilation_zeff', {}):
            # Convert Zeff to rate
            zeff = self.reference_data['annihilation_zeff'][system_name]
            reference = zeff * np.pi * (1/137.036)**2 * 137.036
        else:
            reference = None
        
        if reference is None:
            return {
                'valid': False,
                'reason': 'No reference annihilation rate available',
                'calculated': calculated_rate
            }
        
        # Calculate relative error
        error = abs(calculated_rate - reference)
        relative_error = error / reference
        
        # Check if within tolerance
        valid = relative_error <= tolerance
        
        return {
            'valid': valid,
            'reference': reference,
            'calculated': calculated_rate,
            'error': error,
            'relative_error': relative_error,
            'tolerance': tolerance
        }
    
    def validate_positron_affinity(self,
                                 system_name: str,
                                 calculated_affinity_eV: float,
                                 tolerance: float = 0.5) -> Dict:
        """
        Validate positron affinity against reference data.
        
        Parameters:
        -----------
        system_name : str
            Name of the system
        calculated_affinity_eV : float
            Calculated positron affinity in eV
        tolerance : float
            Absolute tolerance in eV
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Find reference affinity
        reference = self.reference_data.get('positron_affinities_eV', {}).get(system_name)
        
        if reference is None:
            return {
                'valid': False,
                'reason': 'No reference positron affinity available',
                'calculated': calculated_affinity_eV
            }
        
        # Calculate absolute error
        error = abs(calculated_affinity_eV - reference)
        
        # Check if within tolerance
        valid = error <= tolerance
        
        # Special handling for unbound states (negative affinity)
        if reference < 0 and calculated_affinity_eV < 0:
            valid = True  # Both predict unbound state
        
        return {
            'valid': valid,
            'reference': reference,
            'calculated': calculated_affinity_eV,
            'error': error,
            'bound_state': reference > 0,
            'calculated_bound': calculated_affinity_eV > 0,
            'tolerance': tolerance
        }
    
    def validate_relativistic_correction(self,
                                      system_name: str,
                                      calculated_correction: float,
                                      tolerance: float = 0.5) -> Dict:
        """
        Validate relativistic correction against reference data.
        
        Parameters:
        -----------
        system_name : str
            Name of the system
        calculated_correction : float
            Calculated relativistic correction in Hartree
        tolerance : float
            Relative tolerance for validation
            
        Returns:
        --------
        Dict
            Validation results
        """
        # Find reference correction
        reference = self.reference_data.get('relativistic_corrections', {}).get(system_name)
        
        if reference is None:
            return {
                'valid': False,
                'reason': 'No reference relativistic correction available',
                'calculated': calculated_correction
            }
        
        # Calculate relative error
        error = abs(calculated_correction - reference)
        relative_error = error / abs(reference) if reference != 0 else float('inf')
        
        # Check if within tolerance
        valid = relative_error <= tolerance
        
        return {
            'valid': valid,
            'reference': reference,
            'calculated': calculated_correction,
            'error': error,
            'relative_error': relative_error,
            'tolerance': tolerance
        }
    
    def plot_comparison(self, validation_result: Dict, title: str = 'Validation Result', save_path: str = None):
        """
        Create a visual comparison of calculated vs. reference values.
        
        Parameters:
        -----------
        validation_result : Dict
            Results from validation methods
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        if 'reference' not in validation_result or 'calculated' not in validation_result:
            print("Cannot plot: validation result missing required keys.")
            return
        
        plt.figure(figsize=(8, 6))
        
        labels = ['Reference', 'Calculated']
        values = [validation_result['reference'], validation_result['calculated']]
        
        bars = plt.bar(labels, values, color=['blue', 'orange'])
        
        # Add validation status
        if validation_result.get('valid', False):
            plt.axhline(y=validation_result['reference'], color='green', linestyle='--', alpha=0.7)
            status = "✓ Valid"
            color = 'green'
        else:
            plt.axhline(y=validation_result['reference'], color='red', linestyle='--', alpha=0.7)
            status = "✗ Invalid"
            color = 'red'
        
        # Add error information
        if 'error' in validation_result:
            error_text = f"Error: {validation_result['error']:.6f}"
            if 'relative_error' in validation_result:
                error_text += f" ({validation_result['relative_error']*100:.2f}%)"
            plt.title(f"{title}\n{error_text}\n{status}", color=color)
        else:
            plt.title(f"{title}\n{status}", color=color)
        
        plt.ylabel('Value')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add values on top of bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                   f"{value:.6f}", ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()