# test_antiverse_applications.py

import numpy as np
import matplotlib.pyplot as plt
import tests.workflow_positronium as workflow_positronium
import tests.anti_hydrogen_workflow as anti_hydrogen_workflow
from antiverse.specialized.visualization import visualize_annihilation_density
from antiverse.specialized.visualization import plot_wavefunction

def test_positronium():
    """Test positronium calculation and validate against theory."""
    print("="*50)
    print("Testing positronium calculation")
    print("="*50)
    
    # Theoretical values
    theory = {
        'energy': -0.25,  # Hartree
        'lifetime_ns': 0.125,  # ns
    }
    
    # Run workflow
    results = workflow_positronium.positronium_workflow(basis_quality='standard', include_relativistic=True)
    
    # Validate results
    print("\nValidation against theoretical values:")
    print(f"Energy: {results['energy']:.6f} Hartree (theoretical: {theory['energy']:.6f})")
    print(f"Energy error: {abs(results['energy'] - theory['energy']):.6f} Hartree")
    
    lifetime = results.get('lifetime', {}).get('lifetime_ns', float('inf'))
    if lifetime == float('inf'):
        print("Lifetime: infinite (cannot compare to theoretical value)")
    else:
        print(f"Lifetime: {lifetime:.6f} ns (theoretical: {theory['lifetime_ns']:.6f})")
        print(f"Lifetime error: {abs(lifetime - theory['lifetime_ns']):.6f} ns")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot annihilation density
    if 'density_data' in results and results['density_data'] is not None:
        fig_density = visualize_annihilation_density(results['density_data'], slice_dim='3d')
        plt.savefig('positronium_density.png')
        print("Saved annihilation density visualization to 'positronium_density.png'")
        plt.close(fig_density)
    else:
        print("No density data available for visualization")
    
    # Plot wavefunctions
    if 'wavefunction' in results and results['wavefunction'] is not None:
        fig_wf_e = plot_wavefunction(results['wavefunction'], 'electron', [0])
        if fig_wf_e is not None:
            plt.savefig('positronium_electron_wf.png')
            print("Saved electron wavefunction to 'positronium_electron_wf.png'")
            plt.close(fig_wf_e)
        
        fig_wf_p = plot_wavefunction(results['wavefunction'], 'positron', [0])
        if fig_wf_p is not None:
            plt.savefig('positronium_positron_wf.png')
            print("Saved positron wavefunction to 'positronium_positron_wf.png'")
            plt.close(fig_wf_p)
    else:
        print("No wavefunction data available for visualization")
    
    return results

def test_anti_hydrogen():
    """Test anti-hydrogen calculation and validate against theory."""
    print("="*50)
    print("Testing anti-hydrogen calculation")
    print("="*50)
    
    # Theoretical values
    theory = {
        'energy': -0.5,  # Hartree (exact for hydrogen)
    }
    
    # Run workflow
    results = anti_hydrogen_workflow.anti_hydrogen_workflow(basis_quality='standard', include_relativistic=True)
    
    # Validate results
    print("\nValidation against theoretical values:")
    print(f"Energy: {results['energy']:.6f} Hartree (theoretical: {theory['energy']:.6f})")
    print(f"Energy error: {abs(results['energy'] - theory['energy']):.6f} Hartree")
    
    # Visualize results
    print("\nGenerating visualizations...")
    
    # Plot wavefunctions
    if 'wavefunction' in results and results['wavefunction'] is not None:
        fig_wf_p = plot_wavefunction(results['wavefunction'], 'positron', [0, 1])
        if fig_wf_p is not None:
            plt.savefig('anti_hydrogen_positron_wf.png')
            print("Saved positron wavefunction to 'anti_hydrogen_positron_wf.png'")
            plt.close(fig_wf_p)
    else:
        print("No wavefunction data available for visualization")
    
    return results

def run_all_tests():
    """Run all tests and validations."""
    ps_results = test_positronium()
    print("\n")
    ah_results = test_anti_hydrogen()
    
    return {
        'positronium': ps_results,
        'anti_hydrogen': ah_results
    }

if __name__ == "__main__":
    run_all_tests()