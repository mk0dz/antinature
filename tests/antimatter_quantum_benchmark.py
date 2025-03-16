# examples/antimatter_quantum_benchmark.py

import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Use the package-level import instead of reaching into submodules
try:
    # Try importing from the package level first (recommended)
    from anttimatter.qiskit_integration import AntimatterQuantumSolver
except ImportError:
    # Fallback to direct import
    from anttimatter.qiskit_integration.antimatter_solver import AntimatterQuantumSolver

def main():
    """Run antimatter quantum simulations and benchmark the results."""
    print("=== Antimatter Quantum Systems Benchmark ===\n")
    
    # Systems to benchmark
    systems = [
        'positronium',
        'anti_hydrogen',
        'positronium_molecule'
    ]
    
    # Create output directory for plots
    os.makedirs('results', exist_ok=True)
    
    # Collect timing and results
    timings = {}
    energy_results = {}
    
    # Create solver with optimized settings
    solver = AntimatterQuantumSolver(
        use_exact_solver=False,  # Use VQE by default
        optimizer_name='COBYLA',
        max_iterations=300,
        shots=1024
    )
    
    # Run benchmarks for each system
    for system in systems:
        print(f"Solving {system}...")
        
        # Time the execution
        start_time = time.time()
        
        # Solve using specialized ansatz
        try:
            if system == 'positronium':
                results = solver.solve_positronium(ansatz_type='specialized')
            elif system == 'anti_hydrogen':
                results = solver.solve_anti_hydrogen(ansatz_type='specialized')
            elif system == 'positronium_molecule':
                results = solver.solve_positronium_molecule(ansatz_type='specialized')
            
            energy = results['energy']
            was_corrected = results.get('was_corrected', False)
            raw_energy = results.get('raw_energy', energy)
            
            # Also try with hardware-efficient ansatz for comparison
            if system == 'positronium':
                he_results = solver.solve_positronium(ansatz_type='hardware_efficient')
            elif system == 'anti_hydrogen':
                he_results = solver.solve_anti_hydrogen(ansatz_type='hardware_efficient')
            elif system == 'positronium_molecule':
                he_results = solver.solve_positronium_molecule(ansatz_type='hardware_efficient')
                
            he_energy = he_results['energy']
            
            # Try exact solver for reference
            solver.use_exact_solver = True
            if system == 'positronium':
                exact_results = solver.solve_positronium()
            elif system == 'anti_hydrogen':
                exact_results = solver.solve_anti_hydrogen()
            elif system == 'positronium_molecule':
                exact_results = solver.solve_positronium_molecule()
                
            exact_energy = exact_results['energy']
            solver.use_exact_solver = False  # Restore VQE mode
            
            # Store results
            energy_results[system] = {
                'specialized': energy,
                'raw_specialized': raw_energy,
                'hardware_efficient': he_energy,
                'exact': exact_energy,
                'theoretical': results['theoretical_value'],
                'was_corrected': was_corrected
            }
            
            # Create visualization
            solver.visualize_results(results, filename=f'results/{system}_specialized.png')
            solver.visualize_results(he_results, filename=f'results/{system}_hardware.png')
            
            # Compare methods
            comparison = {
                'specialized': results,
                'hardware_efficient': he_results,
                'exact': exact_results
            }
            solver.compare_visualization(comparison, filename=f'results/{system}_comparison.png')
            
        except Exception as e:
            print(f"Error solving {system}: {str(e)}")
            import traceback
            traceback.print_exc()
            energy_results[system] = {'error': str(e)}
        
        # Record timing
        end_time = time.time()
        execution_time = end_time - start_time
        timings[system] = execution_time
        
        print(f"Completed in {execution_time:.2f} seconds\n")
    
    # Print summary of results
    print("\n=== Summary of Results ===\n")
    for system, result in energy_results.items():
        print(f"{system.replace('_', ' ').title()}:")
        if 'error' in result:
            print(f"  Error: {result['error']}")
            continue
            
        theo = result['theoretical']
        spec = result['specialized']
        raw_spec = result.get('raw_specialized', spec)
        hw = result['hardware_efficient']
        exact = result['exact']
        
        print(f"  Theoretical value: {theo:.4f} Hartree")
        print(f"  Specialized ansatz: {spec:.4f} Hartree (error: {abs(spec-theo):.4f})")
        if result.get('was_corrected', False):
            print(f"    Raw value before correction: {raw_spec:.4f} Hartree")
        print(f"  Hardware-efficient ansatz: {hw:.4f} Hartree (error: {abs(hw-theo):.4f})")
        print(f"  Exact solver: {exact:.4f} Hartree (error: {abs(exact-theo):.4f})")
        print(f"  Execution time: {timings[system]:.2f} seconds\n")
    
    # Create timing comparison chart
    plt.figure(figsize=(10, 6))
    plt.bar(timings.keys(), timings.values())
    plt.title('Execution Time by System')
    plt.ylabel('Time (seconds)')
    plt.savefig('results/timing_comparison.png')
    
    print("Benchmark complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    main()