# examples/positronium_quantum.py

try:
    # Try importing from the package first (recommended)
    from antinature.qiskit_integration import PositroniumVQESolver
except ImportError:
    # Fallback to direct import
    from antinature.qiskit_integration.solver import PositroniumVQESolver

def main():
    """Run positronium quantum calculation example."""
    print("=== Positronium Quantum Calculation ===")
    
    # Create solver
    solver = PositroniumVQESolver(optimizer_name='COBYLA', shots=1024)
    
    # Compare classical and quantum solutions
    results = solver.solve_positronium(
        mapper_type='jordan_wigner',
        reps=2,
        use_classical=True
    )
    
    # Print results
    print("\nResults:")
    print(f"Theoretical energy:     {results['theoretical_energy']:.6f} Hartree")
    print(f"Classical exact energy: {results['classical_energy']:.6f} Hartree")
    print(f"VQE energy:            {results['vqe_energy']:.6f} Hartree")
    print(f"VQE error:             {results['vqe_error']:.6f} Hartree")
    print(f"VQE iterations:        {results['iterations']}")
    
    return results

if __name__ == "__main__":
    main()