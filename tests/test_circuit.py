# test_circuits.py

import numpy as np
import sys
import os

# Add parent directory to path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Qiskit not installed. Skipping tests.")
    sys.exit(0)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from antinature.qiskit_integration.circuits import AntinatureCircuits, PositroniumCircuit

def test_circuits():
    """Test the enhanced AntinatureCircuits class."""
    print("Testing AntinatureCircuits...")
    
    # Create circuits generator
    circuits = AntinatureCircuits(
        n_electron_orbitals=2,
        n_positron_orbitals=2
    )
    
    # Test various ansatz types
    print("\nTesting different ansatz configurations:")
    for entanglement in ['linear', 'full', 'circular', 'sca']:
        for rotation_blocks in ['x', 'xy', 'xyz']:
            for initial_state in ['zero', 'uniform']:
                print(f"  Creating ansatz with: entanglement={entanglement}, rotations={rotation_blocks}, init={initial_state}")
                
                try:
                    circuit = circuits.create_antinature_ansatz(
                        reps=1,
                        entanglement=entanglement,
                        rotation_blocks=rotation_blocks,
                        initial_state=initial_state,
                        name=f"test_{entanglement}_{rotation_blocks}_{initial_state}"
                    )
                    
                    # Print circuit details
                    print(f"    Circuit depth: {circuit.depth()}")
                    print(f"    Number of parameters: {circuit.num_parameters}")
                    print(f"    Gate counts: {circuit.count_ops()}")
                except Exception as e:
                    print(f"    Error creating ansatz: {str(e)}")
    
    # Test positronium circuits
    print("\nTesting positronium-specific circuits:")
    try:
        circuit = circuits.create_positronium_circuit()
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating positronium circuit: {str(e)}")
    
    # Test anti-hydrogen circuits
    print("\nTesting anti-hydrogen circuits:")
    try:
        circuit = circuits.create_anti_hydrogen_circuit()
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating anti-hydrogen circuit: {str(e)}")
    
    # Test SU2 ansatz
    print("\nTesting EfficientSU2 ansatz:")
    try:
        circuit = circuits.create_efficient_su2_ansatz(reps=2)
        print(f"  Circuit depth: {circuit.depth()}")
        print(f"  Number of parameters: {circuit.num_parameters}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating SU2 ansatz: {str(e)}")
    
    # Try exporting a circuit
    print("\nTesting circuit export:")
    try:
        circuit = circuits.create_custom_ansatz(reps=1)
        if hasattr(circuits, 'export_circuit'):
            qasm = circuits.export_circuit(circuit, format='qasm')
            print(f"  QASM export size: {len(qasm)} characters")
        else:
            print("  Export function not available")
    except Exception as e:
        print(f"  Error exporting circuit: {str(e)}")
    
    # Visualization (works in interactive environment)
    try:
        print("\nGenerating circuit visualization (saving to circuit.png)...")
        from qiskit.visualization import circuit_drawer
        if 'circuit' in locals():
            fig = circuit_drawer(circuit, output='mpl')
            fig.savefig('circuit.png')
            print("  Circuit visualization saved to circuit.png")
        else:
            print("  No circuit available for visualization")
    except Exception as e:
        print(f"  Could not create visualization: {str(e)}")

def test_positronium_circuit():
    """Test the PositroniumCircuit class."""
    print("\nTesting PositroniumCircuit...")
    
    # Create positronium circuit
    ps_circuit = PositroniumCircuit(
        n_electron_orbitals=1,
        n_positron_orbitals=1
    )
    
    # Test all circuit types
    print("\nGenerating positronium circuits:")
    
    # Ground state
    try:
        circuit = ps_circuit.create_positronium_ground_state()
        print(f"  Ground state circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating ground state circuit: {str(e)}")
    
    # Annihilation detector
    try:
        circuit = ps_circuit.create_annihilation_detector()
        print(f"  Annihilation detector circuit depth: {circuit.depth()}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating annihilation detector: {str(e)}")
    
    # VQE ansatz
    try:
        circuit = ps_circuit.create_vqe_ansatz(reps=2)
        print(f"  VQE ansatz circuit depth: {circuit.depth()}")
        print(f"  Number of parameters: {circuit.num_parameters}")
        print(f"  Gate counts: {circuit.count_ops()}")
    except Exception as e:
        print(f"  Error creating VQE ansatz: {str(e)}")
    
    # Para-ortho detector
    try:
        if hasattr(ps_circuit, 'create_para_ortho_detector'):
            circuit = ps_circuit.create_para_ortho_detector()
            print(f"  Para-ortho detector circuit depth: {circuit.depth()}")
            print(f"  Gate counts: {circuit.count_ops()}")
        else:
            print("  Para-ortho detector method not available")
    except Exception as e:
        print(f"  Error creating para-ortho detector: {str(e)}")

if __name__ == "__main__":
    if HAS_QISKIT:
        test_circuits()
        test_positronium_circuit()
    else:
        print("Qiskit not available. Cannot run tests.")