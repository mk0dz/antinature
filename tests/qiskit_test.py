import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antimatter_qchem import * 
try:
    from antimatter_qchem.qiskit_integration import (
        QiskitNatureAdapter,
        AntimatterCircuits
    )
    has_qiskit = True
except ImportError:
    print("Warning: Qiskit modules not available. Some tests will be skipped.")
    has_qiskit = False

def test_hamiltonian_to_qiskit_operator():
    """Test conversion of antimatter Hamiltonian to Qiskit operators."""
    print("Testing Hamiltonian to Qiskit operator conversion...")
    
    if not has_qiskit:
        print("  Skipping test: Qiskit not available")
        return None, None
    
    # Create a simple system: hydrogen atom
    nuclei = [('H', 1.0, np.array([0.0, 0.0, 0.0]))]
    n_electrons = 1
    n_positrons = 0
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=False,
        include_relativistic=False
    )
    
    # Create a basis set
    mixed_basis = MixedMatterBasis()
    molecule = [('H', np.array([0.0, 0.0, 0.0]))]
    mixed_basis.create_for_molecule(molecule, 'minimal', 'minimal')
    
    print(f"\nBasis set information:")
    print(f"  Electron basis functions: {mixed_basis.n_electron_basis}")
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    print("\nBuilding Hamiltonian...")
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Create Qiskit adapter
    adapter = QiskitNatureAdapter(mapper_type='jordan_wigner')
    
    # Convert to Qiskit operator
    print("\nConverting to Qiskit operator...")
    qiskit_operators = adapter.convert_to_qiskit_hamiltonian(
        hamiltonian_dict,
        n_electrons,
        n_positrons
    )
    
    # Print operator information
    print("\nQiskit operator information:")
    for key, value in qiskit_operators.items():
        if hasattr(value, 'num_qubits'):
            print(f"  {key}: {value.num_qubits} qubits")
        else:
            print(f"  {key}")
    
    if 'electron_operator' in qiskit_operators:
        op = qiskit_operators['electron_operator']
        print(f"\nElectron operator details:")
        print(f"  Number of qubits: {op.num_qubits}")
        print(f"  Number of Pauli terms: {len(op)}")
        
        # Print a few terms for inspection
        print(f"\nSample of Pauli terms:")
        count = 0
        for pauli_op, coeff in list(op)[:5]:  # Print first 5 terms
            print(f"  {coeff.real:.6f} * {pauli_op}")
            count += 1
        
        if len(op) > 5:
            print(f"  ... and {len(op) - 5} more terms")
    
    return qiskit_operators, adapter

def test_vqe_adaptation():
    """Test VQE adaptation for antimatter systems."""
    print("\nTesting VQE adaptation for antimatter systems...")
    
    if not has_qiskit:
        print("  Skipping test: Qiskit not available")
        return None
    
    # Get Qiskit operators from previous test
    qiskit_operators, adapter = test_hamiltonian_to_qiskit_operator()
    
    if qiskit_operators is None or adapter is None:
        return None
    
    # Adapt VQE for antimatter
    print("\nAdapting VQE for antimatter...")
    vqe_adapters = adapter.adapt_vqe_for_antimatter(
        qiskit_operators,
        optimizer_name='COBYLA',
        ansatz_type='efficient_su2',
        ansatz_reps=1
    )
    
    # Print VQE information
    print("\nVQE adaptation information:")
    for key, value in vqe_adapters.items():
        print(f"  {key}")
    
    if 'electron_vqe' in vqe_adapters:
        vqe = vqe_adapters['electron_vqe']
        print(f"\nElectron VQE details:")
        print(f"  Ansatz: {vqe.ansatz}")
        print(f"  Optimizer: {vqe.optimizer}")
    
    # Run classical simulation as a test
    print("\nRunning classical eigenvalue calculation...")
    result = adapter.run_ground_state_calculation(
        qiskit_operators,
        vqe_adapters,
        use_classical_solver=True
    )
    
    # Print results
    print("\nClassical calculation results:")
    for key, value in result.items():
        if key != 'electron_result' and key != 'positron_result':
            print(f"  {key}: {value}")
    
    # Compare with expected hydrogen atom energy (-0.5 Hartree)
    if 'electron_energy' in result:
        energy = result['electron_energy']
        exact = -0.5
        error = energy - exact
        print(f"\nComparison with exact energy:")
        print(f"  Calculated: {energy:.8f} Hartree")
        print(f"  Exact: {exact:.8f} Hartree")
        print(f"  Error: {error:.8f} Hartree ({error/exact*100:.2f}%)")
    
    return result

def test_antimatter_circuits():
    """Test specialized quantum circuits for antimatter."""
    print("\nTesting quantum circuits for antimatter...")
    
    if not has_qiskit:
        print("  Skipping test: Qiskit not available")
        return None
    
    # Create circuit generator
    n_electron_qubits = 2
    n_positron_qubits = 2
    circuits = AntimatterCircuits(n_electron_qubits, n_positron_qubits)
    
    print(f"Created circuit generator with:")
    print(f"  Electron qubits: {circuits.n_electron_qubits}")
    print(f"  Positron qubits: {circuits.n_positron_qubits}")
    print(f"  Total qubits: {circuits.n_total_qubits}")
    
    # Test Hartree-Fock initial state circuit
    print("\nGenerating Hartree-Fock initial state circuit...")
    hf_circuit = circuits.hartree_fock_initial_state(1, 1)
    
    print(f"HF circuit details:")
    print(f"  Number of qubits: {sum(reg.size for reg in hf_circuit.qregs)}")
    print(f"  Number of classical bits: {sum(reg.size for reg in hf_circuit.cregs)}")
    print(f"  Circuit depth: {hf_circuit.depth()}")
    print(f"  Number of operations: {len(hf_circuit.data)}")
    
    # Try to draw the circuit if possible
    try:
        print("\nCircuit diagram (ASCII):")
        print(hf_circuit.draw(output='text'))
    except:
        print("  Could not draw circuit diagram")
    
    # Test annihilation circuit
    print("\nGenerating annihilation detection circuit...")
    try:
        ann_circuit = circuits.annihilation_circuit()
        
        print(f"Annihilation circuit details:")
        print(f"  Number of qubits: {sum(reg.size for reg in ann_circuit.qregs)}")
        print(f"  Number of classical bits: {sum(reg.size for reg in ann_circuit.cregs)}")
        print(f"  Circuit depth: {ann_circuit.depth()}")
        print(f"  Number of operations: {len(ann_circuit.data)}")
        
        # Try to draw the circuit if possible
        try:
            print("\nCircuit diagram (ASCII):")
            print(ann_circuit.draw(output='text'))
        except:
            print("  Could not draw circuit diagram")
    except Exception as e:
        print(f"  Error generating annihilation circuit: {e}")
    
    # Test VQE ansatz
    print("\nGenerating extended VQE ansatz...")
    vqe_circuit = circuits.extended_vqe_ansatz(reps=1, entanglement='linear')
    
    print(f"VQE ansatz details:")
    print(f"  Number of qubits: {vqe_circuit.num_qubits}")
    print(f"  Circuit depth: {vqe_circuit.depth()}")
    print(f"  Number of parameters: {vqe_circuit.num_parameters}")
    
    # Test particle-preserving ansatz
    print("\nGenerating particle-preserving ansatz...")
    pp_circuit = circuits.particle_preserving_ansatz(1, 1, reps=1)
    
    print(f"Particle-preserving ansatz details:")
    print(f"  Number of qubits: {sum(reg.size for reg in pp_circuit.qregs)}")
    print(f"  Circuit depth: {pp_circuit.depth()}")
    print(f"  Number of operations: {len(pp_circuit.data)}")
    
    return circuits

def test_h2_quantum_simulation():
    """Test quantum simulation of H2 molecule."""
    print("\nTesting quantum simulation of H2 molecule...")
    
    if not has_qiskit:
        print("  Skipping test: Qiskit not available")
        return None
    
    # Create H2 molecule
    bond_length = 0.735  # Angstroms
    h1_pos = np.array([0.0, 0.0, 0.0])
    h2_pos = np.array([0.0, 0.0, bond_length])
    
    nuclei = [
        ('H', 1.0, h1_pos),
        ('H', 1.0, h2_pos)
    ]
    
    n_electrons = 2
    n_positrons = 0
    
    # Create a Hamiltonian object
    hamiltonian = AntimatterHamiltonian(
        nuclei, 
        n_electrons, 
        n_positrons,
        include_annihilation=False,
        include_relativistic=False
    )
    
    # Create a basis set
    mixed_basis = MixedMatterBasis()
    molecule = [('H', h1_pos), ('H', h2_pos)]
    mixed_basis.create_for_molecule(molecule, 'minimal', 'minimal')
    
    # Compute integrals
    print("\nComputing integrals...")
    hamiltonian.compute_integrals(mixed_basis)
    
    # Build the Hamiltonian
    hamiltonian_dict = hamiltonian.build_hamiltonian()
    
    # Create Qiskit adapter
    adapter = QiskitNatureAdapter(mapper_type='jordan_wigner')
    
    # Convert to Qiskit operator
    print("\nConverting to Qiskit operator...")
    qiskit_operators = adapter.convert_to_qiskit_hamiltonian(
        hamiltonian_dict,
        n_electrons,
        n_positrons
    )
    
    # Adapt VQE
    print("\nAdapting VQE for H2...")
    vqe_adapters = adapter.adapt_vqe_for_antimatter(
        qiskit_operators,
        optimizer_name='COBYLA',
        ansatz_type='efficient_su2',
        ansatz_reps=1
    )
    
    # Run classical simulation
    print("\nRunning classical eigenvalue calculation...")
    result = adapter.run_ground_state_calculation(
        qiskit_operators,
        vqe_adapters,
        use_classical_solver=True
    )
    
    # Calculate nuclear repulsion
    nuclear_repulsion = 1.0 / bond_length
    
    # Print results
    print("\nH2 calculation results:")
    print(f"  Electronic energy: {result.get('electron_energy', 0):.8f} Hartree")
    print(f"  Nuclear repulsion: {nuclear_repulsion:.8f} Hartree")
    total_energy = result.get('electron_energy', 0) + nuclear_repulsion
    print(f"  Total energy: {total_energy:.8f} Hartree")
    
    # Expected energy for H2 at equilibrium is around -1.13 Hartree
    reference = -1.13
    error = total_energy - reference
    print(f"  Reference energy: {reference:.8f} Hartree")
    print(f"  Error: {error:.8f} Hartree ({error/reference*100:.2f}%)")
    
    return result

if __name__ == "__main__":
    print("=== Qiskit Integration Testing ===\n")
    
    # Test Hamiltonian to Qiskit operator conversion
    qiskit_operators, adapter = test_hamiltonian_to_qiskit_operator()
    
    # Test VQE adaptation
    vqe_result = test_vqe_adaptation()
    
    # Test specialized quantum circuits
    circuits = test_antimatter_circuits()
    
    # Test H2 quantum simulation
    h2_result = test_h2_quantum_simulation()
    
    print("\nAll Qiskit integration tests completed.")