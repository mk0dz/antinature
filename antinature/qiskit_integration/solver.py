# antinature/qiskit_integration/solver.py

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Check Qiskit availability with more detailed error handling
try:
    import qiskit
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter

    # Check for algorithms modules with version compatibility
    try:
        # For newer Qiskit versions
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA

        ALGORITHMS_IMPORT = "qiskit.algorithms"
    except ImportError:
        # For older Qiskit versions
        from qiskit_algorithms import VQE, NumPyMinimumEigensolver
        from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA

        ALGORITHMS_IMPORT = "qiskit_algorithms"

    # Check for primitives
    HAS_PRIMITIVES = False
    try:
        from qiskit.primitives import Estimator

        print("Primitives (Estimator) available.")
        HAS_PRIMITIVES = True
    except ImportError:
        try:
            # Check for new location of Estimator in Qiskit 1.0+ or 2.0+
            from qiskit.primitives import StatevectorEstimator

            # Replace the old API with new one
            Estimator = StatevectorEstimator
            print("Using StatevectorEstimator as replacement for Estimator.")
            HAS_PRIMITIVES = True
        except ImportError:
            print("Primitives not available. Using legacy VQE.")

    HAS_QISKIT = True
except ImportError as e:
    HAS_QISKIT = False
    ALGORITHMS_IMPORT = None
    HAS_PRIMITIVES = False
    print(f"Warning: Qiskit or dependent packages not available. Error: {str(e)}")
    print("Quantum functionality will be limited.")


def create_positronium_circuit(reps: int = 2, include_entanglement: bool = True) -> Any:
    """
    Create a parameterized quantum circuit for positronium VQE.

    Parameters:
    -----------
    reps : int
        Number of repetition layers in the ansatz
    include_entanglement : bool
        Whether to include entanglement gates between electron and positron qubits

    Returns:
    --------
    QuantumCircuit
        Parameterized quantum circuit for VQE

    Notes:
    ------
    Represents positronium as a 2-qubit system with one qubit for the electron
    and one for the positron.
    """
    if not HAS_QISKIT:
        raise ImportError("Qiskit is required for this functionality")

    # Create registers
    e_reg = QuantumRegister(1, 'e')
    p_reg = QuantumRegister(1, 'p')

    # Create circuit
    circuit = QuantumCircuit(e_reg, p_reg)

    # Initialize with Hadamard gates to create superposition
    circuit.h(e_reg[0])
    circuit.h(p_reg[0])

    # Parameters for rotations
    params = []
    for i in range(reps * 6):  # 3 rotations per qubit, 2 qubits
        params.append(Parameter(f'θ_{i}'))

    param_index = 0

    # Build ansatz with repeated blocks
    for r in range(reps):
        # Electron rotations
        circuit.rx(params[param_index], e_reg[0])
        param_index += 1
        circuit.ry(params[param_index], e_reg[0])
        param_index += 1
        circuit.rz(params[param_index], e_reg[0])
        param_index += 1

        # Positron rotations
        circuit.rx(params[param_index], p_reg[0])
        param_index += 1
        circuit.ry(params[param_index], p_reg[0])
        param_index += 1
        circuit.rz(params[param_index], p_reg[0])
        param_index += 1

        # Add entanglement if requested (crucial for positronium)
        if include_entanglement:
            circuit.cx(e_reg[0], p_reg[0])

            # For better results, add a parameterized Z rotation after entanglement
            if r < reps - 1:
                # Add extra parameters for phase adjustment
                phase_param = Parameter(f'φ_{r}')
                params.append(phase_param)
                circuit.rz(phase_param, p_reg[0])
                param_index += 1

    return circuit


class PositroniumVQESolver:
    """
    VQE-based solver for positronium with enhanced features.

    This class provides a specialized Variational Quantum Eigensolver (VQE)
    implementation for positronium systems with customizable optimization
    settings and error handling.
    """

    def __init__(
        self,
        optimizer_name: str = 'COBYLA',
        shots: int = 1024,
        max_iterations: int = 100,
        tol: float = 1e-6,
        callback: Optional[Callable] = None,
    ):
        """
        Initialize the solver with customizable optimization settings.

        Parameters:
        -----------
        optimizer_name : str
            Optimizer to use ('COBYLA', 'SPSA', 'L_BFGS_B', or 'SLSQP')
        shots : int
            Number of shots for each circuit evaluation
        max_iterations : int
            Maximum number of optimizer iterations
        tol : float
            Convergence tolerance
        callback : callable, optional
            Optional callback function for monitoring optimization progress
        """
        if not HAS_QISKIT:
            raise ImportError(
                f"Qiskit is required for this functionality. Please install with 'pip install qiskit {ALGORITHMS_IMPORT}'"
            )

        self.optimizer_name = optimizer_name
        self.shots = shots
        self.max_iterations = max_iterations
        self.tol = tol
        self.callback = callback

        # Initialize estimator if primitives are available
        if HAS_PRIMITIVES:
            try:
                # Try to use the new StatevectorEstimator class if available
                from qiskit.primitives import StatevectorEstimator

                self.estimator = StatevectorEstimator()
                print("Using modern StatevectorEstimator primitive")
            except ImportError:
                # Fall back to legacy Estimator if needed
                self.estimator = Estimator()
                print("Using legacy Estimator primitive")
        else:
            self.estimator = None
            print("Warning: Using legacy VQE without Estimator primitive")

        # Set up optimizer with appropriate parameters
        if optimizer_name == 'COBYLA':
            self.optimizer = COBYLA(maxiter=max_iterations, tol=tol)
        elif optimizer_name == 'SPSA':
            self.optimizer = SPSA(maxiter=max_iterations)
        elif optimizer_name == 'L_BFGS_B':
            self.optimizer = L_BFGS_B(maxiter=max_iterations, ftol=tol)
        elif optimizer_name == 'SLSQP':
            self.optimizer = SLSQP(maxiter=max_iterations, ftol=tol)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def solve_positronium(
        self,
        mapper_type: str = 'jordan_wigner',
        reps: int = 3,
        include_entanglement: bool = True,
        use_classical: bool = False,
        initial_point: Optional[np.ndarray] = None,
        n_tries: int = 3,
    ) -> Dict:
        """
        Solve for positronium ground state using VQE with enhanced features.

        Parameters:
        -----------
        mapper_type : str
            Mapper to use for fermion-to-qubit mapping ('jordan_wigner' or 'parity')
        reps : int
            Number of repetitions in the ansatz
        include_entanglement : bool
            Whether to include entanglement gates in the circuit
        use_classical : bool
            Whether to use classical solver for comparison
        initial_point : np.ndarray, optional
            Initial parameters for the optimizer
        n_tries : int
            Number of optimization attempts with different starting points

        Returns:
        --------
        Dict
            Results including energy and other properties
        """
        try:
            # Import the adapter class directly to avoid circular imports
            from .adapter import PositroniumAdapter
        except ImportError:
            raise ImportError(
                "PositroniumAdapter not found. Please ensure the adapter.py module is available."
            )

        # Create adapter
        try:
            adapter = PositroniumAdapter(mapper_type=mapper_type)
        except Exception as e:
            raise RuntimeError(f"Failed to create PositroniumAdapter: {str(e)}")

        # Create positronium Hamiltonian with optimized parameters
        try:
            problem, qubit_op = adapter.create_positronium_hamiltonian(
                e_repulsion=0.0, p_repulsion=0.0, ep_attraction=-1.0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create positronium Hamiltonian: {str(e)}")

        # Solve classically if requested
        classical_energy = None
        if use_classical:
            try:
                numpy_solver = NumPyMinimumEigensolver()
                classical_result = numpy_solver.compute_minimum_eigenvalue(qubit_op)
                classical_energy = classical_result.eigenvalue.real
                print(
                    f"Classical computation completed. Energy: {classical_energy:.6f}"
                )
            except Exception as e:
                print(f"Warning: Classical solution failed: {str(e)}")

        # Create ansatz using the function (to avoid circular imports)
        try:
            ansatz = create_positronium_circuit(
                reps=reps, include_entanglement=include_entanglement
            )
            print(f"Created ansatz with {ansatz.num_parameters} parameters")
        except Exception as e:
            raise RuntimeError(f"Failed to create positronium circuit: {str(e)}")

        # Generate initial point if not provided
        if initial_point is None:
            num_params = ansatz.num_parameters
            # Use physics-informed initial point
            initial_point = np.random.uniform(-0.1, 0.1, size=num_params)

            # Set some specific values for better convergence
            if include_entanglement and num_params >= 7:
                # Add a slight bias to encourage electron-positron correlation
                for i in range(0, num_params, 6):
                    if i + 2 < num_params:  # rz parameters
                        initial_point[i + 2] = np.pi / 4
                    if i + 5 < num_params:  # rz parameters
                        initial_point[i + 5] = np.pi / 4

        # Best result storage
        best_energy = float('inf')
        best_result = None

        # Multiple attempts with different starting points
        for attempt in range(n_tries):
            try:
                # Adjust initial point for retries
                if attempt > 0:
                    print(f"VQE attempt {attempt+1}/{n_tries}...")
                    # Perturb initial point for new attempts
                    perturbed_point = initial_point + 0.2 * np.random.randn(
                        len(initial_point)
                    )
                    current_point = perturbed_point
                else:
                    current_point = initial_point

                # Initialize VQE with appropriate settings
                if HAS_PRIMITIVES:
                    vqe = VQE(
                        estimator=self.estimator,
                        ansatz=ansatz,
                        optimizer=self.optimizer,
                        initial_point=current_point,
                        callback=self.callback,
                    )
                else:
                    # Legacy initialization for older Qiskit versions
                    vqe = VQE(
                        ansatz=ansatz,
                        optimizer=self.optimizer,
                        initial_point=current_point,
                        callback=self.callback,
                    )

                # Run VQE
                vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)

                # Extract results
                energy = vqe_result.eigenvalue.real

                # Update best result if better
                if energy < best_energy:
                    best_energy = energy
                    best_result = vqe_result

                # If we're close to theoretical value, stop attempts
                if abs(energy - (-0.25)) < 0.05:
                    print(
                        f"Found solution close to theoretical value. Stopping attempts."
                    )
                    break

            except Exception as e:
                print(f"Warning: VQE attempt {attempt+1} failed: {str(e)}")
                if attempt == n_tries - 1:
                    raise RuntimeError(f"All VQE attempts failed. Last error: {str(e)}")

        # Extract results from best run
        vqe_result = best_result
        vqe_energy = best_result.eigenvalue.real
        optimal_parameters = best_result.optimal_parameters
        iterations = getattr(
            vqe_result,
            'optimizer_evals',
            getattr(vqe_result, 'cost_function_evals', None),
        )

        # Compute error from theoretical value
        theoretical_energy = -0.25  # Hartree
        error = abs(vqe_energy - theoretical_energy)

        # Build comprehensive results
        results = {
            'vqe_energy': vqe_energy,
            'parameters': optimal_parameters,
            'classical_energy': classical_energy,
            'theoretical_energy': theoretical_energy,
            'vqe_error': error,
            'iterations': iterations,
            'error_percentage': error / abs(theoretical_energy) * 100,
            'ansatz_depth': ansatz.depth(),
            'ansatz_gate_counts': ansatz.count_ops(),
            'algorithm_version': ALGORITHMS_IMPORT,
            'mapper_type': mapper_type,
            'optimizer': self.optimizer_name,
            'shots': self.shots,
        }

        # Add VQE result object if available
        if vqe_result is not None:
            results['vqe_result'] = vqe_result

        return results
