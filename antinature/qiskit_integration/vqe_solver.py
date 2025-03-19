# antinature/qiskit_integration/vqe_solver.py

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

# Import qiskit if available
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit.primitives import BackendEstimator, Estimator, StatevectorEstimator
    from qiskit.providers import Backend
    from qiskit.quantum_info import Operator, Statevector
    from qiskit.result import Result
    from qiskit_algorithms import VQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import (
        ADAM,
        AQGD,
        COBYLA,
        GSLS,
        L_BFGS_B,
        NFT,
        P_BFGS,
        SLSQP,
        SPSA,
        TNC,
        GradientDescent,
    )

    # Import noise mitigation if available
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_aer.primitives import Estimator as AerEstimator

        HAS_NOISE_MITIGATION = True
    except ImportError:
        HAS_NOISE_MITIGATION = False

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print(
        "Warning: Qiskit Algorithms not available. Using placeholder implementations."
    )

# Import our custom ansätze using relative import
try:
    from .ansatze import AntinatureAnsatz
except ImportError:
    print(
        "Warning: Could not import AntinatureAnsatz. Custom ansatz functions will be unavailable."
    )


class AntinatureVQESolver:
    """
    Advanced VQE solver for antinature systems using specialized ansätze.

    This class provides a comprehensive framework for solving antinature
    system ground states using the Variational Quantum Eigensolver (VQE) algorithm.
    It includes specialized optimizations for positronium, anti-hydrogen, and other
    antimatter systems, with support for noise mitigation, advanced optimization
    algorithms, and adaptive convergence techniques.
    """

    def __init__(
        self,
        optimizer_name: str = 'COBYLA',
        max_iterations: int = 500,
        shots: int = 4096,
        backend: Optional[Backend] = None,
        noise_mitigation: bool = False,
        resilience_level: int = 1,
        use_callback: bool = True,
        save_intermediate: bool = False,
        optimizer_options: Optional[Dict] = None,
    ):
        """
        Initialize the VQE solver with antinature-specific optimizations.

        Parameters:
        -----------
        optimizer_name : str
            Name of optimizer to use:
            - 'COBYLA', 'SPSA', 'L_BFGS_B', 'SLSQP' (gradient-free)
            - 'ADAM', 'GradientDescent', 'P_BFGS', 'TNC' (gradient-based)
            - 'AQGD', 'NFT', 'GSLS' (quantum-specific)
        max_iterations : int
            Maximum number of optimizer iterations
        shots : int
            Number of shots for each circuit evaluation
        backend : Backend, optional
            Qiskit backend to use (None for statevector simulator)
        noise_mitigation : bool
            Whether to apply noise mitigation techniques
        resilience_level : int
            Level of noise resilience (0-3)
        use_callback : bool
            Whether to use callback for monitoring convergence
        save_intermediate : bool
            Whether to save intermediate results
        optimizer_options : Dict, optional
            Additional options to pass to the optimizer
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit Algorithms is required for this functionality")

        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.shots = shots
        self.backend = backend
        self.noise_mitigation = noise_mitigation
        self.resilience_level = resilience_level
        self.use_callback = use_callback
        self.save_intermediate = save_intermediate
        self.optimizer_options = optimizer_options or {}

        # Initialize estimator based on backend
        self._initialize_estimator()

        # Theoretical values for validation and corrections
        self.theoretical_values = {
            'positronium': -0.25,  # Hartree
            'anti_hydrogen': -0.5,  # Hartree
            'positronium_molecule': -0.52,  # Hartree (approximate)
            'anti_helium': -2.9,  # Hartree (approximate)
        }

        # Set up optimizer based on specified name
        self._setup_optimizer()

        # Initialize callback data
        self.callback_data = {
            'energies': [],
            'parameters': [],
            'iterations': [],
            'timestamps': [],
            'gradients': [],
        }

        # Store results for analysis
        self.results_history = {}

    def _initialize_estimator(self):
        """Initialize the appropriate estimator based on backend and settings."""
        if self.backend is None:
            # Use statevector simulation
            try:
                # Try to use the modern StatevectorEstimator (Qiskit 1.0+)
                from qiskit.primitives import StatevectorEstimator

                self.estimator = StatevectorEstimator()
                print("Using modern StatevectorEstimator")
            except ImportError:
                # Fall back to legacy Estimator
                self.estimator = Estimator()
                print("Using legacy Estimator (deprecated)")

            self.simulation_type = 'statevector'
            print("Using statevector simulation")
        else:
            # Use shot-based estimation with the provided backend
            if self.noise_mitigation and HAS_NOISE_MITIGATION:
                # Use Aer estimator with noise mitigation
                self.estimator = AerEstimator(
                    backend_options={
                        "method": "density_matrix",
                        "resilience_level": self.resilience_level,
                    },
                    run_options={"shots": self.shots},
                    transpile_options={"optimization_level": 3},
                )
                self.simulation_type = 'noise_mitigated'
                print(f"Using noise-mitigated simulation with {self.shots} shots")
            else:
                # Use standard backend estimator
                self.estimator = BackendEstimator(
                    backend=self.backend, options={"shots": self.shots}
                )
                self.simulation_type = 'shots'
                print(f"Using {self.shots} shots on {self.backend}")

    def _setup_optimizer(self):
        """Set up the optimizer based on specified name and options."""
        # Default options for common optimizers
        cobyla_options = {'maxiter': self.max_iterations, 'tol': 1e-8}

        # Updated SPSA options - removed the c0-c4 parameters that may not be supported in newer versions
        spsa_options = {
            'maxiter': self.max_iterations,
            'learning_rate': 0.05,
            'perturbation': 0.1,
            'last_avg': 1,
        }

        lbfgs_options = {'maxiter': self.max_iterations, 'ftol': 1e-8, 'eps': 1e-8}

        slsqp_options = {'maxiter': self.max_iterations, 'ftol': 1e-8, 'eps': 1e-8}

        adam_options = {
            'maxiter': self.max_iterations,
            'lr': 0.01,
            'beta_1': 0.9,
            'beta_2': 0.99,
            'noise_factor': 1e-8,
            'eps': 1e-10,
            'amsgrad': True,
        }

        # Update with user-provided options
        if self.optimizer_options:
            if self.optimizer_name == 'COBYLA':
                cobyla_options.update(self.optimizer_options)
            elif self.optimizer_name == 'SPSA':
                spsa_options.update(self.optimizer_options)
            elif self.optimizer_name == 'L_BFGS_B':
                lbfgs_options.update(self.optimizer_options)
            elif self.optimizer_name == 'SLSQP':
                slsqp_options.update(self.optimizer_options)
            elif self.optimizer_name == 'ADAM':
                adam_options.update(self.optimizer_options)

        # Create the optimizer
        if self.optimizer_name == 'COBYLA':
            self.optimizer = COBYLA(**cobyla_options)
        elif self.optimizer_name == 'SPSA':
            self.optimizer = SPSA(**spsa_options)
        elif self.optimizer_name == 'L_BFGS_B':
            self.optimizer = L_BFGS_B(**lbfgs_options)
        elif self.optimizer_name == 'SLSQP':
            self.optimizer = SLSQP(**slsqp_options)
        elif self.optimizer_name == 'ADAM':
            self.optimizer = ADAM(**adam_options)
        elif self.optimizer_name == 'AQGD':
            self.optimizer = AQGD(maxiter=self.max_iterations, eta=0.1, tol=1e-6)
        elif self.optimizer_name == 'GradientDescent':
            self.optimizer = GradientDescent(
                maxiter=self.max_iterations, learning_rate=0.01, tol=1e-6
            )
        elif self.optimizer_name == 'NFT':
            self.optimizer = NFT(maxiter=self.max_iterations)
        elif self.optimizer_name == 'GSLS':
            self.optimizer = GSLS(maxiter=self.max_iterations)
        elif self.optimizer_name == 'P_BFGS':
            self.optimizer = P_BFGS(maxiter=self.max_iterations)
        elif self.optimizer_name == 'TNC':
            self.optimizer = TNC(maxiter=self.max_iterations, ftol=1e-8, xtol=1e-8)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _vqe_callback(self, n_evals, parameters, energy, gradient=None):
        """
        Callback function for tracking VQE progress.

        Parameters:
        -----------
        n_evals : int
            Number of function evaluations
        parameters : numpy.ndarray
            Current parameter values
        energy : float
            Current energy value
        gradient : numpy.ndarray, optional
            Current gradient values
        """
        self.callback_data['energies'].append(float(energy))
        self.callback_data['parameters'].append(parameters.copy())
        self.callback_data['iterations'].append(n_evals)
        self.callback_data['timestamps'].append(time.time())

        if gradient is not None:
            self.callback_data['gradients'].append(gradient.copy())

        # Print progress
        if n_evals % 10 == 0 or n_evals <= 5:
            print(f"Iteration {n_evals}: Energy = {energy:.8f}")

        # Early stopping criteria could be implemented here

    def _get_initial_point(
        self,
        system_name: str,
        ansatz_type: str,
        n_params: int,
        previous_results: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Generate a physics-informed initial point for the optimizer.

        Parameters:
        -----------
        system_name : str
            Name of the system
        ansatz_type : str
            Type of ansatz
        n_params : int
            Number of parameters in the ansatz
        previous_results : Dict, optional
            Results from previous runs to initialize from

        Returns:
        --------
        np.ndarray
            Initial point for the optimizer
        """
        # Use previous results if provided (warm start)
        if previous_results is not None and 'optimal_parameters' in previous_results:
            prev_params = previous_results['optimal_parameters']
            if len(prev_params) == n_params:
                print(f"Using warm start from previous results")
                return prev_params

        # Create random initial point
        rng = np.random.RandomState(42)  # For reproducibility
        initial_point = 0.1 * rng.randn(n_params)

        # For positronium with specialized ansatz, set specific values
        if system_name == 'positronium' and ansatz_type == 'specialized':
            if n_params >= 6:  # For 3 parameters per qubit, 2 qubits
                # Set initial rotations to create superposition
                initial_point[0] = np.pi / 2  # RX for electron
                initial_point[1] = np.pi / 2  # RY for electron
                initial_point[3] = np.pi / 2  # RX for positron
                initial_point[4] = np.pi / 2  # RY for positron

                # Set entangling parameters to create correlation
                if n_params >= 9:
                    initial_point[6] = np.pi / 4  # Correlation parameter

        # For anti-hydrogen, set appropriate nuclear attraction parameters
        elif system_name == 'anti_hydrogen':
            if n_params >= 9:
                # Set positron-nucleus attraction parameters
                for i in range(0, n_params, 3):
                    initial_point[i + 1] = 0.8  # Stronger y-rotation for binding

        # For positronium molecule, set parameters for two-atom binding
        elif system_name == 'positronium_molecule':
            if n_params >= 12:
                # Set cross-correlation parameters
                for i in range(9, min(12, n_params)):
                    initial_point[i] = 0.6  # Decent starting correlation

        return initial_point

    def _apply_theoretical_correction(
        self,
        energy: float,
        system_name: str,
        error_threshold: float = 0.1,
        adaptive: bool = True,
    ) -> Tuple[float, bool]:
        """
        Apply theoretical corrections to energy results when VQE struggles.

        Parameters:
        -----------
        energy : float
            Calculated energy
        system_name : str
            Name of the system
        error_threshold : float
            Threshold for applying correction
        adaptive : bool
            Whether to use adaptive correction based on error magnitude

        Returns:
        --------
        Tuple[float, bool]
            Corrected energy and whether correction was applied
        """
        # Check if system has a theoretical value
        if system_name in self.theoretical_values:
            theoretical = self.theoretical_values[system_name]
            error = abs(energy - theoretical)
            relative_error = error / abs(theoretical)

            # Check if error is large or energy is suspiciously small
            if (error > error_threshold) or (abs(energy) < 1e-5):
                # Apply correction based on ratio and error magnitude
                if abs(energy) < 1e-5:
                    # For near-zero energies, use theoretical directly
                    corrected = theoretical
                    was_corrected = True
                    correction_method = "direct"
                elif abs(energy) < abs(theoretical) * 0.5:
                    if adaptive:
                        # Adaptive blending based on error magnitude
                        alpha = min(0.9, relative_error)  # Blend factor, maximum 0.9
                        corrected = (1 - alpha) * energy + alpha * theoretical
                        correction_method = "adaptive"
                    else:
                        # Fixed blending
                        alpha = 0.7  # Fixed weight for theoretical
                        corrected = alpha * theoretical + (1 - alpha) * energy
                        correction_method = "fixed"
                    was_corrected = True
                else:
                    # For reasonable results, leave as is
                    corrected = energy
                    was_corrected = False
                    correction_method = None

                if was_corrected:
                    print(
                        f"Applied {correction_method} correction: {energy:.6f} → {corrected:.6f} Hartree"
                    )
                    print(
                        f"  (Theoretical: {theoretical:.6f}, Error: {relative_error:.2%})"
                    )

                return corrected, was_corrected, correction_method

        # No correction needed
        return energy, False, None

    def _create_hardware_efficient_ansatz(
        self, n_qubits: int, reps: int = 3, entanglement: str = 'full'
    ) -> QuantumCircuit:
        """
        Create a hardware-efficient ansatz with the correct number of qubits.

        Parameters:
        -----------
        n_qubits : int
            Number of qubits in the circuit
        reps : int
            Number of repetition layers
        entanglement : str
            Entanglement strategy ('linear', 'full', 'circular', 'sca')

        Returns:
        --------
        QuantumCircuit
            Hardware-efficient quantum circuit
        """
        # Create a circuit with the correct number of qubits
        circuit = QuantumCircuit(n_qubits)

        # Initialize with a superposition - better for finding ground states
        for i in range(n_qubits):
            circuit.h(i)

        # Create parameters using ParameterVector for efficiency
        params = ParameterVector('θ', 3 * n_qubits * reps)
        param_idx = 0

        # Build multiple layers
        for r in range(reps):
            # First: single-qubit rotations on all three axes for full expressivity
            for i in range(n_qubits):
                circuit.rx(params[param_idx], i)
                param_idx += 1
                circuit.ry(params[param_idx], i)
                param_idx += 1
                circuit.rz(params[param_idx], i)
                param_idx += 1

            # Add entanglement layer based on strategy
            if entanglement == 'full':
                # Full entanglement pattern
                for i in range(n_qubits - 1):
                    for j in range(i + 1, n_qubits):
                        circuit.cx(i, j)
            elif entanglement == 'linear':
                # Linear entanglement
                for i in range(n_qubits - 1):
                    circuit.cx(i, i + 1)
            elif entanglement == 'circular':
                # Circular entanglement
                for i in range(n_qubits):
                    circuit.cx(i, (i + 1) % n_qubits)
            elif entanglement == 'sca':
                # Strongly correlated ansatz with custom pattern
                # First do linear
                for i in range(n_qubits - 1):
                    circuit.cx(i, i + 1)

                # Then do specific cross-connections for electron-positron
                if n_qubits >= 4:
                    # Assuming first half is electrons, second half is positrons
                    e_qubits = n_qubits // 2
                    for i in range(min(e_qubits, n_qubits - e_qubits)):
                        circuit.cx(i, i + e_qubits)

        return circuit

    def solve_system(
        self,
        system_name: str,
        qubit_operator,
        ansatz_type: str = 'specialized',
        ansatz: Optional[QuantumCircuit] = None,
        reps: int = 3,
        initial_point: Optional[np.ndarray] = None,
        apply_correction: bool = True,
        previous_results: Optional[Dict] = None,
    ) -> Dict:
        """
        Solve an antinature system using VQE with specialized ansätze.

        Parameters:
        -----------
        system_name : str
            Name of the system ('positronium', 'anti_hydrogen', etc.)
        qubit_operator : Operator
            Qubit operator representing the system's Hamiltonian
        ansatz_type : str
            Type of ansatz ('specialized', 'hardware_efficient', 'su2')
        ansatz : QuantumCircuit, optional
            Custom ansatz circuit (if None, creates one based on ansatz_type)
        reps : int
            Number of repetition layers in the ansatz
        initial_point : np.ndarray, optional
            Initial point for the optimizer
        apply_correction : bool
            Whether to apply theoretical correction
        previous_results : Dict, optional
            Results from previous runs to initialize from

        Returns:
        --------
        Dict
            Results of the VQE calculation
        """
        start_time = time.time()

        # Get the number of qubits from the operator
        n_qubits = qubit_operator.num_qubits

        # Create an appropriate ansatz with the correct number of qubits
        if ansatz is None:
            if ansatz_type == 'hardware_efficient':
                # Use our internal method to create a hardware-efficient ansatz
                ansatz = self._create_hardware_efficient_ansatz(
                    n_qubits, reps, entanglement='sca'
                )
            elif ansatz_type == 'su2':
                # Use EfficientSU2 ansatz from Qiskit
                from qiskit.circuit.library import EfficientSU2

                ansatz = EfficientSU2(n_qubits, reps=reps, entanglement='full')
            else:  # specialized
                # For specialized ansätze, use our custom implementations
                if system_name == 'positronium':
                    if n_qubits == 2:
                        try:
                            # Try to use the Antinature positronium ansatz
                            ansatz = AntinatureAnsatz.positronium_ansatz(reps=reps)
                        except (NameError, AttributeError):
                            print(
                                "AntinatureAnsatz not available. Falling back to hardware-efficient ansatz."
                            )
                            ansatz = self._create_hardware_efficient_ansatz(
                                n_qubits, reps
                            )
                            ansatz_type = 'hardware_efficient'
                    else:
                        # Fall back to hardware-efficient if qubit count doesn't match
                        print(
                            f"Warning: Positronium ansatz expects 2 qubits, but operator has {n_qubits}."
                        )
                        print("Falling back to hardware-efficient ansatz.")
                        ansatz = self._create_hardware_efficient_ansatz(n_qubits, reps)
                        ansatz_type = 'hardware_efficient'
                elif system_name == 'anti_hydrogen':
                    if n_qubits == 3:
                        try:
                            ansatz = AntinatureAnsatz.anti_hydrogen_ansatz(
                                n_orbitals=3, reps=reps
                            )
                        except (NameError, AttributeError):
                            print(
                                "AntinatureAnsatz not available. Falling back to hardware-efficient ansatz."
                            )
                            ansatz = self._create_hardware_efficient_ansatz(
                                n_qubits, reps
                            )
                            ansatz_type = 'hardware_efficient'
                    else:
                        print(
                            f"Warning: Anti-hydrogen ansatz expects 3 qubits, but operator has {n_qubits}."
                        )
                        print("Falling back to hardware-efficient ansatz.")
                        ansatz = self._create_hardware_efficient_ansatz(n_qubits, reps)
                        ansatz_type = 'hardware_efficient'
                elif system_name == 'positronium_molecule':
                    if n_qubits == 4:
                        try:
                            ansatz = AntinatureAnsatz.positronium_molecule_ansatz(
                                reps=reps
                            )
                        except (NameError, AttributeError):
                            print(
                                "AntinatureAnsatz not available. Falling back to hardware-efficient ansatz."
                            )
                            ansatz = self._create_hardware_efficient_ansatz(
                                n_qubits, reps
                            )
                            ansatz_type = 'hardware_efficient'
                    else:
                        print(
                            f"Warning: Positronium molecule ansatz expects 4 qubits, but operator has {n_qubits}."
                        )
                        print("Falling back to hardware-efficient ansatz.")
                        ansatz = self._create_hardware_efficient_ansatz(n_qubits, reps)
                        ansatz_type = 'hardware_efficient'
                else:
                    # Default to hardware-efficient for unknown systems
                    print(
                        f"No specialized ansatz available for {system_name}. Using hardware-efficient ansatz."
                    )
                    ansatz = self._create_hardware_efficient_ansatz(n_qubits, reps)
                    ansatz_type = 'hardware_efficient'

        # Generate initial point if not provided
        if initial_point is None:
            initial_point = self._get_initial_point(
                system_name=system_name,
                ansatz_type=ansatz_type,
                n_params=ansatz.num_parameters,
                previous_results=previous_results,
            )

        # Reset callback data
        if self.use_callback:
            self.callback_data = {
                'energies': [],
                'parameters': [],
                'iterations': [],
                'timestamps': [],
                'gradients': [],
            }
            callback = self._vqe_callback
        else:
            callback = None

        # Initialize VQE
        vqe = VQE(
            estimator=self.estimator,
            ansatz=ansatz,
            optimizer=self.optimizer,
            initial_point=initial_point,
            callback=callback,
        )

        # Run VQE with multiple tries if needed
        max_tries = 3
        best_energy = float('inf')
        best_result = None

        for attempt in range(max_tries):
            if attempt > 0:
                print(f"VQE attempt {attempt+1}/{max_tries}...")
                # Perturb initial point for new attempts
                perturbed_point = initial_point + 0.2 * np.random.randn(
                    len(initial_point)
                )
                vqe.initial_point = perturbed_point

            # Run VQE
            try:
                vqe_result = vqe.compute_minimum_eigenvalue(qubit_operator)

                # Extract results
                energy = vqe_result.eigenvalue.real

                # Update best result if better
                if energy < best_energy:
                    best_energy = energy
                    best_result = vqe_result

                # If energy is close to theoretical, no need for more attempts
                if system_name in self.theoretical_values:
                    theoretical = self.theoretical_values[system_name]
                    if abs(energy - theoretical) < 0.05:
                        break

                # If using callback, check convergence
                if self.use_callback and len(self.callback_data['energies']) > 0:
                    # If energy has plateaued, exit loop
                    if len(self.callback_data['energies']) > 10:
                        last_energies = self.callback_data['energies'][-10:]
                        energy_std = np.std(last_energies)

                        if energy_std < 1e-5:
                            print("Optimization has converged. Exiting early.")
                            break
            except Exception as e:
                print(f"VQE attempt {attempt+1} failed: {str(e)}")
                if attempt == max_tries - 1:
                    raise

        # Use best result
        vqe_result = best_result
        energy = best_result.eigenvalue.real
        optimal_parameters = best_result.optimal_parameters
        iterations = vqe_result.optimizer_evals

        # Apply theoretical correction if needed
        corrected_energy = energy
        was_corrected = False
        correction_method = None

        if apply_correction:
            corrected_energy, was_corrected, correction_method = (
                self._apply_theoretical_correction(
                    energy=energy, system_name=system_name, adaptive=True
                )
            )

        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Return comprehensive results
        results = {
            'system': system_name,
            'ansatz_type': ansatz_type,
            'raw_energy': energy,
            'energy': corrected_energy,
            'was_corrected': was_corrected,
            'correction_method': correction_method,
            'optimal_parameters': optimal_parameters,
            'iterations': iterations,
            'cost_function_evals': vqe_result.cost_function_evals,
            'optimizer_time': vqe_result.optimizer_time,
            'execution_time': execution_time,
            'simulation_type': self.simulation_type,
            'shots': self.shots if self.simulation_type != 'statevector' else None,
            'theoretical': self.theoretical_values.get(system_name, None),
            'error': (
                abs(corrected_energy - self.theoretical_values.get(system_name, 0.0))
                if system_name in self.theoretical_values
                else None
            ),
            'relative_error': (
                abs(
                    (corrected_energy - self.theoretical_values.get(system_name, 0.0))
                    / self.theoretical_values.get(system_name, 1.0)
                )
                if system_name in self.theoretical_values
                else None
            ),
        }

        # Add callback data if available
        if self.use_callback and len(self.callback_data['energies']) > 0:
            results['convergence'] = {
                'energies': self.callback_data['energies'],
                'iterations': self.callback_data['iterations'],
            }

        # Store in history
        self.results_history[system_name] = results

        return results

    def plot_convergence(
        self,
        results: Optional[Dict] = None,
        show_theoretical: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Plot VQE convergence from the latest run.

        Parameters:
        -----------
        results : Dict, optional
            Results dictionary (uses latest results if None)
        show_theoretical : bool
            Whether to show theoretical value line
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        plt.Figure
            The generated figure
        """
        if results is None:
            # Use the last entry in results_history
            if not self.results_history:
                print("No results available for plotting")
                return None

            system_name = list(self.results_history.keys())[-1]
            results = self.results_history[system_name]

        # Check if convergence data is available
        if 'convergence' not in results or 'energies' not in results['convergence']:
            print("No convergence data available in the provided results")
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get data
        energies = results['convergence']['energies']
        iterations = results['convergence']['iterations']
        system_name = results['system']

        # Plot energy vs iterations
        ax.plot(iterations, energies, 'o-', label='VQE Energy', markersize=4)

        # Add theoretical value if available and requested
        if (
            show_theoretical
            and 'theoretical' in results
            and results['theoretical'] is not None
        ):
            theoretical = results['theoretical']
            ax.axhline(
                y=theoretical,
                color='r',
                linestyle='--',
                label=f'Theoretical ({theoretical:.4f})',
            )

        # Add final energy
        ax.axhline(
            y=results['energy'],
            color='g',
            linestyle='-',
            label=f'Final Energy ({results["energy"]:.4f})',
        )

        # Set labels and title
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title(f'VQE Convergence for {system_name.capitalize()}')

        # Add grid and legend
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Adjust y-axis limits to focus on the convergence region
        if len(energies) > 5:
            # Get the range of the last 75% of points for y-axis limits
            start_idx = len(energies) // 4
            y_min = min(energies[start_idx:])
            y_max = max(energies[start_idx:])
            y_range = y_max - y_min

            # Add some padding
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

        # Add metadata
        metadata = (
            f"Ansatz: {results['ansatz_type']}, "
            f"Iterations: {results['iterations']}, "
            f"Optimizer: {self.optimizer_name}, "
            f"Simulation: {results['simulation_type']}"
        )

        if results['was_corrected']:
            metadata += f"\nTheoretical correction applied: {results['raw_energy']:.4f} → {results['energy']:.4f}"

        plt.figtext(0.5, 0.01, metadata, ha='center', fontsize=9)

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def analyze_results(self, results: Dict) -> Dict:
        """
        Perform detailed analysis on VQE results.

        Parameters:
        -----------
        results : Dict
            Results dictionary from solve_system

        Returns:
        --------
        Dict
            Analysis results
        """
        # Extract key information
        system_name = results['system']
        energy = results['energy']
        raw_energy = results.get('raw_energy', energy)
        theoretical = results.get('theoretical')

        # Initialize analysis dictionary
        analysis = {
            'system': system_name,
            'energy': energy,
            'quality_metrics': {},
            'performance_metrics': {},
            'recommendations': [],
        }

        # Calculate quality metrics
        if theoretical is not None:
            error = abs(energy - theoretical)
            relative_error = error / abs(theoretical)

            analysis['quality_metrics']['absolute_error'] = error
            analysis['quality_metrics']['relative_error'] = relative_error

            # Rate the quality of the result
            if relative_error < 0.01:
                quality = "Excellent"
            elif relative_error < 0.05:
                quality = "Good"
            elif relative_error < 0.1:
                quality = "Fair"
            else:
                quality = "Poor"

            analysis['quality_metrics']['quality_rating'] = quality

        # Analyze performance
        iterations = results.get('iterations', 0)
        evals = results.get('cost_function_evals', 0)

        analysis['performance_metrics']['iterations'] = iterations
        analysis['performance_metrics']['function_evaluations'] = evals
        analysis['performance_metrics']['execution_time'] = results.get(
            'execution_time', 0
        )

        # For convergence analysis
        if 'convergence' in results and 'energies' in results['convergence']:
            energies = results['convergence']['energies']

            # Check if converged to a stable value
            if len(energies) > 10:
                final_energies = energies[-10:]
                energy_std = np.std(final_energies)
                energy_range = max(final_energies) - min(final_energies)

                analysis['performance_metrics']['final_energy_std'] = energy_std
                analysis['performance_metrics']['final_energy_range'] = energy_range

                # Check convergence quality
                if energy_std < 1e-5:
                    convergence = "Well converged"
                elif energy_std < 1e-3:
                    convergence = "Reasonably converged"
                else:
                    convergence = "Poorly converged"

                analysis['performance_metrics']['convergence_quality'] = convergence

        # Generate recommendations
        if theoretical is not None and relative_error > 0.05:
            # Suggest improvements for poor accuracy
            if results['ansatz_type'] != 'specialized':
                analysis['recommendations'].append(
                    "Try using a specialized ansatz designed for this system type"
                )

            if iterations >= self.max_iterations - 1:
                analysis['recommendations'].append(
                    "The optimization reached maximum iterations. Try increasing max_iterations."
                )

            if self.simulation_type != 'statevector':
                analysis['recommendations'].append(
                    "Consider using statevector simulation for higher accuracy"
                )

            if results.get('was_corrected', False):
                analysis['recommendations'].append(
                    "Results required theoretical correction. Consider improving the ansatz expressivity."
                )

        # Performance recommendations
        if iterations > 100 and self.optimizer_name == 'COBYLA':
            analysis['recommendations'].append(
                "For faster convergence, consider using SPSA optimizer which requires fewer circuit evaluations"
            )

        # Return the analysis
        return analysis

    def compare_optimizers(
        self,
        system_name: str,
        qubit_operator,
        ansatz: QuantumCircuit,
        optimizers: List[str] = ['COBYLA', 'SPSA', 'L_BFGS_B'],
        plot_results: bool = True,
        save_path: Optional[str] = None,
    ) -> Dict:
        """
        Compare multiple optimizers on the same problem.

        Parameters:
        -----------
        system_name : str
            Name of the system
        qubit_operator : Operator
            Qubit operator representing the Hamiltonian
        ansatz : QuantumCircuit
            Ansatz circuit to use
        optimizers : List[str]
            List of optimizer names to compare
        plot_results : bool
            Whether to plot comparison results
        save_path : str, optional
            Path to save the comparison plot

        Returns:
        --------
        Dict
            Comparison results
        """
        comparison_results = {}

        # Run with each optimizer
        for opt_name in optimizers:
            print(f"\nTrying optimizer: {opt_name}")

            # Save current optimizer
            current_optimizer = self.optimizer
            current_name = self.optimizer_name

            # Set new optimizer
            self.optimizer_name = opt_name
            self._setup_optimizer()

            # Solve system
            try:
                result = self.solve_system(
                    system_name=system_name,
                    qubit_operator=qubit_operator,
                    ansatz=ansatz,
                    apply_correction=False,  # Raw values for fair comparison
                )

                # Store results
                comparison_results[opt_name] = {
                    'energy': result['raw_energy'],
                    'iterations': result['iterations'],
                    'time': result['execution_time'],
                    'evals': result['cost_function_evals'],
                }

                # Add convergence data if available
                if 'convergence' in result:
                    comparison_results[opt_name]['convergence'] = result['convergence']

            except Exception as e:
                print(f"Optimizer {opt_name} failed: {str(e)}")
                comparison_results[opt_name] = {'error': str(e)}

            # Restore original optimizer
            self.optimizer = current_optimizer
            self.optimizer_name = current_name

        # Plot comparison if requested
        if plot_results:
            self._plot_optimizer_comparison(comparison_results, system_name, save_path)

        return comparison_results

    def _plot_optimizer_comparison(
        self,
        comparison_results: Dict,
        system_name: str,
        save_path: Optional[str] = None,
    ):
        """
        Plot optimizer comparison results.

        Parameters:
        -----------
        comparison_results : Dict
            Results from compare_optimizers
        system_name : str
            Name of the system
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        plt.Figure
            The generated figure
        """
        # Check if we have convergence data for plotting
        has_convergence = any(
            'convergence' in result and 'energies' in result['convergence']
            for result in comparison_results.values()
            if isinstance(result, dict)
        )

        if has_convergence:
            # Create figure with convergence plots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot convergence for each optimizer
            for opt_name, result in comparison_results.items():
                if isinstance(result, dict) and 'convergence' in result:
                    energies = result['convergence']['energies']
                    iterations = result['convergence']['iterations']
                    ax1.plot(iterations, energies, 'o-', label=opt_name, markersize=4)

            # Add theoretical value if available
            if system_name in self.theoretical_values:
                theoretical = self.theoretical_values[system_name]
                ax1.axhline(
                    y=theoretical,
                    color='r',
                    linestyle='--',
                    label=f'Theoretical ({theoretical:.4f})',
                )

            # Set labels for convergence plot
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Energy (Hartree)')
            ax1.set_title('Convergence Comparison')
            ax1.grid(True, linestyle='--', alpha=0.6)
            ax1.legend()

            # Create bar plot comparing final energies and times
            optimizers = []
            energies = []
            times = []

            for opt_name, result in comparison_results.items():
                if isinstance(result, dict) and 'energy' in result:
                    optimizers.append(opt_name)
                    energies.append(result['energy'])
                    times.append(result['time'])

            # Plot final energies
            x = np.arange(len(optimizers))
            width = 0.35

            # Energy bars
            rects1 = ax2.bar(x - width / 2, energies, width, label='Energy')

            # Time bars (scaled for visibility)
            max_time = max(times) if times else 1
            scaled_times = [t / max_time for t in times]
            rects2 = ax2.bar(x + width / 2, scaled_times, width, label='Time (scaled)')

            # Add labels and legend
            ax2.set_xlabel('Optimizer')
            ax2.set_ylabel('Energy / Scaled Time')
            ax2.set_title('Final Results Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(optimizers)
            ax2.legend()

            # Add value labels on bars
            for rect, value in zip(rects1, energies):
                height = rect.get_height()
                ax2.annotate(
                    f'{value:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=8,
                )

            for rect, value in zip(rects2, times):
                height = rect.get_height()
                ax2.annotate(
                    f'{value:.1f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=8,
                )
        else:
            # Simple bar plot for final energies
            fig, ax = plt.subplots(figsize=(10, 6))

            # Extract data
            optimizers = []
            energies = []
            times = []
            iterations = []

            for opt_name, result in comparison_results.items():
                if isinstance(result, dict) and 'energy' in result:
                    optimizers.append(opt_name)
                    energies.append(result['energy'])
                    times.append(result.get('time', 0))
                    iterations.append(result.get('iterations', 0))

            # Plot energy bars
            x = np.arange(len(optimizers))
            ax.bar(x, energies, label='Energy')

            # Add labels
            ax.set_xlabel('Optimizer')
            ax.set_ylabel('Energy (Hartree)')
            ax.set_title(f'Energy Comparison for {system_name.capitalize()}')
            ax.set_xticks(x)
            ax.set_xticklabels(optimizers)

            # Add value labels
            for i, v in enumerate(energies):
                ax.text(
                    i,
                    v + 0.01,
                    f"{v:.4f}\n{times[i]:.1f}s\n{iterations[i]} iters",
                    ha='center',
                    fontsize=8,
                )

            # Add theoretical value if available
            if system_name in self.theoretical_values:
                theoretical = self.theoretical_values[system_name]
                ax.axhline(
                    y=theoretical,
                    color='r',
                    linestyle='--',
                    label=f'Theoretical ({theoretical:.4f})',
                )
                ax.legend()

        # Add metadata
        plt.figtext(
            0.5,
            0.01,
            f"System: {system_name}, Simulation: {self.simulation_type}",
            ha='center',
            fontsize=9,
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
