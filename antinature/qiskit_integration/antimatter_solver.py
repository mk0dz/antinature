# antinature/qiskit_integration/antinature_solver.py

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import qiskit if available
try:
    from qiskit.primitives import Estimator
    from qiskit_algorithms import VQE, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP, SPSA

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit Algorithms not available. Limited functionality.")

# Import our quantum systems with relative import
from .systems import AntinatureQuantumSystems
from .vqe_solver import AntinatureVQESolver


class AntinatureQuantumSolver:
    """
    Quantum solver for antinature systems using Qiskit.

    This class serves as the main interface for solving antinature systems
    using quantum computing approaches, supporting VQE and exact solvers.
    """

    def __init__(
        self,
        use_exact_solver: bool = False,
        optimizer_name: str = 'COBYLA',
        max_iterations: int = 500,
        shots: int = 2048,
        mapper_type: str = 'jordan_wigner',
    ):
        """
        Initialize the antinature quantum solver.

        Parameters:
        -----------
        use_exact_solver : bool
            Whether to use NumPyMinimumEigensolver (exact) instead of VQE
        optimizer_name : str
            Name of VQE optimizer to use ('COBYLA', 'SPSA', 'L_BFGS_B', 'SLSQP')
        max_iterations : int
            Maximum number of iterations for VQE
        shots : int
            Number of shots for VQE circuit evaluation
        mapper_type : str
            Fermion-to-qubit mapping ('jordan_wigner', 'parity', 'bravyi_kitaev')
        """
        if not HAS_QISKIT:
            raise ImportError("Qiskit is required for this solver")

        self.use_exact_solver = use_exact_solver
        self.optimizer_name = optimizer_name
        self.max_iterations = max_iterations
        self.shots = shots
        self.mapper_type = mapper_type

        # Initialize the quantum systems object
        self.systems = AntinatureQuantumSystems(mapper_type=mapper_type)

        # Initialize VQE solver if needed
        if not use_exact_solver:
            self.vqe_solver = AntinatureVQESolver(
                optimizer_name=optimizer_name,
                max_iterations=max_iterations,
                shots=shots,
            )

        # Reference values for validation
        self.theoretical_values = {
            'positronium': -0.25,  # Hartree
            'anti_hydrogen': -0.5,  # Hartree
            'positronium_molecule': -0.52,  # Hartree (approximate)
            'anti_helium': -2.9,  # Hartree (approximate)
        }

    def solve_positronium(
        self,
        ansatz_type: str = 'specialized',
        reps: int = 3,
        apply_correction: bool = True,
    ) -> Dict:
        """
        Solve the quantum problem for positronium.

        Parameters:
        -----------
        ansatz_type : str
            Type of ansatz to use ('specialized', 'hardware_efficient')
        reps : int
            Number of repetition layers in the ansatz
        apply_correction : bool
            Whether to apply theoretical corrections to results

        Returns:
        --------
        Dict
            Results of the quantum simulation
        """
        # Get Hamiltonian for positronium (using dedicated method)
        hamiltonian, circuit = self.systems.positronium()

        # Prepare result dictionary with system info
        result = {
            'system': 'positronium',
            'n_qubits': hamiltonian.num_qubits,
            'theoretical_value': self.theoretical_values['positronium'],
        }

        if self.use_exact_solver:
            # Use NumPyMinimumEigensolver for exact solution
            solver = NumPyMinimumEigensolver()
            calc_result = solver.compute_minimum_eigenvalue(hamiltonian)

            # Extract and add results
            result['energy'] = calc_result.eigenvalue.real
            # No eigenvector in the result for some versions of qiskit
            if hasattr(calc_result, 'eigenvector'):
                result['eigenvector'] = calc_result.eigenvector
            result['exact'] = True
        else:
            # Use VQE for approximate solution
            vqe_result = self.vqe_solver.solve_system(
                system_name='positronium',
                qubit_operator=hamiltonian,
                ansatz_type=ansatz_type,
                reps=reps,
                apply_correction=apply_correction,
            )

            # Merge results
            result.update(vqe_result)
            result['exact'] = False

        return result

    def solve_anti_hydrogen(
        self,
        ansatz_type: str = 'specialized',
        reps: int = 3,
        apply_correction: bool = True,
    ) -> Dict:
        """
        Solve the quantum problem for anti-hydrogen.

        Parameters:
        -----------
        ansatz_type : str
            Type of ansatz to use ('specialized', 'hardware_efficient')
        reps : int
            Number of repetition layers in the ansatz
        apply_correction : bool
            Whether to apply theoretical corrections to results

        Returns:
        --------
        Dict
            Results of the quantum simulation
        """
        # Get Hamiltonian for anti-hydrogen
        hamiltonian, circuit = self.systems.anti_hydrogen()

        # Prepare result dictionary with system info
        result = {
            'system': 'anti_hydrogen',
            'n_qubits': hamiltonian.num_qubits,
            'theoretical_value': self.theoretical_values['anti_hydrogen'],
        }

        if self.use_exact_solver:
            # Use NumPyMinimumEigensolver for exact solution
            solver = NumPyMinimumEigensolver()
            calc_result = solver.compute_minimum_eigenvalue(hamiltonian)

            # Extract and add results
            result['energy'] = calc_result.eigenvalue.real
            # No eigenvector in the result for some versions of qiskit
            if hasattr(calc_result, 'eigenvector'):
                result['eigenvector'] = calc_result.eigenvector
            result['exact'] = True
        else:
            # Use VQE for approximate solution
            vqe_result = self.vqe_solver.solve_system(
                system_name='anti_hydrogen',
                qubit_operator=hamiltonian,
                ansatz_type=ansatz_type,
                reps=reps,
                apply_correction=apply_correction,
            )

            # Merge results
            result.update(vqe_result)
            result['exact'] = False

        return result

    def solve_positronium_molecule(
        self,
        ansatz_type: str = 'specialized',
        reps: int = 3,
        apply_correction: bool = True,
    ) -> Dict:
        """
        Solve the quantum problem for positronium molecule.

        Parameters:
        -----------
        ansatz_type : str
            Type of ansatz to use ('specialized', 'hardware_efficient')
        reps : int
            Number of repetition layers in the ansatz
        apply_correction : bool
            Whether to apply theoretical corrections to results

        Returns:
        --------
        Dict
            Results of the quantum simulation
        """
        # Get Hamiltonian for positronium molecule
        hamiltonian, circuit = self.systems.positronium_molecule()

        # Prepare result dictionary with system info
        result = {
            'system': 'positronium_molecule',
            'n_qubits': hamiltonian.num_qubits,
            'theoretical_value': self.theoretical_values['positronium_molecule'],
        }

        if self.use_exact_solver:
            # Use NumPyMinimumEigensolver for exact solution
            solver = NumPyMinimumEigensolver()
            calc_result = solver.compute_minimum_eigenvalue(hamiltonian)

            # Extract and add results
            result['energy'] = calc_result.eigenvalue.real
            # No eigenvector in the result for some versions of qiskit
            if hasattr(calc_result, 'eigenvector'):
                result['eigenvector'] = calc_result.eigenvector
            result['exact'] = True
        else:
            # Use VQE for approximate solution
            vqe_result = self.vqe_solver.solve_system(
                system_name='positronium_molecule',
                qubit_operator=hamiltonian,
                ansatz_type=ansatz_type,
                reps=reps,
                apply_correction=apply_correction,
            )

            # Merge results
            result.update(vqe_result)
            result['exact'] = False

        return result

    def solve_custom_system(
        self,
        hamiltonian,
        circuit=None,
        system_name: str = 'custom',
        ansatz_type: str = 'hardware_efficient',
        reps: int = 3,
        apply_correction: bool = False,
    ) -> Dict:
        """
        Solve a custom antinature system with user-provided Hamiltonian.

        Parameters:
        -----------
        hamiltonian : Operator
            Qubit operator representing the system Hamiltonian
        circuit : QuantumCircuit, optional
            Optional quantum circuit to use as ansatz
        system_name : str
            Name of the system (for reference)
        ansatz_type : str
            Type of ansatz to use if circuit not provided
        reps : int
            Number of repetition layers in the ansatz
        apply_correction : bool
            Whether to apply theoretical corrections

        Returns:
        --------
        Dict
            Results of the quantum simulation
        """
        # Prepare result dictionary with system info
        result = {
            'system': system_name,
            'n_qubits': hamiltonian.num_qubits,
            'theoretical_value': self.theoretical_values.get(system_name),
        }

        if self.use_exact_solver:
            # Use NumPyMinimumEigensolver for exact solution
            solver = NumPyMinimumEigensolver()
            calc_result = solver.compute_minimum_eigenvalue(hamiltonian)

            # Extract and add results
            result['energy'] = calc_result.eigenvalue.real
            # No eigenvector in the result for some versions of qiskit
            if hasattr(calc_result, 'eigenvector'):
                result['eigenvector'] = calc_result.eigenvector
            result['exact'] = True
        else:
            # Use VQE for approximate solution
            vqe_result = self.vqe_solver.solve_system(
                system_name=system_name,
                qubit_operator=hamiltonian,
                ansatz_type=ansatz_type,
                reps=reps,
                apply_correction=apply_correction,
            )

            # Merge results
            result.update(vqe_result)
            result['exact'] = False

        return result

    def compare_methods(
        self,
        system_name: str = 'positronium',
        ansatz_types: List[str] = ['specialized', 'hardware_efficient'],
        exact: bool = True,
    ) -> Dict[str, Dict]:
        """
        Compare different solution methods for a given antinature system.

        Parameters:
        -----------
        system_name : str
            Name of the system to solve
        ansatz_types : List[str]
            List of ansatz types to compare
        exact : bool
            Whether to include exact solution in comparison

        Returns:
        --------
        Dict[str, Dict]
            Dictionary of results from different methods
        """
        results = {}

        # Solve with each ansatz type
        for ansatz_type in ansatz_types:
            if system_name == 'positronium':
                result = self.solve_positronium(ansatz_type=ansatz_type)
            elif system_name == 'anti_hydrogen':
                result = self.solve_anti_hydrogen(ansatz_type=ansatz_type)
            elif system_name == 'positronium_molecule':
                result = self.solve_positronium_molecule(ansatz_type=ansatz_type)
            else:
                raise ValueError(f"Unknown system: {system_name}")

            results[f'vqe_{ansatz_type}'] = result

        # Include exact solution if requested
        if exact:
            # Temporarily switch to exact solver
            original_setting = self.use_exact_solver
            self.use_exact_solver = True

            if system_name == 'positronium':
                result = self.solve_positronium()
            elif system_name == 'anti_hydrogen':
                result = self.solve_anti_hydrogen()
            elif system_name == 'positronium_molecule':
                result = self.solve_positronium_molecule()
            else:
                raise ValueError(f"Unknown system: {system_name}")

            results['exact'] = result

            # Restore original setting
            self.use_exact_solver = original_setting

        return results

    def visualize_results(self, results: Dict, filename: Optional[str] = None) -> None:
        """
        Visualize the results of quantum simulations.

        Parameters:
        -----------
        results : Dict
            Results dictionary from a solve method
        filename : str, optional
            Filename to save the visualization (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Extract relevant data
            system_name = results.get('system', 'Unknown')
            energy = results.get('energy', 0.0)
            theoretical = results.get('theoretical_value', None)

            # Create bar chart
            bars = plt.bar(
                ['Quantum', 'Theoretical'],
                [energy, theoretical] if theoretical is not None else [energy, 0],
            )

            # Set colors
            bars[0].set_color('blue')
            if theoretical is not None:
                bars[1].set_color('green')

                # Add error line
                error = abs(energy - theoretical)
                plt.plot(
                    [0, 1], [energy, theoretical], 'r--', label=f'Error: {error:.4f}'
                )

            # Add labels and title
            plt.ylabel('Energy (Hartree)')
            plt.title(f'Energy of {system_name.replace("_", " ").title()}')
            plt.ylim(
                min(energy, theoretical if theoretical is not None else 0) * 1.2, 0
            )

            # Add text annotations
            was_corrected = results.get('was_corrected', False)
            if was_corrected:
                plt.text(
                    0,
                    energy * 1.05,
                    f"Corrected\n{results.get('raw_energy', 0):.4f} → {energy:.4f}",
                )
            else:
                plt.text(0, energy * 1.05, f"{energy:.4f}")

            if theoretical is not None:
                plt.text(1, theoretical * 1.05, f"{theoretical:.4f}")

            # Add method info
            method_info = (
                'Exact'
                if results.get('exact', False)
                else f"VQE ({results.get('ansatz_type', 'unknown')})"
            )
            iterations = results.get('iterations', None)
            if iterations is not None:
                method_info += f"\nIterations: {iterations}"

            plt.figtext(0.15, 0.02, method_info)

            # Add error percentage if theoretical value exists
            if theoretical is not None:
                error_percent = abs(energy - theoretical) / abs(theoretical) * 100
                plt.figtext(0.7, 0.02, f"Error: {error_percent:.2f}%")

            plt.tight_layout()

            # Save or show
            if filename:
                plt.savefig(filename)
                print(f"Visualization saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib is required for visualization")
            print(f"Results: {system_name} energy = {energy:.6f} Hartree")
            if theoretical is not None:
                print(f"Theoretical value: {theoretical:.6f} Hartree")
                print(
                    f"Error: {abs(energy - theoretical):.6f} Hartree ({error_percent:.2f}%)"
                )

    def compare_visualization(
        self, results: Dict[str, Dict], filename: Optional[str] = None
    ) -> None:
        """
        Visualize comparison between different methods.

        Parameters:
        -----------
        results : Dict[str, Dict]
            Dictionary of results from different methods
        filename : str, optional
            Filename to save the visualization (if None, display only)
        """
        try:
            import matplotlib.pyplot as plt

            # Create a new figure
            plt.figure(figsize=(10, 6))

            # Extract system name from the first result
            first_key = list(results.keys())[0]
            system_name = results[first_key].get('system', 'Unknown')
            theoretical = results[first_key].get('theoretical_value', None)

            # Prepare data for bar chart
            method_names = list(results.keys())
            energies = [results[method].get('energy', 0.0) for method in method_names]

            # Create bar chart
            bars = plt.bar(method_names, energies)

            # Set colors
            for i, bar in enumerate(bars):
                bar.set_color(['blue', 'orange', 'green', 'red'][i % 4])

            # Add theoretical line if available
            if theoretical is not None:
                plt.axhline(
                    y=theoretical,
                    color='r',
                    linestyle='--',
                    label=f'Theoretical: {theoretical:.4f}',
                )

            # Add labels and title
            plt.ylabel('Energy (Hartree)')
            plt.title(f'Energy Comparison for {system_name.replace("_", " ").title()}')

            # Adjust y-axis limits to show all bars clearly
            min_energy = min(
                energies + ([theoretical] if theoretical is not None else [])
            )
            plt.ylim(min_energy * 1.2, 0)

            # Add text annotations
            for i, energy in enumerate(energies):
                plt.text(i, energy * 1.05, f"{energy:.4f}")

            # Add legend
            plt.legend()

            plt.tight_layout()

            # Save or show
            if filename:
                plt.savefig(filename)
                print(f"Visualization saved to {filename}")
            else:
                plt.show()

        except ImportError:
            print("Matplotlib is required for visualization")
            print(f"Results comparison for {system_name}:")
            for method, result in results.items():
                energy = result.get('energy', 0.0)
                print(f"  {method}: {energy:.6f} Hartree")
            if theoretical is not None:
                print(f"Theoretical value: {theoretical:.6f} Hartree")
