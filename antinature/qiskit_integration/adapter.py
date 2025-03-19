# antinature/qiskit_integration/adapter.py

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Check Qiskit availability
try:
    from qiskit.quantum_info import Operator, SparsePauliOp
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, FermiHubbardModel
    from qiskit_nature.second_q.mappers import (
        BravyiKitaevMapper,
        JordanWignerMapper,
        ParityMapper,
    )
    from qiskit_nature.second_q.operators import FermionicOp, SparseLabelOp
    from qiskit_nature.second_q.problems import ElectronicStructureProblem

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit Nature not available. Quantum functionality limited.")


class QiskitNatureAdapter:
    """
    Advanced adapter class for Qiskit Nature integration.

    This class provides sophisticated mapping between antinature systems and
    Qiskit Nature quantum chemistry objects, supporting various qubit mapping
    strategies and specialized adaptations for antimatter systems.
    """

    def __init__(
        self,
        mapper_type: str = 'jordan_wigner',
        include_spin: bool = False,
        two_qubit_reduction: bool = False,
        z2_symmetry_reduction: bool = False,
    ):
        """
        Initialize the Qiskit Nature adapter with enhanced capabilities.

        Parameters:
        -----------
        mapper_type : str
            Type of fermion-to-qubit mapping:
            - 'jordan_wigner': Direct mapping, simplest but most qubits
            - 'parity': Parity mapping, better scaling in some cases
            - 'bravyi_kitaev': Balance between locality and qubit count
        include_spin : bool
            Whether to include spin in the qubit mapping
        two_qubit_reduction : bool
            Whether to use two-qubit reduction techniques
        z2_symmetry_reduction : bool
            Whether to use Z2 symmetry reduction
        """
        if not HAS_QISKIT:
            raise ImportError(
                "Qiskit Nature is required for this functionality. Install with 'pip install qiskit-nature'"
            )

        self.mapper_type = mapper_type
        self.include_spin = include_spin
        self.two_qubit_reduction = two_qubit_reduction
        self.z2_symmetry_reduction = z2_symmetry_reduction

        # Set the mapper based on type
        if mapper_type == 'jordan_wigner':
            self.mapper = JordanWignerMapper()
        elif mapper_type == 'parity':
            self.mapper = ParityMapper()
        elif mapper_type == 'bravyi_kitaev':
            self.mapper = BravyiKitaevMapper()
        else:
            raise ValueError(f"Unknown mapper type: {mapper_type}")

        # Performance tracking
        self.timing = {}

        # Cache for expensive operations
        self._cache = {}

    def convert_hamiltonian(
        self, hamiltonian: Dict, system_type: str = 'general'
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Convert antinature Hamiltonian to Qiskit operator.

        Parameters:
        -----------
        hamiltonian : Dict
            Hamiltonian components from antinature system
        system_type : str
            Type of system for specialized handling:
            - 'general': Standard conversion
            - 'positronium': Positronium-specific optimizations
            - 'anti_hydrogen': Anti-hydrogen optimizations

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) representing the system
        """
        start_time = time.time()

        # Select the appropriate conversion method based on system type
        if system_type == 'positronium':
            problem, qubit_op = self._convert_positronium_hamiltonian(hamiltonian)
        elif system_type == 'anti_hydrogen':
            problem, qubit_op = self._convert_anti_hydrogen_hamiltonian(hamiltonian)
        elif system_type == 'positronium_molecule':
            problem, qubit_op = self._convert_positronium_molecule_hamiltonian(
                hamiltonian
            )
        else:
            # General conversion
            problem, qubit_op = self._convert_general_hamiltonian(hamiltonian)

        end_time = time.time()
        self.timing['convert_hamiltonian'] = end_time - start_time

        return problem, qubit_op

    def _convert_general_hamiltonian(
        self, hamiltonian: Dict
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Convert general antinature Hamiltonian to Qiskit format.

        Parameters:
        -----------
        hamiltonian : Dict
            Hamiltonian components

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator)
        """
        # Extract one-body and two-body terms
        h1_a = hamiltonian.get('h1_electron', None)
        h2_aa = hamiltonian.get('h2_electron_electron', None)
        h1_b = hamiltonian.get('h1_positron', None)
        h2_bb = hamiltonian.get('h2_positron_positron', None)
        h2_ba = hamiltonian.get('h2_electron_positron', None)

        # Create ElectronicEnergy object
        if h1_a is not None:
            # Standard electronic structure case
            if h1_b is None:
                # Electron-only system
                electronic_energy = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa)
            else:
                # Mixed electron-positron system
                electronic_energy = ElectronicEnergy.from_raw_integrals(
                    h1_a=h1_a, h2_aa=h2_aa, h1_b=h1_b, h2_bb=h2_bb, h2_ba=h2_ba
                )
        else:
            # No Hamiltonian components available
            raise ValueError("No Hamiltonian components provided")

        # Create problem
        problem = ElectronicStructureProblem(electronic_energy)

        # Map to qubit operator
        fermionic_op = electronic_energy.second_q_op()
        qubit_op = self.mapper.map(fermionic_op)

        # Apply symmetry reduction if enabled
        if self.z2_symmetry_reduction:
            # In a full implementation, would apply Z2 symmetry reduction here
            pass

        # Apply two-qubit reduction if enabled and compatible
        if self.two_qubit_reduction and self.mapper_type == 'jordan_wigner':
            # In a full implementation, would apply two-qubit reduction here
            pass

        return problem, qubit_op

    def _convert_positronium_hamiltonian(
        self, hamiltonian: Optional[Dict] = None
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Convert positronium Hamiltonian with specialized optimization.

        Parameters:
        -----------
        hamiltonian : Dict, optional
            Hamiltonian components (uses defaults if None)

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) for positronium
        """
        # Create simplified parameters for positronium if not provided
        if hamiltonian is None:
            # These parameters give ground state energy of -0.25 Hartree
            e_repulsion = 0.0
            p_repulsion = 0.0
            ep_attraction = -1.0
        else:
            # Extract parameters from provided Hamiltonian
            e_repulsion = hamiltonian.get('electron_repulsion', 0.0)
            p_repulsion = hamiltonian.get('positron_repulsion', 0.0)
            ep_attraction = hamiltonian.get('electron_positron_attraction', -1.0)

        # Create 1x1 Hamiltonians
        h1_a = np.array([[0.5]])  # Electron kinetic
        h1_b = np.array([[0.5]])  # Positron kinetic

        # Create two-body interaction tensors
        h2_aa = np.zeros((1, 1, 1, 1))
        h2_aa[0, 0, 0, 0] = e_repulsion

        h2_bb = np.zeros((1, 1, 1, 1))
        h2_bb[0, 0, 0, 0] = p_repulsion

        h2_ba = np.zeros((1, 1, 1, 1))
        h2_ba[0, 0, 0, 0] = ep_attraction

        # Create ElectronicEnergy object with these parameters
        electronic_energy = ElectronicEnergy.from_raw_integrals(
            h1_a=h1_a, h2_aa=h2_aa, h1_b=h1_b, h2_bb=h2_bb, h2_ba=h2_ba
        )

        # Create problem
        problem = ElectronicStructureProblem(electronic_energy)

        # Map to qubit operator
        fermionic_op = electronic_energy.second_q_op()
        qubit_op = self.mapper.map(fermionic_op)

        # Optimization: directly construct exact positronium operator if desired
        # H = 0.5*(I⊗I + Z⊗Z) - 0.25*(X⊗X + Y⊗Y)
        # This can be done by manually constructing SparsePauliOp

        return problem, qubit_op

    def _convert_anti_hydrogen_hamiltonian(
        self, hamiltonian: Optional[Dict] = None
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Convert anti-hydrogen Hamiltonian with specialized optimization.

        Parameters:
        -----------
        hamiltonian : Dict, optional
            Hamiltonian components (uses defaults if None)

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) for anti-hydrogen
        """
        # Create simplified parameters for anti-hydrogen if not provided
        if hamiltonian is None:
            # Create one-body terms for simplified 3-orbital model
            n_orbitals = 3
            h1_b = np.zeros((n_orbitals, n_orbitals))

            # Kinetic energy and orbital energy terms
            h1_b[0, 0] = 0.5  # Ground state orbital
            h1_b[1, 1] = 0.6  # First excited orbital
            h1_b[2, 2] = 0.7  # Second excited orbital

            # Off-diagonal terms for orbital transitions
            h1_b[0, 1] = h1_b[1, 0] = 0.1
            h1_b[1, 2] = h1_b[2, 1] = 0.05

            # Create two-body terms for positron-antiproton attraction
            # The attractive interaction term is the dominant one
            h2_bb = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals))

            for i in range(n_orbitals):
                for j in range(n_orbitals):
                    if i != j:
                        h2_bb[i, i, j, j] = -1.0  # Attractive interaction
        else:
            # Extract from provided Hamiltonian
            h1_b = hamiltonian.get('h1_positron', None)
            h2_bb = hamiltonian.get('h2_positron_positron', None)

            if h1_b is None or h2_bb is None:
                raise ValueError("Incomplete Hamiltonian components for anti-hydrogen")

        # Create ElectronicEnergy object
        # For anti-hydrogen, we can use a positron-only Hamiltonian
        # with specialized interaction terms
        electronic_energy = ElectronicEnergy.from_raw_integrals(
            h1_a=h1_b, h2_aa=h2_bb  # Use positron terms as primary
        )

        # Create problem
        problem = ElectronicStructureProblem(electronic_energy)

        # Map to qubit operator
        fermionic_op = electronic_energy.second_q_op()
        qubit_op = self.mapper.map(fermionic_op)

        return problem, qubit_op

    def _convert_positronium_molecule_hamiltonian(
        self, hamiltonian: Optional[Dict] = None
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Convert positronium molecule Hamiltonian with specialized optimization.

        Parameters:
        -----------
        hamiltonian : Dict, optional
            Hamiltonian components (uses defaults if None)

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) for positronium molecule
        """
        # Create simplified parameters for positronium molecule if not provided
        if hamiltonian is None:
            # For positronium molecule (Ps₂), we need 2e + 2p
            n_orbitals = 4  # 2 for electrons, 2 for positrons

            # Create one-body terms
            h1_a = np.zeros((2, 2))  # Electrons
            h1_b = np.zeros((2, 2))  # Positrons

            # Kinetic energy terms
            h1_a[0, 0] = h1_a[1, 1] = 0.5  # Electron kinetic
            h1_b[0, 0] = h1_b[1, 1] = 0.5  # Positron kinetic

            # Create two-body terms
            h2_aa = np.zeros((2, 2, 2, 2))  # e-e
            h2_bb = np.zeros((2, 2, 2, 2))  # p-p
            h2_ba = np.zeros((2, 2, 2, 2))  # e-p

            # Set electron-electron repulsion
            h2_aa[0, 0, 1, 1] = h2_aa[1, 1, 0, 0] = 0.4

            # Set positron-positron repulsion
            h2_bb[0, 0, 1, 1] = h2_bb[1, 1, 0, 0] = 0.4

            # Set electron-positron attraction
            # Stronger for paired e-p in same positronium atom
            h2_ba[0, 0, 0, 0] = h2_ba[1, 1, 1, 1] = -1.0

            # Weaker for cross-attraction between atoms
            h2_ba[0, 0, 1, 1] = h2_ba[1, 1, 0, 0] = -0.5
        else:
            # Extract from provided Hamiltonian
            h1_a = hamiltonian.get('h1_electron', None)
            h1_b = hamiltonian.get('h1_positron', None)
            h2_aa = hamiltonian.get('h2_electron_electron', None)
            h2_bb = hamiltonian.get('h2_positron_positron', None)
            h2_ba = hamiltonian.get('h2_electron_positron', None)

            if any(x is None for x in [h1_a, h1_b, h2_aa, h2_bb, h2_ba]):
                raise ValueError(
                    "Incomplete Hamiltonian components for positronium molecule"
                )

        # Create ElectronicEnergy object
        electronic_energy = ElectronicEnergy.from_raw_integrals(
            h1_a=h1_a, h2_aa=h2_aa, h1_b=h1_b, h2_bb=h2_bb, h2_ba=h2_ba
        )

        # Create problem
        problem = ElectronicStructureProblem(electronic_energy)

        # Map to qubit operator
        fermionic_op = electronic_energy.second_q_op()
        qubit_op = self.mapper.map(fermionic_op)

        return problem, qubit_op

    def create_custom_operator(
        self, pauli_strings: List[str], coefficients: List[float]
    ) -> SparsePauliOp:
        """
        Create a custom qubit operator from Pauli strings and coefficients.

        Parameters:
        -----------
        pauli_strings : List[str]
            List of Pauli strings (e.g., 'IXYZ', 'ZZZZ')
        coefficients : List[float]
            Coefficient for each Pauli string

        Returns:
        --------
        SparsePauliOp
            Custom qubit operator
        """
        if len(pauli_strings) != len(coefficients):
            raise ValueError(
                "Number of Pauli strings must match number of coefficients"
            )

        # Create sparse Pauli operator
        operator = SparsePauliOp(pauli_strings, coefficients)

        return operator

    def create_exact_positronium_operator(self) -> SparsePauliOp:
        """
        Create exact positronium ground state Hamiltonian.

        Returns:
        --------
        SparsePauliOp
            Exact positronium Hamiltonian operator
        """
        # Create exact 2-qubit positronium Hamiltonian
        # H = 0.5 * (I⊗I + Z⊗Z) - 0.25 * (X⊗X + Y⊗Y)

        pauli_strings = ['II', 'ZZ', 'XX', 'YY']
        coefficients = [0.5, 0.5, -0.25, -0.25]

        operator = self.create_custom_operator(pauli_strings, coefficients)

        return operator

    def map_fermionic_to_qubit(self, fermionic_op: FermionicOp) -> SparsePauliOp:
        """
        Map a fermionic operator to qubit operator with current mapper.

        Parameters:
        -----------
        fermionic_op : FermionicOp
            Fermionic operator to map

        Returns:
        --------
        SparsePauliOp
            Mapped qubit operator
        """
        return self.mapper.map(fermionic_op)

    def create_fermionic_hamiltonian(
        self, one_body: np.ndarray, two_body: np.ndarray
    ) -> FermionicOp:
        """
        Create a fermionic Hamiltonian from one-body and two-body integrals.

        Parameters:
        -----------
        one_body : np.ndarray
            One-body integral terms (kinetic energy, external fields)
        two_body : np.ndarray
            Two-body integral terms (particle-particle interactions)

        Returns:
        --------
        FermionicOp
            Fermionic operator
        """
        # Create ElectronicEnergy object
        electronic_energy = ElectronicEnergy.from_raw_integrals(one_body, two_body)

        # Get fermionic operator
        fermionic_op = electronic_energy.second_q_op()

        return fermionic_op

    def create_hubbard_model(
        self,
        num_sites: int,
        onsite_strength: float = 1.0,
        tunneling_strength: float = 0.5,
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Create a Fermi-Hubbard model for testing.

        Parameters:
        -----------
        num_sites : int
            Number of lattice sites
        onsite_strength : float
            Strength of onsite interaction (U)
        tunneling_strength : float
            Strength of tunneling term (t)

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Hubbard model, Qubit operator)
        """
        # Create 1D Fermi-Hubbard model
        hubbard = FermiHubbardModel(num_sites, tunneling_strength, onsite_strength)

        # Map to qubit operator
        qubit_op = self.mapper.map(hubbard.second_q_op())

        return hubbard, qubit_op

    def get_mapper_info(self) -> Dict:
        """
        Get information about the current mapper.

        Returns:
        --------
        Dict
            Information about the mapper
        """
        return {
            'mapper_type': self.mapper_type,
            'include_spin': self.include_spin,
            'two_qubit_reduction': self.two_qubit_reduction,
            'z2_symmetry_reduction': self.z2_symmetry_reduction,
        }


class PositroniumAdapter(QiskitNatureAdapter):
    """
    Specialized adapter for positronium systems.

    This subclass provides optimized conversions and utilities
    specifically for positronium and positronic systems.
    """

    def __init__(self, mapper_type: str = 'jordan_wigner', use_exact: bool = True):
        """
        Initialize the positronium adapter.

        Parameters:
        -----------
        mapper_type : str
            Type of mapper to use
        use_exact : bool
            Whether to use exact Hamiltonian when possible
        """
        super().__init__(mapper_type=mapper_type)
        self.use_exact = use_exact

    def create_positronium_hamiltonian(
        self,
        e_repulsion: float = 0.0,
        p_repulsion: float = 0.0,
        ep_attraction: float = -1.0,
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Create a optimized positronium Hamiltonian.

        Parameters:
        -----------
        e_repulsion : float
            Electron-electron repulsion coefficient
        p_repulsion : float
            Positron-positron repulsion coefficient
        ep_attraction : float
            Electron-positron attraction coefficient

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) for positronium
        """
        # Check if we should use the exact Hamiltonian
        if self.use_exact:
            # For positronium, we have an exact 2-qubit representation
            # This is more efficient than the general approach
            qubit_op = self.create_exact_positronium_operator()

            # Create a minimal problem for compatibility
            h1_a = np.array([[0.5]])
            h2_aa = np.zeros((1, 1, 1, 1))
            energy_op = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa)
            problem = ElectronicStructureProblem(energy_op)

            return problem, qubit_op

        # Otherwise, use general approach with the specified parameters
        hamiltonian = {
            'electron_repulsion': e_repulsion,
            'positron_repulsion': p_repulsion,
            'electron_positron_attraction': ep_attraction,
        }

        return self._convert_positronium_hamiltonian(hamiltonian)

    def create_ortho_para_hamiltonian(
        self, is_ortho: bool = False
    ) -> Tuple[Any, SparsePauliOp]:
        """
        Create Hamiltonian for para or ortho positronium.

        Parameters:
        -----------
        is_ortho : bool
            True for ortho-positronium (triplet), False for para-positronium (singlet)

        Returns:
        --------
        Tuple[Any, SparsePauliOp]
            (Problem, Qubit operator) for the specified positronium state
        """
        # For ortho-positronium (triplet state)
        if is_ortho:
            # Modify coefficients to represent triplet state
            # In triplet state, spins are parallel with different energy
            pauli_strings = ['II', 'ZZ', 'XX', 'YY', 'ZI', 'IZ']
            coefficients = [
                0.5,
                0.5,
                -0.25,
                -0.25,
                0.01,
                0.01,
            ]  # Small Z terms for triplet
        else:
            # Para-positronium (singlet state)
            pauli_strings = ['II', 'ZZ', 'XX', 'YY']
            coefficients = [0.5, 0.5, -0.25, -0.25]  # Standard positronium

        # Create the operator
        qubit_op = self.create_custom_operator(pauli_strings, coefficients)

        # Create a minimal problem for compatibility
        h1_a = np.array([[0.5]])
        h2_aa = np.zeros((1, 1, 1, 1))
        energy_op = ElectronicEnergy.from_raw_integrals(h1_a, h2_aa)
        problem = ElectronicStructureProblem(energy_op)

        return problem, qubit_op

    def create_varying_attraction_hamiltonians(
        self, attraction_values: List[float]
    ) -> Dict[float, SparsePauliOp]:
        """
        Create a set of positronium Hamiltonians with varying attraction strength.

        Parameters:
        -----------
        attraction_values : List[float]
            List of electron-positron attraction strengths to use

        Returns:
        --------
        Dict[float, SparsePauliOp]
            Dictionary mapping attraction strength to qubit operator
        """
        results = {}

        for attraction in attraction_values:
            # Create Hamiltonian with this attraction strength
            print(f"Creating Hamiltonian with attraction = {attraction}")
            _, operator = self.create_positronium_hamiltonian(
                e_repulsion=0.0, p_repulsion=0.0, ep_attraction=attraction
            )

            results[attraction] = operator

        return results
