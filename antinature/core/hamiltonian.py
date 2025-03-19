# antinature/core/hamiltonian.py

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class AntinatureHamiltonian:
    """
    Optimized Hamiltonian builder for antinature molecular systems.

    This class constructs all required Hamiltonian matrices for mixed
    electron-positron systems with enhanced performance, better vectorization,
    and support for all types of interactions.
    """

    def __init__(
        self,
        molecular_data,
        basis_set,
        integral_engine,
        include_annihilation: bool = True,
        include_relativistic: bool = False,
        use_symmetry: bool = True,
        use_incremental_build: bool = True,
    ):
        """
        Initialize an antinature Hamiltonian.

        Parameters:
        -----------
        molecular_data : MolecularData
            Molecular structure information
        basis_set : MixedMatterBasis
            Basis set for the calculation
        integral_engine : AntinatureIntegralEngine
            Engine for integral computation
        include_annihilation : bool
            Whether to include annihilation terms
        include_relativistic : bool
            Whether to include relativistic corrections
        use_symmetry : bool
            Whether to use matrix symmetry for optimization
        use_incremental_build : bool
            Whether to build matrices incrementally (for large systems)
        """
        self.molecular_data = molecular_data
        self.basis_set = basis_set
        self.integral_engine = integral_engine

        # Set the integral engine on the basis set
        self.basis_set.set_integral_engine(self.integral_engine)

        self.include_annihilation = include_annihilation
        self.include_relativistic = include_relativistic
        self.use_symmetry = use_symmetry
        self.use_incremental_build = use_incremental_build

        # Extract key data
        self.nuclei = molecular_data.nuclei if hasattr(molecular_data, 'nuclei') else []
        self.n_electrons = (
            molecular_data.n_electrons if hasattr(molecular_data, 'n_electrons') else 0
        )
        self.n_positrons = (
            molecular_data.n_positrons if hasattr(molecular_data, 'n_positrons') else 0
        )

        # Initialize storage for computed matrices
        self.matrices = {}

        # For performance tracking
        self.timings = {}

    def build_hamiltonian(self) -> Dict:
        """
        Construct the complete Hamiltonian for the antinature system.

        This method orchestrates the construction of all necessary matrices.

        Returns:
        --------
        Dict
            Dictionary containing all Hamiltonian components
        """
        start_time = time.time()

        # 1. Build overlap matrix
        self.build_overlap_matrix()

        # 2. Build kinetic and nuclear attraction matrices
        self.build_one_body_matrices()

        # 3. Combine one-body terms into core Hamiltonian
        self.build_core_hamiltonian()

        # 4. Compute two-electron integrals for Coulomb interactions
        if self.n_electrons > 1:
            self.compute_electron_repulsion_integrals()

        # 5. Handle positron-specific components
        if self.n_positrons > 0:
            # Compute positron repulsion
            if self.n_positrons > 1:
                self.compute_positron_repulsion_integrals()

            # Compute electron-positron attraction
            if self.n_electrons > 0:
                self.compute_electron_positron_attraction()

        # 6. Include annihilation terms if requested
        if self.include_annihilation and self.n_electrons > 0 and self.n_positrons > 0:
            self.compute_annihilation_operator()

        # 7. Apply relativistic corrections if requested
        if self.include_relativistic:
            self.build_relativistic_corrections()

        # Track total time
        end_time = time.time()
        self.timings['total'] = end_time - start_time

        return self.matrices

    def build_overlap_matrix(self) -> np.ndarray:
        """
        Construct the overlap matrix S efficiently.

        The overlap matrix S_μν = <μ|ν> represents the overlap between basis functions.

        Returns:
        --------
        np.ndarray
            Overlap matrix for the combined basis
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis
        n_basis = n_e_basis + n_p_basis

        # Initialize overlap matrix
        S = np.zeros((n_basis, n_basis))

        # Calculate overlap between electron basis functions
        for i in range(n_e_basis):
            for j in range(i + 1):  # Use symmetry for efficiency
                basis_i = self.basis_set.get_basis_function(i)
                basis_j = self.basis_set.get_basis_function(j)

                # Calculate overlap using integral engine
                S[i, j] = self.integral_engine.overlap_integral(basis_i, basis_j)

                # Use symmetry
                if i != j:
                    S[j, i] = S[i, j]

        # Calculate overlap between positron basis functions
        for i in range(n_p_basis):
            for j in range(i + 1):  # Use symmetry
                ii = i + n_e_basis
                jj = j + n_e_basis

                basis_i = self.basis_set.get_basis_function(ii)
                basis_j = self.basis_set.get_basis_function(jj)

                S[ii, jj] = self.integral_engine.overlap_integral(basis_i, basis_j)

                # Use symmetry
                if i != j:
                    S[jj, ii] = S[ii, jj]

        # Electron-positron overlap (typically zero unless special basis used)
        # We'll leave it as zeros by default, but could calculate if needed

        # Store in matrices dictionary
        self.matrices['overlap'] = S

        # Track performance
        end_time = time.time()
        self.timings['overlap'] = end_time - start_time

        return S

    def build_one_body_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build kinetic energy and nuclear attraction matrices.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (kinetic_matrix, nuclear_attraction_matrix)
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis
        n_basis = n_e_basis + n_p_basis

        # Initialize matrices
        T = np.zeros((n_basis, n_basis))  # Kinetic energy
        V_nuc_e = np.zeros((n_e_basis, n_e_basis))  # Nuclear attraction (electrons)
        V_nuc_p = (
            np.zeros((n_p_basis, n_p_basis)) if n_p_basis > 0 else None
        )  # Nuclear attraction (positrons)

        # Calculate kinetic energy elements
        for i in range(n_basis):
            for j in range(i + 1):  # Use symmetry
                basis_i = self.basis_set.get_basis_function(i)
                basis_j = self.basis_set.get_basis_function(j)

                # Calculate kinetic energy integral
                T[i, j] = self.integral_engine.kinetic_integral(basis_i, basis_j)

                # Use symmetry
                if i != j:
                    T[j, i] = T[i, j]

        # Calculate nuclear attraction for electrons (attractive)
        for i in range(n_e_basis):
            for j in range(i + 1):  # Use symmetry
                basis_i = self.basis_set.get_basis_function(i)
                basis_j = self.basis_set.get_basis_function(j)

                # Sum over all nuclei
                v_sum = 0.0
                for atom, charge, pos in self.nuclei:
                    # Nuclear attraction integral (negative for attraction)
                    v_sum -= charge * self.integral_engine.nuclear_attraction_integral(
                        basis_i, basis_j, pos
                    )

                V_nuc_e[i, j] = v_sum

                # Use symmetry
                if i != j:
                    V_nuc_e[j, i] = V_nuc_e[i, j]

        # Calculate nuclear attraction for positrons (repulsive)
        if n_p_basis > 0:
            for i in range(n_p_basis):
                for j in range(i + 1):  # Use symmetry
                    basis_i = self.basis_set.get_basis_function(i + n_e_basis)
                    basis_j = self.basis_set.get_basis_function(j + n_e_basis)

                    # Sum over all nuclei
                    v_sum = 0.0
                    for atom, charge, pos in self.nuclei:
                        # Nuclear repulsion integral (positive for repulsion)
                        v_sum += (
                            charge
                            * self.integral_engine.nuclear_attraction_integral(
                                basis_i, basis_j, pos
                            )
                        )

                    V_nuc_p[i, j] = v_sum

                    # Use symmetry
                    if i != j:
                        V_nuc_p[j, i] = V_nuc_p[i, j]

        # Store in matrices dictionary
        self.matrices['kinetic'] = T
        self.matrices['kinetic_e'] = T[:n_e_basis, :n_e_basis]
        if n_p_basis > 0:
            self.matrices['kinetic_p'] = T[n_e_basis:, n_e_basis:]

        self.matrices['nuclear_attraction_e'] = V_nuc_e
        if n_p_basis > 0:
            self.matrices['nuclear_attraction_p'] = V_nuc_p

        # Track performance
        end_time = time.time()
        self.timings['one_body'] = end_time - start_time

        return T, (V_nuc_e, V_nuc_p)

    def build_core_hamiltonian(self) -> Dict:
        """
        Construct the one-electron (core) Hamiltonian matrices.

        The core Hamiltonian represents kinetic energy + nuclear attraction.

        Returns:
        --------
        Dict
            Dictionary with core Hamiltonian for electrons and positrons
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis

        # Check if kinetic and nuclear attraction are already computed
        if (
            'kinetic_e' not in self.matrices
            or 'nuclear_attraction_e' not in self.matrices
        ):
            self.build_one_body_matrices()

        # Combine for electrons: H_core = T + V
        T_e = self.matrices['kinetic_e']
        V_e = self.matrices['nuclear_attraction_e']
        H_core_e = T_e + V_e

        # Combine for positrons if needed
        H_core_p = None
        if n_p_basis > 0:
            T_p = self.matrices['kinetic_p']
            V_p = self.matrices['nuclear_attraction_p']
            H_core_p = T_p + V_p

        # Store in matrices dictionary
        self.matrices['H_core_electron'] = H_core_e
        if n_p_basis > 0:
            self.matrices['H_core_positron'] = H_core_p

        # Track performance
        end_time = time.time()
        self.timings['core_hamiltonian'] = end_time - start_time

        return {'electron': H_core_e, 'positron': H_core_p}

    def compute_electron_repulsion_integrals(self) -> Union[np.ndarray, Any]:
        """
        Compute the electron-electron repulsion integrals efficiently.

        For large basis sets, the four-center integrals may be stored in a
        compressed format or calculated on-the-fly due to memory constraints.

        Returns:
        --------
        Union[np.ndarray, Any]
            Four-dimensional array of ERIs or a calculator object
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis

        # Check if system is too large for full storage
        full_storage = (
            n_e_basis <= 100
        )  # Arbitrary threshold; adjust based on memory constraints

        if full_storage:
            # For smaller systems, use full in-memory storage
            eri = np.zeros((n_e_basis, n_e_basis, n_e_basis, n_e_basis))

            # Compute unique integrals using symmetry
            for i in range(n_e_basis):
                for j in range(i + 1):  # i >= j
                    for k in range(n_e_basis):
                        for l in range(k + 1):  # k >= l
                            # Use symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = ...
                            if (i == j) and (k > l):
                                continue
                            if (i > j) and (k > l) and (i > k):
                                continue

                            # Get basis functions
                            basis_i = self.basis_set.get_basis_function(i)
                            basis_j = self.basis_set.get_basis_function(j)
                            basis_k = self.basis_set.get_basis_function(k)
                            basis_l = self.basis_set.get_basis_function(l)

                            # Calculate integral
                            integral = self.integral_engine.electron_repulsion_integral(
                                basis_i, basis_j, basis_k, basis_l
                            )

                            # Store using permutational symmetry
                            # (chemist's notation: (ij|kl))
                            eri[i, j, k, l] = integral
                            if j != i:
                                eri[j, i, k, l] = integral
                            if l != k:
                                eri[i, j, l, k] = integral
                            if j != i and l != k:
                                eri[j, i, l, k] = integral

                            # (kl|ij) symmetry
                            if i != k or j != l:
                                eri[k, l, i, j] = integral
                                if j != i:
                                    eri[k, l, j, i] = integral
                                if l != k:
                                    eri[l, k, i, j] = integral
                                if j != i and l != k:
                                    eri[l, k, j, i] = integral

            self.matrices['electron_repulsion'] = eri

        else:
            # For larger systems, use a calculator that computes ERIs on demand
            # with caching of frequently used values
            class ERICalculator:
                def __init__(self, basis_set, integral_engine):
                    self.basis_set = basis_set
                    self.integral_engine = integral_engine
                    self._cache = {}

                def __getitem__(self, indices):
                    i, j, k, l = indices

                    # Check cache first
                    cache_key = (min(i, j), max(i, j), min(k, l), max(k, l))
                    if cache_key in self._cache:
                        value = self._cache[cache_key]
                        # Apply permutation if needed
                        if (i > j) != (cache_key[0] > cache_key[1]) or (k > l) != (
                            cache_key[2] > cache_key[3]
                        ):
                            return value

                    # Calculate the integral
                    basis_i = self.basis_set.get_basis_function(i)
                    basis_j = self.basis_set.get_basis_function(j)
                    basis_k = self.basis_set.get_basis_function(k)
                    basis_l = self.basis_set.get_basis_function(l)

                    value = self.integral_engine.electron_repulsion_integral(
                        basis_i, basis_j, basis_k, basis_l
                    )

                    # Cache the value
                    self._cache[cache_key] = value

                    return value

            # Create calculator instance
            eri = ERICalculator(self.basis_set, self.integral_engine)

            # For consistency, still store in matrices dictionary
            self.matrices['electron_repulsion'] = eri

        # Track performance
        end_time = time.time()
        self.timings['electron_repulsion'] = end_time - start_time

        return self.matrices['electron_repulsion']

    def compute_positron_repulsion_integrals(self) -> Union[np.ndarray, Any]:
        """
        Compute the positron-positron repulsion integrals.
        Similar to electron-electron repulsion but for positrons.

        Returns:
        --------
        Union[np.ndarray, Any]
            Four-dimensional array of positron ERIs or a calculator object
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis

        if n_p_basis <= 0:
            self.matrices['positron_repulsion'] = None
            return None

        # Check if system is too large for full storage
        full_storage = n_p_basis <= 100  # Arbitrary threshold

        if full_storage:
            # For smaller systems, use full in-memory storage
            eri = np.zeros((n_p_basis, n_p_basis, n_p_basis, n_p_basis))

            # Compute unique integrals using symmetry
            for i in range(n_p_basis):
                for j in range(i + 1):  # i >= j
                    for k in range(n_p_basis):
                        for l in range(k + 1):  # k >= l
                            # Use symmetry: (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) = (kl|ij) = ...
                            if (i == j) and (k > l):
                                continue
                            if (i > j) and (k > l) and (i > k):
                                continue

                            # Get basis functions (add n_e_basis to get positron basis)
                            basis_i = self.basis_set.get_basis_function(i + n_e_basis)
                            basis_j = self.basis_set.get_basis_function(j + n_e_basis)
                            basis_k = self.basis_set.get_basis_function(k + n_e_basis)
                            basis_l = self.basis_set.get_basis_function(l + n_e_basis)

                            # Calculate integral
                            integral = self.integral_engine.positron_repulsion_integral(
                                basis_i, basis_j, basis_k, basis_l
                            )

                            # Store using permutational symmetry
                            eri[i, j, k, l] = integral
                            if j != i:
                                eri[j, i, k, l] = integral
                            if l != k:
                                eri[i, j, l, k] = integral
                            if j != i and l != k:
                                eri[j, i, l, k] = integral

                            # (kl|ij) symmetry
                            if i != k or j != l:
                                eri[k, l, i, j] = integral
                                if j != i:
                                    eri[k, l, j, i] = integral
                                if l != k:
                                    eri[l, k, i, j] = integral
                                if j != i and l != k:
                                    eri[l, k, j, i] = integral

            self.matrices['positron_repulsion'] = eri

        else:
            # For larger systems, use on-the-fly calculation with caching
            class PositronERICalculator:
                def __init__(self, basis_set, integral_engine, n_e_basis):
                    self.basis_set = basis_set
                    self.integral_engine = integral_engine
                    self.n_e_basis = n_e_basis
                    self._cache = {}

                def __getitem__(self, indices):
                    i, j, k, l = indices

                    # Check cache first
                    cache_key = (min(i, j), max(i, j), min(k, l), max(k, l))
                    if cache_key in self._cache:
                        value = self._cache[cache_key]
                        # Apply permutation if needed
                        if (i > j) != (cache_key[0] > cache_key[1]) or (k > l) != (
                            cache_key[2] > cache_key[3]
                        ):
                            return value

                    # Calculate the integral
                    basis_i = self.basis_set.get_basis_function(i + self.n_e_basis)
                    basis_j = self.basis_set.get_basis_function(j + self.n_e_basis)
                    basis_k = self.basis_set.get_basis_function(k + self.n_e_basis)
                    basis_l = self.basis_set.get_basis_function(l + self.n_e_basis)

                    value = self.integral_engine.positron_repulsion_integral(
                        basis_i, basis_j, basis_k, basis_l
                    )

                    # Cache the value
                    self._cache[cache_key] = value

                    return value

            # Create calculator instance
            eri = PositronERICalculator(self.basis_set, self.integral_engine, n_e_basis)

            # Store in matrices dictionary
            self.matrices['positron_repulsion'] = eri

        # Track performance
        end_time = time.time()
        self.timings['positron_repulsion'] = end_time - start_time

        return self.matrices['positron_repulsion']

    def compute_electron_positron_attraction(self) -> Union[np.ndarray, Any]:
        """
        Compute electron-positron attraction integrals.

        Returns:
        --------
        Union[np.ndarray, Any]
            Four-dimensional array of e-p attraction integrals or a calculator
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis

        if n_e_basis <= 0 or n_p_basis <= 0:
            self.matrices['electron_positron_attraction'] = None
            return None

        # Check if system is too large for full storage
        full_storage = n_e_basis * n_p_basis <= 10000  # Arbitrary threshold

        if full_storage:
            # For smaller systems, use full in-memory storage
            eri = np.zeros((n_e_basis, n_e_basis, n_p_basis, n_p_basis))

            # Compute all integrals with appropriate symmetry
            for i in range(n_e_basis):
                for j in range(i + 1):  # i >= j for electron indices
                    for k in range(n_p_basis):
                        for l in range(k + 1):  # k >= l for positron indices
                            # Get basis functions
                            e_basis_i = self.basis_set.get_basis_function(i)
                            e_basis_j = self.basis_set.get_basis_function(j)
                            p_basis_k = self.basis_set.get_basis_function(k + n_e_basis)
                            p_basis_l = self.basis_set.get_basis_function(l + n_e_basis)

                            # Calculate integral
                            integral = self.integral_engine.electron_positron_attraction_integral(
                                e_basis_i, e_basis_j, p_basis_k, p_basis_l
                            )

                            # Store using partial permutational symmetry
                            # We still have (ij|kl) = (ji|kl) = (ij|lk) = (ji|lk) symmetry
                            # but not (kl|ij) since e-p is different from p-e
                            eri[i, j, k, l] = integral
                            if j != i:
                                eri[j, i, k, l] = integral
                            if l != k:
                                eri[i, j, l, k] = integral
                            if j != i and l != k:
                                eri[j, i, l, k] = integral

            self.matrices['electron_positron_attraction'] = eri

        else:
            # For larger systems, use on-the-fly calculation with caching
            class ElectronPositronIntegralCalculator:
                def __init__(self, basis_set, integral_engine, n_e_basis):
                    self.basis_set = basis_set
                    self.integral_engine = integral_engine
                    self.n_e_basis = n_e_basis
                    self._cache = {}

                def __getitem__(self, indices):
                    i, j, k, l = indices

                    # Check cache first - use symmetry within e-e and p-p indices
                    cache_key = (min(i, j), max(i, j), min(k, l), max(k, l))
                    if cache_key in self._cache:
                        value = self._cache[cache_key]
                        # Apply permutation if needed
                        if (i > j) != (cache_key[0] > cache_key[1]) or (k > l) != (
                            cache_key[2] > cache_key[3]
                        ):
                            return value

                    # Calculate the integral
                    e_basis_i = self.basis_set.get_basis_function(i)
                    e_basis_j = self.basis_set.get_basis_function(j)
                    p_basis_k = self.basis_set.get_basis_function(k + self.n_e_basis)
                    p_basis_l = self.basis_set.get_basis_function(l + self.n_e_basis)

                    value = self.integral_engine.electron_positron_attraction_integral(
                        e_basis_i, e_basis_j, p_basis_k, p_basis_l
                    )

                    # Cache the value
                    self._cache[cache_key] = value

                    return value

            # Create calculator instance
            eri = ElectronPositronIntegralCalculator(
                self.basis_set, self.integral_engine, n_e_basis
            )

            # Store in matrices dictionary
            self.matrices['electron_positron_attraction'] = eri

        # Track performance
        end_time = time.time()
        self.timings['electron_positron_attraction'] = end_time - start_time

        return self.matrices['electron_positron_attraction']

    def compute_annihilation_operator(self) -> Optional[np.ndarray]:
        """
        Compute the annihilation operator for electron-positron pairs.

        Returns:
        --------
        Optional[np.ndarray]
            Matrix representation of the annihilation operator
        """
        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis

        if not self.include_annihilation or n_e_basis <= 0 or n_p_basis <= 0:
            self.matrices['annihilation'] = None
            return None

        # Initialize annihilation matrix
        annihilation = np.zeros((n_e_basis, n_p_basis))

        # The annihilation integral is related to the overlap of
        # electron and positron functions at the same point
        for i in range(n_e_basis):
            for j in range(n_p_basis):
                e_basis = self.basis_set.get_basis_function(i)
                p_basis = self.basis_set.get_basis_function(j + n_e_basis)

                # Check if integral engine has an annihilation method
                if hasattr(self.integral_engine, 'annihilation_integral'):
                    annihilation[i, j] = self.integral_engine.annihilation_integral(
                        e_basis, p_basis
                    )
                else:
                    # Approximate using a delta function-like approach
                    # This is a simplified approach; a proper implementation would use
                    # explicit annihilation integrals

                    # For s-type functions, can be calculated analytically
                    if all(x == 0 for x in e_basis.angular_momentum) and all(
                        x == 0 for x in p_basis.angular_momentum
                    ):
                        alpha = e_basis.exponent
                        beta = p_basis.exponent
                        gamma = alpha + beta

                        Ra = e_basis.center
                        Rb = p_basis.center

                        # Calculate exp(-alpha*beta/gamma * |Ra-Rb|²)
                        diff = Ra - Rb
                        dist_squared = np.sum(diff**2)

                        # Approximation based on overlap at coincident points
                        prefactor = (4 * alpha * beta / gamma**2) ** 1.5
                        exp_term = np.exp(-alpha * beta / gamma * dist_squared)

                        annihilation[i, j] = (
                            prefactor
                            * exp_term
                            * e_basis.normalization
                            * p_basis.normalization
                        )
                    else:
                        # For non-s-type functions, use a more complex approach or set to zero
                        # In many cases, p, d orbitals have zero density at origin so annihilation is small
                        annihilation[i, j] = 0.0

        # Store in matrices dictionary
        self.matrices['annihilation'] = annihilation

        # Track performance
        end_time = time.time()
        self.timings['annihilation'] = end_time - start_time

        return annihilation

    def build_relativistic_corrections(self) -> Dict:
        """
        Compute relativistic correction terms.

        This includes:
        1. Mass-velocity correction
        2. Darwin term
        3. Spin-orbit coupling (if spin is included)

        Returns:
        --------
        Dict
            Dictionary of relativistic correction matrices
        """
        if not self.include_relativistic:
            return {}

        start_time = time.time()

        n_e_basis = self.basis_set.n_electron_basis
        n_p_basis = self.basis_set.n_positron_basis

        # Initialize matrices
        mass_velocity_e = np.zeros((n_e_basis, n_e_basis))
        darwin_e = np.zeros((n_e_basis, n_e_basis))

        # For positrons if needed
        mass_velocity_p = np.zeros((n_p_basis, n_p_basis)) if n_p_basis > 0 else None
        darwin_p = np.zeros((n_p_basis, n_p_basis)) if n_p_basis > 0 else None

        # Calculate mass-velocity correction for electrons
        # This is related to p⁴ operator: -1/(8c²) ∇⁴
        for i in range(n_e_basis):
            for j in range(i + 1):  # Use symmetry
                basis_i = self.basis_set.get_basis_function(i)
                basis_j = self.basis_set.get_basis_function(j)

                # Calculate using integral engine if available
                if hasattr(self.integral_engine, 'mass_velocity_integral'):
                    mass_velocity_e[i, j] = self.integral_engine.mass_velocity_integral(
                        basis_i, basis_j
                    )
                else:
                    # Approximate using relation to kinetic energy
                    # For Gaussian basis functions, related to second derivative of kinetic energy
                    alpha = basis_i.exponent
                    beta = basis_j.exponent

                    # Simplified mass-velocity approximation
                    t_ij = self.integral_engine.kinetic_integral(basis_i, basis_j)
                    mass_velocity_e[i, j] = t_ij * t_ij / (2.0 * (alpha + beta))

                # Use symmetry
                if i != j:
                    mass_velocity_e[j, i] = mass_velocity_e[i, j]

        # Calculate Darwin term for electrons
        # This is: (πZ/2c²) δ(r)
        for i in range(n_e_basis):
            for j in range(i + 1):  # Use symmetry
                basis_i = self.basis_set.get_basis_function(i)
                basis_j = self.basis_set.get_basis_function(j)

                darwin_sum = 0.0

                for _, charge, position in self.nuclei:
                    # Calculate using integral engine if available
                    if hasattr(self.integral_engine, 'darwin_integral'):
                        darwin_term = self.integral_engine.darwin_integral(
                            basis_i, basis_j, position
                        )
                    else:
                        # Approximate based on value at the nucleus
                        # Only s-type functions contribute significantly
                        if all(x == 0 for x in basis_i.angular_momentum) and all(
                            x == 0 for x in basis_j.angular_momentum
                        ):
                            # Values at the nucleus
                            alpha = basis_i.exponent
                            beta = basis_j.exponent
                            gamma = alpha + beta

                            Ra = basis_i.center
                            Rb = basis_j.center
                            Rc = position

                            # Gaussian product center
                            P = (alpha * Ra + beta * Rb) / gamma

                            # Distance terms
                            diff_AB = Ra - Rb
                            diff_PC = P - Rc

                            # Exponential terms
                            exp_term = np.exp(
                                -alpha * beta / gamma * np.sum(diff_AB**2)
                            )
                            exp_term *= np.exp(-gamma * np.sum(diff_PC**2))

                            # Prefactor for delta function
                            prefactor = basis_i.normalization * basis_j.normalization
                            prefactor *= (4 * alpha * beta / gamma**2) ** 1.5

                            darwin_term = prefactor * exp_term
                        else:
                            # Non-s-type functions have zero density at nucleus
                            darwin_term = 0.0

                    darwin_sum += charge * darwin_term

                darwin_e[i, j] = darwin_sum

                # Use symmetry
                if i != j:
                    darwin_e[j, i] = darwin_e[i, j]

        # Similar calculations for positrons if needed
        if n_p_basis > 0:
            # For positrons, core calculations are similar but with sign changes
            # that reflect the opposite charge
            for i in range(n_p_basis):
                for j in range(i + 1):
                    basis_i = self.basis_set.get_basis_function(i + n_e_basis)
                    basis_j = self.basis_set.get_basis_function(j + n_e_basis)

                    # Mass-velocity term (same formula as electrons)
                    if hasattr(self.integral_engine, 'mass_velocity_integral'):
                        mass_velocity_p[i, j] = (
                            self.integral_engine.mass_velocity_integral(
                                basis_i, basis_j
                            )
                        )
                    else:
                        alpha = basis_i.exponent
                        beta = basis_j.exponent
                        t_ij = self.integral_engine.kinetic_integral(basis_i, basis_j)
                        mass_velocity_p[i, j] = t_ij * t_ij / (2.0 * (alpha + beta))

                    # Darwin term (opposite sign compared to electrons)
                    darwin_sum = 0.0
                    for _, charge, position in self.nuclei:
                        if hasattr(self.integral_engine, 'darwin_integral'):
                            darwin_term = self.integral_engine.darwin_integral(
                                basis_i, basis_j, position
                            )
                        else:
                            # Similar approximation as for electrons
                            if all(x == 0 for x in basis_i.angular_momentum) and all(
                                x == 0 for x in basis_j.angular_momentum
                            ):
                                alpha = basis_i.exponent
                                beta = basis_j.exponent
                                gamma = alpha + beta

                                Ra = basis_i.center
                                Rb = basis_j.center
                                Rc = position

                                P = (alpha * Ra + beta * Rb) / gamma

                                diff_AB = Ra - Rb
                                diff_PC = P - Rc

                                exp_term = np.exp(
                                    -alpha * beta / gamma * np.sum(diff_AB**2)
                                )
                                exp_term *= np.exp(-gamma * np.sum(diff_PC**2))

                                prefactor = (
                                    basis_i.normalization * basis_j.normalization
                                )
                                prefactor *= (4 * alpha * beta / gamma**2) ** 1.5

                                darwin_term = prefactor * exp_term
                            else:
                                darwin_term = 0.0

                        # Note: opposite sign for positrons
                        darwin_sum -= charge * darwin_term

                    darwin_p[i, j] = darwin_sum

                    # Use symmetry
                    if i != j:
                        mass_velocity_p[j, i] = mass_velocity_p[i, j]
                        darwin_p[j, i] = darwin_p[i, j]

        # Scale by physical constants
        c_squared = 137.036**2  # Speed of light squared in atomic units

        mass_velocity_e *= -1.0 / (8.0 * c_squared)
        darwin_e *= np.pi / (2.0 * c_squared)

        if n_p_basis > 0:
            mass_velocity_p *= -1.0 / (8.0 * c_squared)
            darwin_p *= np.pi / (2.0 * c_squared)

        # Store results
        relativistic = {'mass_velocity_e': mass_velocity_e, 'darwin_e': darwin_e}

        if n_p_basis > 0:
            relativistic.update(
                {'mass_velocity_p': mass_velocity_p, 'darwin_p': darwin_p}
            )

        # Store in matrices
        self.matrices.update(relativistic)

        # Track performance
        end_time = time.time()
        self.timings['relativistic'] = end_time - start_time

        return relativistic

    def get_performance_report(self) -> Dict:
        """
        Get a detailed performance report for Hamiltonian construction.

        Returns:
        --------
        Dict
            Performance timing information
        """
        total = sum(self.timings.values())

        # Calculate percentages
        percentages = {
            k: 100.0 * v / total if total > 0 else 0.0 for k, v in self.timings.items()
        }

        # Create report
        report = {
            'times': self.timings,
            'percentages': percentages,
            'total_time': total,
            'matrix_dimensions': {
                'overlap': (
                    self.matrices['overlap'].shape
                    if 'overlap' in self.matrices
                    else None
                ),
                'core_e': (
                    self.matrices['H_core_electron'].shape
                    if 'H_core_electron' in self.matrices
                    else None
                ),
                'core_p': (
                    self.matrices['H_core_positron'].shape
                    if 'H_core_positron' in self.matrices
                    else None
                ),
            },
        }

        return report
