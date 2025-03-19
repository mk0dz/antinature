# antinature/core/molecular_data.py

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class MolecularData:
    """
    A comprehensive data structure for antinature molecular systems.

    This class stores all necessary information about molecular structure,
    including atoms, electrons, positrons, and charges. It also provides
    utilities for calculating basic molecular properties and visualization.
    """

    def __init__(
        self,
        atoms: List[Tuple[str, np.ndarray]],
        n_electrons: int = 0,
        n_positrons: int = 0,
        charge: int = 0,
        multiplicity: int = 1,
        units: str = 'bohr',
        name: str = '',
        description: str = '',
        is_positronium: bool = False,
    ):
        """
        Initialize molecular data for an antinature system.

        Parameters:
        -----------
        atoms : List[Tuple[str, np.ndarray]]
            List of (element symbol, position array) tuples
        n_electrons : int
            Number of electrons in the system
        n_positrons : int
            Number of positrons in the system
        charge : int
            Total charge of the system
        multiplicity : int
            Spin multiplicity (2S+1)
        units : str
            Units for atomic positions ('bohr' or 'angstrom')
        name : str
            Name or identifier for the molecule
        description : str
            Optional description of the molecular system
        is_positronium : bool
            Whether this system is a positronium bound state
        """
        self.name = name
        self.description = description
        self.is_positronium = is_positronium

        # Store atomic data
        self.atoms = []
        for atom, position in atoms:
            # Convert position to bohr if given in angstroms
            pos = position.copy()
            if units.lower() == 'angstrom':
                pos = pos * 1.8897259886  # Convert Å to Bohr

            self.atoms.append((atom, pos))

        # Count number of atoms
        self.n_atoms = len(self.atoms)

        # Particle information
        self.n_electrons = n_electrons
        self.n_positrons = n_positrons
        self.charge = charge
        self.multiplicity = multiplicity

        # Calculate number of alpha and beta electrons
        self.n_alpha = n_electrons // 2
        self.n_beta = n_electrons // 2
        if n_electrons % 2 == 1:
            self.n_alpha += 1

        # Calculate number of alpha and beta positrons
        self.n_alpha_positrons = n_positrons // 2
        self.n_beta_positrons = n_positrons // 2
        if n_positrons % 2 == 1:
            self.n_alpha_positrons += 1

        # Create nuclei data with atomic charges
        self.nuclei = []
        self.nuclear_charges = []
        self.atomic_numbers = []
        self.atomic_masses = []

        for atom, position in self.atoms:
            # Get properties from element symbol
            atomic_number = self._get_atomic_number(atom)
            nuclear_charge = atomic_number
            atomic_mass = self._get_atomic_mass(atom)

            self.nuclei.append((atom, nuclear_charge, position))
            self.nuclear_charges.append(nuclear_charge)
            self.atomic_numbers.append(atomic_number)
            self.atomic_masses.append(atomic_mass)

        # Convert lists to numpy arrays
        self.nuclear_charges = np.array(self.nuclear_charges)
        self.atomic_numbers = np.array(self.atomic_numbers)
        self.atomic_masses = np.array(self.atomic_masses)

        # Extract coordinates in more convenient form
        self.geometry = np.vstack([position for _, position in self.atoms])

        # Initialize additional properties
        self._nuclear_repulsion_energy = None
        self._center_of_mass = None
        self._is_antinature_system = n_positrons > 0

    @staticmethod
    def _get_atomic_number(element: str) -> int:
        """Get atomic number from element symbol."""
        atomic_numbers = {
            'H': 1,
            'He': 2,
            'Li': 3,
            'Be': 4,
            'B': 5,
            'C': 6,
            'N': 7,
            'O': 8,
            'F': 9,
            'Ne': 10,
            'Na': 11,
            'Mg': 12,
            'Al': 13,
            'Si': 14,
            'P': 15,
            'S': 16,
            'Cl': 17,
            'Ar': 18,
            'K': 19,
            'Ca': 20,
            'Sc': 21,
            'Ti': 22,
            'V': 23,
            'Cr': 24,
            'Mn': 25,
            'Fe': 26,
            'Co': 27,
            'Ni': 28,
            'Cu': 29,
            'Zn': 30,
            'Ga': 31,
            'Ge': 32,
            'As': 33,
            'Se': 34,
            'Br': 35,
            'Kr': 36,
        }
        return atomic_numbers.get(element, 0)

    @staticmethod
    def _get_atomic_mass(element: str) -> float:
        """Get atomic mass from element symbol in atomic mass units."""
        atomic_masses = {
            'H': 1.00782503,
            'He': 4.00260325,
            'Li': 7.0160045,
            'Be': 9.0121822,
            'B': 11.0093055,
            'C': 12.0000000,
            'N': 14.0030740,
            'O': 15.9949146,
            'F': 18.9984032,
            'Ne': 19.9924356,
            'Na': 22.9897677,
            'Mg': 23.9850419,
            'Al': 26.9815386,
            'Si': 27.9769271,
            'P': 30.9737620,
            'S': 31.9720707,
            'Cl': 34.9688527,
            'Ar': 39.9623831,
            'K': 38.9637074,
            'Ca': 39.9625906,
            'Sc': 44.9559102,
            'Ti': 47.9479471,
            'V': 50.9439617,
            'Cr': 51.9405119,
            'Mn': 54.9380496,
            'Fe': 55.9349421,
            'Co': 58.9332002,
            'Ni': 57.9353479,
            'Cu': 62.9296011,
            'Zn': 63.9291466,
            'Ga': 68.9255776,
            'Ge': 73.9211782,
            'As': 74.9215964,
            'Se': 79.9165218,
            'Br': 78.9183376,
            'Kr': 83.9115083,
        }
        return atomic_masses.get(element, 0.0)

    def get_nuclear_repulsion_energy(self) -> float:
        """
        Calculate nuclear repulsion energy.

        Returns:
        --------
        float
            Nuclear repulsion energy in atomic units (Hartree)
        """
        if self._nuclear_repulsion_energy is not None:
            return self._nuclear_repulsion_energy

        energy = 0.0
        for i, (_, charge_i, pos_i) in enumerate(self.nuclei):
            for j, (_, charge_j, pos_j) in enumerate(self.nuclei[i + 1 :], i + 1):
                r_ij = np.linalg.norm(pos_i - pos_j)
                energy += charge_i * charge_j / r_ij

        self._nuclear_repulsion_energy = energy
        return energy

    def get_center_of_mass(self) -> np.ndarray:
        """
        Calculate center of mass of the molecular system.

        Returns:
        --------
        np.ndarray
            Center of mass coordinates
        """
        if self._center_of_mass is not None:
            return self._center_of_mass

        total_mass = sum(self.atomic_masses)

        if total_mass == 0:
            # If no masses (e.g., for a positronium system),
            # just return geometric center
            self._center_of_mass = np.mean(self.geometry, axis=0)
            return self._center_of_mass

        center = np.zeros(3)
        for i, mass in enumerate(self.atomic_masses):
            center += mass * self.geometry[i]

        self._center_of_mass = center / total_mass
        return self._center_of_mass

    def get_interatomic_distances(self) -> np.ndarray:
        """
        Calculate all interatomic distances.

        Returns:
        --------
        np.ndarray
            Matrix of interatomic distances
        """
        n_atoms = len(self.atoms)
        distances = np.zeros((n_atoms, n_atoms))

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                pos_i = self.geometry[i]
                pos_j = self.geometry[j]
                dist = np.linalg.norm(pos_i - pos_j)

                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def get_bonds(self, scale_factor: float = 1.3) -> List[Tuple[int, int]]:
        """
        Identify chemical bonds based on typical bond lengths.

        Parameters:
        -----------
        scale_factor : float
            Factor to multiply typical bond lengths

        Returns:
        --------
        List[Tuple[int, int]]
            List of (atom_index_1, atom_index_2) tuples representing bonds
        """
        # Typical covalent radii in Bohr
        covalent_radii = {
            'H': 0.74,
            'He': 0.82,
            'Li': 1.33,
            'Be': 1.02,
            'B': 0.85,
            'C': 0.75,
            'N': 0.71,
            'O': 0.63,
            'F': 0.64,
            'Ne': 0.67,
            'Na': 1.55,
            'Mg': 1.39,
            'Al': 1.26,
            'Si': 1.16,
            'P': 1.11,
            'S': 1.03,
            'Cl': 0.99,
            'Ar': 0.96,
        }

        bonds = []
        for i in range(self.n_atoms):
            element_i = self.atoms[i][0]
            radius_i = covalent_radii.get(element_i, 1.0)

            for j in range(i + 1, self.n_atoms):
                element_j = self.atoms[j][0]
                radius_j = covalent_radii.get(element_j, 1.0)

                # Calculate distance between atoms
                distance = np.linalg.norm(self.geometry[i] - self.geometry[j])

                # Check if distance is within typical bond length
                max_bond_length = (radius_i + radius_j) * scale_factor

                if distance <= max_bond_length:
                    bonds.append((i, j))

        return bonds

    def translate(self, displacement: np.ndarray):
        """
        Translate molecule by a displacement vector.

        Parameters:
        -----------
        displacement : np.ndarray
            3D displacement vector
        """
        # Translate all atoms
        for i, (element, _) in enumerate(self.atoms):
            new_position = self.geometry[i] + displacement
            self.atoms[i] = (element, new_position)

            # Update nuclei as well
            if i < len(self.nuclei):
                self.nuclei[i] = (element, self.nuclear_charges[i], new_position)

        # Update geometry array
        self.geometry = np.vstack([position for _, position in self.atoms])

        # Reset cached values
        self._center_of_mass = None
        self._nuclear_repulsion_energy = None

    def rotate(self, rotation_matrix: np.ndarray):
        """
        Rotate molecule using a rotation matrix.

        Parameters:
        -----------
        rotation_matrix : np.ndarray
            3x3 rotation matrix
        """
        # Check if valid rotation matrix
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")

        # Rotate all atoms
        for i, (element, _) in enumerate(self.atoms):
            new_position = rotation_matrix @ self.geometry[i]
            self.atoms[i] = (element, new_position)

            # Update nuclei as well
            if i < len(self.nuclei):
                self.nuclei[i] = (element, self.nuclear_charges[i], new_position)

        # Update geometry array
        self.geometry = np.vstack([position for _, position in self.atoms])

        # Reset center of mass
        self._center_of_mass = None
        # Nuclear repulsion energy should be invariant under rotation

    def to_center_of_mass(self):
        """Translate molecule to center of mass."""
        com = self.get_center_of_mass()
        self.translate(-com)

    def to_standard_orientation(self):
        """
        Orient molecule according to standard orientation
        (principal axes aligned with coordinate axes).
        """
        # First move to center of mass
        self.to_center_of_mass()

        # Calculate inertia tensor
        inertia_tensor = np.zeros((3, 3))

        for i, mass in enumerate(self.atomic_masses):
            r = self.geometry[i]
            r2 = np.sum(r * r)

            # Diagonal elements
            for j in range(3):
                inertia_tensor[j, j] += mass * (r2 - r[j] ** 2)

            # Off-diagonal elements
            for j in range(3):
                for k in range(j + 1, 3):
                    inertia_tensor[j, k] -= mass * r[j] * r[k]
                    inertia_tensor[k, j] = inertia_tensor[j, k]

        # Diagonalize inertia tensor
        eigenvalues, eigenvectors = np.linalg.eigh(inertia_tensor)

        # Sort eigenvectors by eigenvalues (ascending)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # The eigenvectors form the rotation matrix
        # to align principal axes with coordinate axes
        rotation_matrix = eigenvectors.T

        # Apply rotation
        self.rotate(rotation_matrix)

    def is_linear(self, tolerance: float = 1e-6) -> bool:
        """
        Check if the molecule is linear.

        Parameters:
        -----------
        tolerance : float
            Tolerance for linearity check

        Returns:
        --------
        bool
            True if molecule is linear, False otherwise
        """
        if self.n_atoms < 3:
            return True

        # Move to center of mass
        com = self.get_center_of_mass()
        positions = self.geometry - com

        # Calculate inertia tensor
        inertia_tensor = np.zeros((3, 3))

        for i, mass in enumerate(self.atomic_masses):
            r = positions[i]
            r2 = np.sum(r * r)

            # Diagonal elements
            for j in range(3):
                inertia_tensor[j, j] += mass * (r2 - r[j] ** 2)

            # Off-diagonal elements
            for j in range(3):
                for k in range(j + 1, 3):
                    inertia_tensor[j, k] -= mass * r[j] * r[k]
                    inertia_tensor[k, j] = inertia_tensor[j, k]

        # Get eigenvalues of inertia tensor
        eigenvalues = np.linalg.eigvalsh(inertia_tensor)

        # For a linear molecule, one eigenvalue should be approximately zero
        # Sort eigenvalues in ascending order
        eigenvalues.sort()

        # Check if smallest eigenvalue is close to zero
        return eigenvalues[0] < tolerance

    def get_formula(self) -> str:
        """
        Get molecular formula.

        Returns:
        --------
        str
            Molecular formula
        """
        elements = {}
        for atom, _ in self.atoms:
            elements[atom] = elements.get(atom, 0) + 1

        # Build formula string
        formula = ""
        # Standard order: C, H, then alphabetical
        if 'C' in elements:
            formula += 'C'
            if elements['C'] > 1:
                formula += str(elements['C'])
            del elements['C']

        if 'H' in elements:
            formula += 'H'
            if elements['H'] > 1:
                formula += str(elements['H'])
            del elements['H']

        # Add remaining elements in alphabetical order
        for element in sorted(elements.keys()):
            formula += element
            if elements[element] > 1:
                formula += str(elements[element])

        # Add positron information if present
        if self.n_positrons > 0:
            formula += f"(e+{self.n_positrons})"

        # Add charge if non-zero
        if self.charge != 0:
            formula += (
                f" {'+'*self.charge if self.charge > 0 else '-'*abs(self.charge)}"
            )

        return formula

    def visualize(self, show_bonds: bool = True, save_path: str = None):
        """
        Visualize the molecular structure in 3D.

        Parameters:
        -----------
        show_bonds : bool
            Whether to show bonds between atoms
        save_path : str, optional
            Path to save the image
        """
        # Element colors (CPK coloring)
        colors = {
            'H': 'white',
            'He': 'cyan',
            'Li': 'purple',
            'Be': 'darkgreen',
            'B': 'salmon',
            'C': 'black',
            'N': 'blue',
            'O': 'red',
            'F': 'green',
            'Ne': 'cyan',
            'Na': 'purple',
            'Mg': 'darkgreen',
            'Al': 'gray',
            'Si': 'gold',
            'P': 'orange',
            'S': 'yellow',
            'Cl': 'green',
            'Ar': 'cyan',
            'K': 'purple',
            'Ca': 'darkgreen',
        }

        # Element sizes based on atomic radii (scaled)
        sizes = {
            'H': 25,
            'He': 35,
            'Li': 145,
            'Be': 105,
            'B': 85,
            'C': 70,
            'N': 65,
            'O': 60,
            'F': 50,
            'Ne': 45,
            'Na': 180,
            'Mg': 150,
            'Al': 125,
            'Si': 110,
            'P': 100,
            'S': 100,
            'Cl': 100,
            'Ar': 95,
            'K': 220,
            'Ca': 180,
        }

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot atoms
        for i, (element, position) in enumerate(self.atoms):
            color = colors.get(element, 'gray')
            size = sizes.get(element, 70)

            ax.scatter(
                position[0],
                position[1],
                position[2],
                color=color,
                s=size,
                edgecolor='black',
                alpha=0.8,
                label=(
                    f"{i+1}: {element}" if i < 10 else ""
                ),  # Show labels for first 10 atoms
            )

            # Add positrons if present
            if self.n_positrons > 0 and i == 0:  # Just a visual indicator
                ax.scatter(
                    position[0] + 0.5,
                    position[1] + 0.5,
                    position[2] + 0.5,
                    color='magenta',
                    s=50,
                    edgecolor='black',
                    alpha=0.7,
                    label=f"Positron (x{self.n_positrons})",
                )

        # Plot bonds if requested
        if show_bonds and self.n_atoms > 1:
            bonds = self.get_bonds()

            for i, j in bonds:
                pos_i = self.geometry[i]
                pos_j = self.geometry[j]

                # Draw line between bonded atoms
                ax.plot(
                    [pos_i[0], pos_j[0]],
                    [pos_i[1], pos_j[1]],
                    [pos_i[2], pos_j[2]],
                    color='gray',
                    linestyle='-',
                    linewidth=2,
                    alpha=0.6,
                )

        # Set labels and title
        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_zlabel('Z (Bohr)')

        title = f"{self.name + ': ' if self.name else ''}{self.get_formula()}"
        if self._is_antinature_system:
            title += " (antinature System)"
        ax.set_title(title)

        # Set equal aspect ratio
        max_range = (
            max(
                [
                    self.geometry[:, 0].max() - self.geometry[:, 0].min(),
                    self.geometry[:, 1].max() - self.geometry[:, 1].min(),
                    self.geometry[:, 2].max() - self.geometry[:, 2].min(),
                ]
            )
            / 2.0
        )

        # Add small value to avoid identical limits error
        if max_range < 1e-10:
            max_range = 0.5  # Default to 0.5 if molecule is a single atom

        mid_x = (self.geometry[:, 0].max() + self.geometry[:, 0].min()) * 0.5
        mid_y = (self.geometry[:, 1].max() + self.geometry[:, 1].min()) * 0.5
        mid_z = (self.geometry[:, 2].max() + self.geometry[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add legend if not too many atoms
        if self.n_atoms <= 10 or self.n_positrons > 0:
            ax.legend(loc='upper right')

        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Try to show the plot, but handle non-interactive environments gracefully
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display the plot interactively ({str(e)})")
            print("Plot is saved if a save_path was provided.")

        return fig

    def to_xyz_string(self, comment: str = None) -> str:
        """
        Convert molecule to XYZ format string.

        Parameters:
        -----------
        comment : str, optional
            Comment line for XYZ file

        Returns:
        --------
        str
            XYZ format string
        """
        if comment is None:
            if self.name:
                comment = self.name
            else:
                comment = self.get_formula()

        # Start with number of atoms and comment
        xyz = f"{self.n_atoms}\n{comment}\n"

        # Add atoms
        for element, position in self.atoms:
            # Convert position to Angstroms for XYZ format
            pos_angstrom = position / 1.8897259886
            xyz += f"{element:2s}  {pos_angstrom[0]:12.6f}  {pos_angstrom[1]:12.6f}  {pos_angstrom[2]:12.6f}\n"

        return xyz

    def save_xyz(self, filename: str, comment: str = None):
        """
        Save molecule to XYZ file.

        Parameters:
        -----------
        filename : str
            Output filename
        comment : str, optional
            Comment line for XYZ file
        """
        xyz_string = self.to_xyz_string(comment)

        with open(filename, 'w') as f:
            f.write(xyz_string)

    @classmethod
    def from_xyz(
        cls,
        filename: str,
        n_electrons: int = None,
        n_positrons: int = 0,
        charge: int = 0,
    ):
        """
        Create MolecularData from XYZ file.

        Parameters:
        -----------
        filename : str
            XYZ filename
        n_electrons : int, optional
            Number of electrons (defaults to sum of atomic numbers - charge)
        n_positrons : int, optional
            Number of positrons
        charge : int, optional
            Total charge

        Returns:
        --------
        MolecularData
            Molecule object
        """
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Parse number of atoms and comment
        n_atoms = int(lines[0].strip())
        comment = lines[1].strip()

        # Parse atoms
        atoms = []
        for i in range(2, 2 + n_atoms):
            tokens = lines[i].strip().split()
            element = tokens[0]
            position = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])

            # Convert from Angstroms to Bohr
            position = position * 1.8897259886

            atoms.append((element, position))

        # If n_electrons not provided, calculate from atomic numbers
        if n_electrons is None:
            n_electrons = (
                sum(cls._get_atomic_number(atom[0]) for atom in atoms) - charge
            )

        # Create object
        molecule = cls(
            atoms=atoms,
            n_electrons=n_electrons,
            n_positrons=n_positrons,
            charge=charge,
            name=comment,
        )

        return molecule

    @classmethod
    def positronium(cls):
        """
        Create a positronium system (e- + e+).

        Returns:
        --------
        MolecularData
            Positronium object
        """
        # Use a dummy hydrogen atom at origin
        atoms = [('H', np.array([0.0, 0.0, 0.0]))]

        return cls(
            atoms=atoms,
            n_electrons=1,
            n_positrons=1,
            charge=0,
            name="Positronium",
            description="Positronium (e- + e+) bound state",
            is_positronium=True,
        )

    @classmethod
    def anti_hydrogen(cls):
        """
        Create an anti-hydrogen system (p̄ + e+).

        Returns:
        --------
        MolecularData
            Anti-hydrogen object
        """
        # Use a hydrogen atom at origin
        atoms = [('H', np.array([0.0, 0.0, 0.0]))]

        return cls(
            atoms=atoms,
            n_electrons=0,
            n_positrons=1,
            charge=0,  # Anti-proton has -1 charge, positron has +1
            name="Anti-hydrogen",
            description="Anti-hydrogen (p̄ + e+) atom",
        )

    def __str__(self):
        """String representation of the molecular system."""
        s = f"Molecule: {self.name if self.name else self.get_formula()}\n"

        if self.description:
            s += f"Description: {self.description}\n"

        s += f"Number of atoms: {self.n_atoms}\n"
        s += f"Number of electrons: {self.n_electrons}\n"
        s += f"Number of positrons: {self.n_positrons}\n"
        s += f"Charge: {self.charge}\n"
        s += f"Nuclear repulsion energy: {self.get_nuclear_repulsion_energy():.6f} Hartree\n"

        s += "\nAtoms (Bohr):\n"
        for i, (element, position) in enumerate(self.atoms):
            s += f"  {i+1:2d}: {element:2s}  {position[0]:12.6f}  {position[1]:12.6f}  {position[2]:12.6f}\n"

        return s
