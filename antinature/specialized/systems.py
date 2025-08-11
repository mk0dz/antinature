"""
Predefined antimatter and exotic matter systems.

This module provides factory methods for creating commonly used
antimatter systems with correct parameters.
"""

import numpy as np
from ..core.molecular_data import MolecularData


class AntinatureSystems:
    """
    Factory class for creating predefined antimatter systems.
    """
    
    @staticmethod
    def get_system(name: str) -> MolecularData:
        """
        Get a predefined system by name.
        
        Parameters:
        -----------
        name : str
            Name of the system
            
        Returns:
        --------
        MolecularData
            The molecular data for the requested system
        """
        systems = {
            'positronium': AntinatureSystems.positronium,
            'Ps': AntinatureSystems.positronium,
            'anti-hydrogen': AntinatureSystems.anti_hydrogen,
            'anti-H': AntinatureSystems.anti_hydrogen,
            'Ps2': AntinatureSystems.positronium_molecule,
            'PsH': AntinatureSystems.positronium_hydride,
            'muonium': AntinatureSystems.muonium,
            'Mu': AntinatureSystems.muonium,
            'antimuonium': AntinatureSystems.antimuonium,
            'anti-He+': AntinatureSystems.anti_helium_ion,
            'protonium': AntinatureSystems.protonium,
        }
        
        if name not in systems:
            raise ValueError(f"Unknown system: {name}")
            
        return systems[name]()
    
    @staticmethod
    def positronium() -> MolecularData:
        """Create positronium (e⁺e⁻) system."""
        return MolecularData(
            atoms=[],
            n_electrons=1,
            n_positrons=1,
            charge=0,
            name="Positronium",
            description="Positronium (e⁺e⁻) - bound electron-positron pair",
            is_positronium=True
        )
    
    @staticmethod
    def anti_hydrogen() -> MolecularData:
        """Create anti-hydrogen (p̄e⁺) system."""
        # Anti-hydrogen with anti-proton nucleus
        mol_data = MolecularData(
            atoms=[('H', np.array([0.0, 0.0, 0.0]))],
            n_electrons=0,
            n_positrons=1,
            charge=0,
            name="Anti-hydrogen",
            description="Anti-hydrogen (p̄e⁺) atom"
        )
        # Override nuclear charge for anti-proton
        mol_data.nuclei = [('H', -1, np.array([0.0, 0.0, 0.0]))]
        mol_data.nuclear_charges = [-1]
        return mol_data
    
    @staticmethod
    def positronium_molecule() -> MolecularData:
        """Create Ps₂ molecule (e⁺e⁻e⁺e⁻)."""
        # Create two positronium centers separated by typical Ps-Ps distance
        separation = 4.0  # Bohr radii (optimized for Ps₂)
        mol_data = MolecularData(
            atoms=[
                ('Ps', np.array([-separation/2, 0.0, 0.0])),  # First Ps center
                ('Ps', np.array([separation/2, 0.0, 0.0]))    # Second Ps center
            ],
            n_electrons=2,
            n_positrons=2,
            charge=0,
            name="Ps2",
            description="Positronium molecule - bound state of two positronium atoms"
        )
        
        # Override nuclear charges - Ps centers have no nuclear charge
        mol_data.nuclei = [
            ('Ps', 0, np.array([-separation/2, 0.0, 0.0])),
            ('Ps', 0, np.array([separation/2, 0.0, 0.0]))
        ]
        mol_data.nuclear_charges = [0, 0]
        
        return mol_data
    
    @staticmethod
    def positronium_hydride() -> MolecularData:
        """Create PsH molecule (HPs)."""
        return MolecularData(
            atoms=[('H', np.array([0.0, 0.0, 0.0]))],
            n_electrons=2,  # 1 from H + 1 from Ps
            n_positrons=1,  # 1 from Ps
            charge=0,  # Overall neutral: H(+1) + e(-1) + Ps(e- + e+) = 0
            name="PsH",
            description="Positronium hydride - hydrogen bound to positronium"
        )
    
    @staticmethod
    def muonium() -> MolecularData:
        """Create muonium (μ⁺e⁻) system."""
        return MolecularData(
            atoms=[('Mu', np.array([0.0, 0.0, 0.0]))],
            n_electrons=1,
            n_positrons=0,
            charge=0,  # Overall neutral: μ⁺(+1) + e⁻(-1) = 0
            name="Muonium",
            description="Muonium (μ⁺e⁻) - muon-electron bound state"
        )
    
    @staticmethod
    def antimuonium() -> MolecularData:
        """Create antimuonium (μ⁻e⁺) system."""
        mol_data = MolecularData(
            atoms=[('Mu-', np.array([0.0, 0.0, 0.0]))],
            n_electrons=0,
            n_positrons=1,
            charge=0,
            name="Antimuonium",
            description="Antimuonium (μ⁻e⁺) - antimuon-positron bound state"
        )
        # μ⁻ has -1 charge
        mol_data.nuclei = [('Mu-', -1, np.array([0.0, 0.0, 0.0]))]
        mol_data.nuclear_charges = [-1]
        return mol_data
    
    @staticmethod
    def anti_helium_ion() -> MolecularData:
        """Create anti-He⁺ ion (anti-helium with one positron)."""
        mol_data = MolecularData(
            atoms=[('He', np.array([0.0, 0.0, 0.0]))],
            n_electrons=0,
            n_positrons=1,
            charge=-1,  # Overall charge: He̅(-2) + e⁺(+1) = -1
            name="Anti-He+",
            description="Anti-helium ion (He̅⁺) with one positron"
        )
        # Anti-helium nucleus has -2 charge
        mol_data.nuclei = [('He', -2, np.array([0.0, 0.0, 0.0]))]
        mol_data.nuclear_charges = [-2]
        return mol_data
    
    @staticmethod
    def protonium() -> MolecularData:
        """Create protonium (p̄p) system."""
        # Proton-antiproton bound state
        # Place them at typical separation distance
        separation = 2.0  # Bohr radii
        
        mol_data = MolecularData(
            atoms=[
                ('H', np.array([0.0, 0.0, -separation/2])),  # Proton
                ('H', np.array([0.0, 0.0, separation/2]))     # Position for antiproton
            ],
            n_electrons=0,
            n_positrons=0,
            charge=0,
            name="Protonium",
            description="Proton-antiproton bound system"
        )
        
        # Set up nuclei with correct charges
        mol_data.nuclei = [
            ('H', 1, np.array([0.0, 0.0, -separation/2])),   # Proton: +1
            ('H', -1, np.array([0.0, 0.0, separation/2]))     # Antiproton: -1
        ]
        mol_data.nuclear_charges = [1, -1]
        
        return mol_data
    
    @staticmethod
    def dipositronium() -> MolecularData:
        """Create dipositronium (e⁺e⁺) - unbound repulsive system."""
        return MolecularData(
            atoms=[],
            n_electrons=0,
            n_positrons=2,
            charge=2,
            name="Dipositronium",
            description="Dipositronium (e⁺e⁺) - repulsive system"
        )
    
    @staticmethod
    def tripositronium() -> MolecularData:
        """Create tripositronium (e⁺e⁻e⁺) three-body system."""
        return MolecularData(
            atoms=[],
            n_electrons=1,
            n_positrons=2,
            charge=1,
            name="Tripositronium",
            description="Tripositronium (e⁺e⁻e⁺) - three-body system"
        )