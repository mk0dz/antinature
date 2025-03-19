# test_basis.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antinature.core.basis import (
    BasisSet,
    GaussianBasisFunction,
    MixedMatterBasis,
    PositronBasis,
)
from antinature.core.integral_engine import AntinatureIntegralEngine


def test_basis_functions():
    """Test the improved basis function implementations."""
    print("Testing Basis Functions...")

    # Create a simple s-type function
    s_func = GaussianBasisFunction(
        center=np.array([0.0, 0.0, 0.0]), exponent=1.0, angular_momentum=(0, 0, 0)
    )
    print(f"s-function: {s_func}")

    # Create a p-type function
    p_func = GaussianBasisFunction(
        center=np.array([0.0, 0.0, 0.0]), exponent=0.8, angular_momentum=(1, 0, 0)
    )
    print(f"p-function: {p_func}")

    # Evaluate at a point
    point = np.array([0.5, 0.0, 0.0])
    print(f"s-function at {point}: {s_func.evaluate(point):.6f}")
    print(f"p-function at {point}: {p_func.evaluate(point):.6f}")

    print("\nTesting BasisSet...")

    # Create a basis set for hydrogen
    h_basis = BasisSet(name="H-STO3G")

    # Add functions manually
    h_basis.add_function(s_func)
    h_basis.add_function(p_func)
    print(f"Hydrogen basis: {h_basis}")

    # Get unique centers and function types
    centers = h_basis.get_unique_centers()
    print(f"Unique centers: {len(centers)}")

    types = h_basis.get_function_types()
    print(f"Function types: {types}")

    # Create a simple basis set for water
    water_basis = BasisSet(name="Water-Simple")
    water_basis.add_function(s_func)  # Add an s function
    water_basis.add_function(p_func)  # Add a p function
    print(f"Water basis: {water_basis}")

    # Get function count
    print(f"Number of functions: {water_basis.n_basis}")

    print("\nTesting PositronBasis...")

    # Create a positron basis
    h_pos_basis = PositronBasis(name="H-positron")

    # Add functions manually
    h_pos_basis.add_function(s_func)
    h_pos_basis.add_function(p_func)
    print(f"Positron basis: {h_pos_basis}")

    print("\nTesting MixedMatterBasis...")

    # Create a mixed basis with our electron and positron basis sets
    mixed_basis = MixedMatterBasis(electron_basis=h_basis, positron_basis=h_pos_basis)

    # Create integral engine
    engine = AntinatureIntegralEngine()
    mixed_basis.set_integral_engine(engine)

    # Get basis function counts
    print(f"Mixed basis electron count: {mixed_basis.n_electron_basis}")
    print(f"Mixed basis positron count: {mixed_basis.n_positron_basis}")
    print(f"Mixed basis total count: {mixed_basis.n_total_basis}")

    # Calculate overlap between electron and positron function
    e_index = 0  # First electron function (s-type)
    p_index = mixed_basis.n_electron_basis  # First positron function (s-type)

    overlap = mixed_basis.overlap_integral(e_index, p_index)
    print(f"Overlap between electron and positron s-functions: {overlap:.6f}")

    # Calculate kinetic energy
    kinetic = mixed_basis.kinetic_integral(e_index, p_index)
    print(
        f"Kinetic energy integral between electron and positron s-functions: {kinetic:.6f}"
    )

    # Calculate nuclear attraction
    nuclear_center = np.array([0.0, 0.0, 0.0])
    nuclear = mixed_basis.nuclear_attraction_integral(e_index, p_index, nuclear_center)
    print(f"Nuclear attraction integral with center at origin: {nuclear:.6f}")

    print("\nBasis function tests completed successfully!")


if __name__ == "__main__":
    test_basis_functions()
