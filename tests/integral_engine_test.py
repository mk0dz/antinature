# test_integral_engine.py
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from antinature.core.basis import GaussianBasisFunction
from antinature.core.integral_engine import AntinatureIntegralEngine


def test_integral_engine():
    """Test the improved integral engine with simple basis functions."""
    # Create engine
    engine = AntinatureIntegralEngine()

    # Create two simple s-type functions
    s1 = GaussianBasisFunction(
        center=np.array([0.0, 0.0, 0.0]), exponent=1.0, angular_momentum=(0, 0, 0)
    )

    s2 = GaussianBasisFunction(
        center=np.array([0.5, 0.0, 0.0]), exponent=1.5, angular_momentum=(0, 0, 0)
    )

    # Create p-type function
    p1 = GaussianBasisFunction(
        center=np.array([0.0, 0.0, 0.0]), exponent=1.0, angular_momentum=(1, 0, 0)
    )

    # Test overlap integral
    overlap_s1s2 = engine.overlap_integral(s1, s2)
    print(f"Overlap <s1|s2>: {overlap_s1s2:.6f}")

    # Test kinetic integral
    kinetic_s1s2 = engine.kinetic_integral(s1, s2)
    print(f"Kinetic <s1|-∇²/2|s2>: {kinetic_s1s2:.6f}")

    # Test nuclear attraction integral
    nuclear_s1s2 = engine.nuclear_attraction_integral(s1, s2, np.array([0.0, 0.0, 0.0]))
    print(f"Nuclear <s1|1/r|s2>: {nuclear_s1s2:.6f}")

    # Test electron repulsion integral
    eri_s1s2s1s2 = engine.electron_repulsion_integral(s1, s2, s1, s2)
    print(f"ERI <s1s2|s1s2>: {eri_s1s2s1s2:.6f}")

    # Test p-orbitals
    overlap_p1s2 = engine.overlap_integral(p1, s2)
    print(f"Overlap <p1|s2>: {overlap_p1s2:.6f}")

    # Print performance report
    print("\nPerformance Report:")
    for key, value in engine.get_performance_report().items():
        if key != 'total' and key != 'cache_size' and key != 'cache_limit':
            print(f"  {key}: {value:.6f} seconds")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_integral_engine()
