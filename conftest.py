import pytest
import sys
import platform

# Check Python version
PYTHON_VERSION = sys.version_info
IS_PY38 = PYTHON_VERSION.major == 3 and PYTHON_VERSION.minor == 8

# Check if Qiskit is available
try:
    import qiskit
    HAS_QISKIT = True
    QISKIT_VERSION = qiskit.__version__
except ImportError:
    HAS_QISKIT = False
    QISKIT_VERSION = '0.0.0'

# Check if specific Qiskit modules are available
try:
    import qiskit_nature
    HAS_QISKIT_NATURE = True
except ImportError:
    HAS_QISKIT_NATURE = False

try:
    import qiskit_algorithms
    HAS_QISKIT_ALGORITHMS = True
except ImportError:
    HAS_QISKIT_ALGORITHMS = False

try:
    import qiskit_aer
    HAS_QISKIT_AER = True
except ImportError:
    HAS_QISKIT_AER = False

@pytest.fixture(scope="session", autouse=True)
def check_dependencies():
    """Check and report about available dependencies at the start of the test session."""
    print(f"\nRunning tests with Python {platform.python_version()}")
    print(f"Qiskit available: {HAS_QISKIT}")
    if HAS_QISKIT:
        print(f"Qiskit version: {QISKIT_VERSION}")
    print(f"Qiskit Nature available: {HAS_QISKIT_NATURE}")
    print(f"Qiskit Algorithms available: {HAS_QISKIT_ALGORITHMS}")
    print(f"Qiskit Aer available: {HAS_QISKIT_AER}")

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "requires_qiskit: mark test as requiring qiskit")
    config.addinivalue_line("markers", "requires_qiskit_nature: mark test as requiring qiskit_nature")
    config.addinivalue_line("markers", "requires_qiskit_algorithms: mark test as requiring qiskit_algorithms")
    config.addinivalue_line("markers", "requires_qiskit_aer: mark test as requiring qiskit_aer")
    config.addinivalue_line("markers", "skip_on_py38: mark test to be skipped on Python 3.8")

def pytest_runtest_setup(item):
    """Skip tests based on markers and available dependencies."""
    # Skip tests that require specific Python versions
    if any(mark.name == 'skip_on_py38' for mark in item.iter_markers()) and IS_PY38:
        pytest.skip("Test skipped on Python 3.8")
        
    # Skip tests that require Qiskit and its modules
    if any(mark.name == 'requires_qiskit' for mark in item.iter_markers()) and not HAS_QISKIT:
        pytest.skip("Test requires Qiskit")
        
    if any(mark.name == 'requires_qiskit_nature' for mark in item.iter_markers()) and not HAS_QISKIT_NATURE:
        pytest.skip("Test requires Qiskit Nature")
        
    if any(mark.name == 'requires_qiskit_algorithms' for mark in item.iter_markers()) and not HAS_QISKIT_ALGORITHMS:
        pytest.skip("Test requires Qiskit Algorithms")
        
    if any(mark.name == 'requires_qiskit_aer' for mark in item.iter_markers()) and not HAS_QISKIT_AER:
        pytest.skip("Test requires Qiskit Aer") 