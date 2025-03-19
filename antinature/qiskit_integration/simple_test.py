"""Simple test script to check qiskit-nature imports."""

import sys

print("Python version:", sys.version)

# Try to import qiskit
try:
    import qiskit

    print("Qiskit version:", qiskit.__version__)
    print("Qiskit package location:", qiskit.__file__)
except ImportError as e:
    print("Failed to import qiskit:", e)

# Try to import qiskit_nature
try:
    import qiskit_nature

    print("Qiskit Nature version:", qiskit_nature.__version__)
    print("Qiskit Nature package location:", qiskit_nature.__file__)
    print("Qiskit Nature dir:", dir(qiskit_nature))
except ImportError as e:
    print("Failed to import qiskit_nature:", e)

# Try specific imports
try:
    from qiskit_nature.second_q.hamiltonians import ElectronicEnergy

    print("Successfully imported ElectronicEnergy from second_q")
except ImportError as e:
    print("Failed to import from qiskit_nature.second_q:", e)

# Try qiskit-nature version 0.7.2 structure
try:
    import importlib

    nature_modules = [name for name in dir(qiskit_nature) if not name.startswith('_')]
    print("Qiskit Nature modules:", nature_modules)

    for module_name in nature_modules:
        try:
            module = importlib.import_module(f"qiskit_nature.{module_name}")
            print(f"Module {module_name} contents:", dir(module))
        except ImportError as e:
            print(f"Failed to import qiskit_nature.{module_name}:", e)
except Exception as e:
    print("Error exploring qiskit_nature structure:", e)

print("Test completed")
