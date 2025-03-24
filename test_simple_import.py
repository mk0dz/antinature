import sys

print(f"Python {sys.version}")
print("Testing basic import...")

try:
    import antinature

    print("✅ Basic import successful")
except Exception as e:
    print(f"❌ Error importing base package: {type(e).__name__}: {e}")

print("\nTesting with forced empty qiskit...")
# Remove any existing qiskit module
if 'qiskit' in sys.modules:
    del sys.modules['qiskit']

# Create empty qiskit module
sys.modules['qiskit'] = type('qiskit', (), {})

try:
    import antinature

    print("✅ Import with fake qiskit successful")
except Exception as e:
    print(f"❌ Error importing with fake qiskit: {type(e).__name__}: {e}")
