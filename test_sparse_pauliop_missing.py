import sys

print('Python', sys.version)
print('Simulating problematic Qiskit import...')

# Remove qiskit if it's already imported
if 'qiskit' in sys.modules:
    del sys.modules['qiskit']

# Also clean up any submodules
for key in list(sys.modules.keys()):
    if key.startswith('qiskit.'):
        del sys.modules[key]

# Instead of using the fake module approach which causes issues,
# let's just make sure qiskit isn't in sys.modules
try:
    # Try importing the base package
    import antinature

    print('✅ Basic import successful')

    # Try importing qiskit integration subpackage
    try:
        import antinature.qiskit_integration

        print('✅ Qiskit integration import successful despite missing Qiskit')
    except Exception as e:
        print(f'❌ Error importing qiskit integration: {type(e).__name__}: {e}')
except Exception as e:
    print(f'❌ Error importing base package: {type(e).__name__}: {e}')
