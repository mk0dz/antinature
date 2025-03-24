import sys

print('Python', sys.version)
print('Simulating missing Qiskit...')

# Simulate missing qiskit
sys.modules['qiskit'] = None

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

# Ensure we see some output
print("Test completed successfully!")
