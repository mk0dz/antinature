#!/usr/bin/env python3
"""
Simple validator script to check if examples can import all required modules.
This doesn't run the examples but verifies all imports are resolved correctly.
"""

import sys
import importlib
import importlib.util
import os
from pathlib import Path

def check_imports(file_path):
    """
    Check if a Python file can import all its dependencies.
    Returns (success, missing_modules).
    """
    print(f"Checking imports for {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract import statements
    lines = content.split('\n')
    import_lines = [line for line in lines if line.strip().startswith('import ') or 
                   line.strip().startswith('from ')]
    
    missing_modules = []
    
    for line in import_lines:
        # Handle multi-line imports
        if '\\' in line:
            continue
            
        # Parse the module name
        if line.strip().startswith('import '):
            modules = line.replace('import ', '').split(',')
            modules = [m.strip().split(' as ')[0] for m in modules]
        else:  # from ... import ...
            try:
                module = line.replace('from ', '').split(' import ')[0].strip()
                modules = [module]
            except IndexError:
                print(f"  Skipping malformed import: {line}")
                continue
        
        # Check each module
        for module in modules:
            # Skip relative imports and known special cases
            if module.startswith('.') or module == 'typing' or module == 'numpy' or module == 'matplotlib':
                continue
                
            try:
                importlib.import_module(module)
                print(f"  ✓ {module}")
            except ImportError as e:
                if 'qiskit' in module and 'Optional Qiskit' in str(e):
                    print(f"  ⚠ {module} (optional)")
                else:
                    print(f"  ✗ {module} - {str(e)}")
                    missing_modules.append(module)
    
    return len(missing_modules) == 0, missing_modules

def main():
    examples_dir = Path('./examples')
    
    if not examples_dir.exists():
        print("Error: examples directory not found!")
        return 1
    
    example_files = list(examples_dir.glob('*.py'))
    
    if not example_files:
        print("Error: No example files found!")
        return 1
    
    print(f"Found {len(example_files)} example files to check")
    
    all_success = True
    for example in example_files:
        success, missing = check_imports(example)
        if not success:
            all_success = False
            print(f"❌ {example.name} has missing dependencies: {', '.join(missing)}")
        else:
            print(f"✅ {example.name} imports look good")
        print()
    
    if all_success:
        print("All examples passed import checking!")
        return 0
    else:
        print("Some examples have missing dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 