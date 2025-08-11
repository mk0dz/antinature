#!/usr/bin/env python3
"""
Convert Python files with markdown comments to Jupyter notebooks.

This script converts our clean Python example files into Jupyter notebooks
by separating the markdown comments (lines starting with #) from the Python code.
"""

import os
import json
from pathlib import Path

def convert_python_to_notebook(python_file, output_file):
    """Convert a Python file with markdown comments to a Jupyter notebook."""
    
    with open(python_file, 'r') as f:
        lines = f.readlines()
    
    cells = []
    current_cell = []
    current_cell_type = None
    
    for line in lines:
        line = line.rstrip()
        
        # Check if this is a markdown comment line
        if line.startswith('# #') or line.startswith('# '):
            # This is markdown content
            if current_cell_type == 'code' and current_cell:
                # Save the previous code cell
                cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": current_cell
                })
                current_cell = []
            
            # Start or continue markdown cell
            current_cell_type = 'markdown'
            # Remove the '# ' prefix
            if line.startswith('# #'):
                current_cell.append(line[2:] + '\n')  # Remove '# '
            elif line.startswith('# '):
                current_cell.append(line[2:] + '\n')  # Remove '# '
            else:
                current_cell.append('\n')
                
        elif line.startswith('#') and not line.strip() == '#':
            # Regular comment line (part of code)
            if current_cell_type == 'markdown' and current_cell:
                # Save the previous markdown cell
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": current_cell
                })
                current_cell = []
            
            current_cell_type = 'code'
            current_cell.append(line + '\n')
            
        else:
            # This is Python code or empty line
            if current_cell_type == 'markdown' and current_cell:
                # Save the previous markdown cell
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": current_cell
                })
                current_cell = []
            
            current_cell_type = 'code'
            current_cell.append(line + '\n')
    
    # Save the last cell
    if current_cell:
        if current_cell_type == 'markdown':
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": current_cell
            })
        else:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": current_cell
            })
    
    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write the notebook
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Converted {python_file} → {output_file}")

def main():
    """Convert all Python files in the Content directory to notebooks."""
    
    content_dir = Path("antinature-web/src/Content")
    
    # Convert examples
    examples_dir = content_dir / "examples"
    if examples_dir.exists():
        for py_file in examples_dir.glob("*.py"):
            nb_file = py_file.with_suffix(".ipynb")
            convert_python_to_notebook(py_file, nb_file)
    
    # Convert tutorials
    tutorials_dir = content_dir / "tutorials"
    if tutorials_dir.exists():
        for py_file in tutorials_dir.glob("*.py"):
            nb_file = py_file.with_suffix(".ipynb")
            convert_python_to_notebook(py_file, nb_file)
    
    print("\n✅ All Python files converted to Jupyter notebooks!")
    print("The notebooks are ready for use in the documentation.")

if __name__ == "__main__":
    main()
