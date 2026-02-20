#!/usr/bin/env python3
"""
Script to remove alland pulsar-related imports from PyTorch3D codebase.
This ensures clean imports without pulsar dependencies.
"""

import os
import re
import fnmatch
from pathlib import Path

def find_python_files(directory):
    """Find all Python files in the directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def backup_file(file_path):
    """Create a backup of the file before modifying."""
    backup_path = file_path + '.bak'
    if not os.path.exists(backup_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"  Backup created: {backup_path}")

def process_import_lines(content):
    """Process import lines to remove pulsar imports."""
    lines = content.split('\n')
    modified_lines = []
    modifications = []
    
    for i, line in enumerate(lines):
        original_line = line
        
# #         # Pattern 1: from pytorch3d.renderer.points.pulsar import ...  # REMOVED: Pulsar not available  # REMOVED: Pulsar not available
        if re.search(r'from\s+pytorch3d\.renderer\.points\.pulsar\s+import', line):
            new_line = f"# {line}  # REMOVED: Pulsar not available"
            modified_lines.append(new_line)
            modifications.append(f"Line {i+1}: Commented pulsar import")
            continue
            
# #         # Pattern 2: from .pulsar.unified import PulsarPointsRenderer  # REMOVED: Pulsar not available  # REMOVED: Pulsar not available
        if re.search(r'from\s+\.pulsar\.unified\s+import\s+', line):
            new_line = f"# {line}  # REMOVED: Pulsar not available"
            modified_lines.append(new_line)
            modifications.append(f"Line {i+1}: Commented pulsar import")
            continue
            
        # Pattern 3:in import lists
        if '' in line and ('import' in line or 'from' in line):
            # Removefrom import list
            new_line = re.sub(r',?\s*PulsarPointsRenderer\s*,?', '', line)
            # Clean up any resulting double commas or trailing commas
            new_line = re.sub(r',\s*,', ',', new_line)
            new_line = re.sub(r',\s*\)', ')', new_line)
            new_line = re.sub(r'import\s*,', 'import', new_line)
            
            if new_line != original_line:
                modified_lines.append(new_line)
                modifications.append(f"Line {i+1}: Removedfrom import")
            else:
                modified_lines.append(line)
            continue
        
        # Pattern 4: Standalone PulsarPointsRenderer, in lists
        if re.search(r'^\s*PulsarPointsRenderer\s*,?\s*$', line):
            new_line = f"# {line}  # REMOVED: Pulsar not available"
            modified_lines.append(new_line)
            modifications.append(f"Line {i+1}: Commented PulsarPointsRenderer reference")
            continue
            
        # Keep line as-is
        modified_lines.append(line)
    
    return '\n'.join(modified_lines), modifications

def process_test_functions(content):
    """Comment out or modify test functions that use pulsar."""
    lines = content.split('\n')
    modified_lines = []
    modifications = []
    in_pulsar_test = False
    indent_level = 0
    
    for i, line in enumerate(lines):
        # Check if this is a pulsar test function
        if re.search(r'def\s+.*pulsar.*\(', line, re.IGNORECASE):
            in_pulsar_test = True
            indent_level = len(line) - len(line.lstrip())
            new_line = f"{line}\n{' ' * (indent_level + 4)}pytest.skip('Pulsar renderer not available')"
            modified_lines.append(new_line)
            modifications.append(f"Line {i+1}: Added skip to pulsar test function")
            continue
            
        # Check if we're still in a pulsar test function
        if in_pulsar_test:
            current_indent = len(line) - len(line.lstrip()) if line.strip() else float('inf')
            if line.strip() and current_indent <= indent_level:
                in_pulsar_test = False
            
        # Skip lines that instantiate PulsarPointsRenderer
        if 'PulsarPointsRenderer(' in line:
            new_line = f"# {line}  # REMOVED: Pulsar not available"
            modified_lines.append(new_line)
            modifications.append(f"Line {i+1}: Commented PulsarPointsRenderer instantiation")
            continue
            
        modified_lines.append(line)
    
    return '\n'.join(modified_lines), modifications

def clean_file(file_path):
    """Clean a single Python file of pulsar references."""
    print(f"\nProcessing: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False
    
    # Check if file contains pulsar references
    if 'pulsar' not in original_content.lower():
        print(f"  No pulsar references found")
        return False
    
    # Create backup
    backup_file(file_path)
    
    # Process the content
    modified_content = original_content
    all_modifications = []
    
    # Process imports
    modified_content, import_mods = process_import_lines(modified_content)
    all_modifications.extend(import_mods)
    
    # Process test functions (only for test files)
    if 'test' in file_path.lower():
        modified_content, test_mods = process_test_functions(modified_content)
        all_modifications.extend(test_mods)
    
    # Write the modified content
    if all_modifications:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"  Modified ({len(all_modifications)} changes):")
            for mod in all_modifications:
                print(f"    - {mod}")
            return True
        except Exception as e:
            print(f"  Error writing file: {e}")
            return False
    else:
        print(f"  No modifications needed")
        return False

def main():
    """Main function to clean all Python files."""
    print("PyTorch3D Pulsar Cleanup Script")
    print("=" * 50)
    
    # Get the current directory (should be pytorch3d root)
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    # Find all Python files
    python_files = find_python_files(current_dir)
    print(f"Found {len(python_files)} Python files")
    
    # Process each file
    modified_files = 0
    
    for file_path in python_files:
        # Skip backup files
        if file_path.endswith('.bak'):
            continue
            
        if clean_file(file_path):
            modified_files += 1
    
    print(f"\nCleanup Complete!")
    print(f"Modified {modified_files} files")
    print(f"\nTo restore original files, rename .bak files back to .py")
    print(f"To test: python -c \"import pytorch3d; from pytorch3d import _C; print('SUCCESS!')\"")

if __name__ == "__main__":
    main()