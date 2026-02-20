# Manual fix for PyTorch3D CUDA header issues
# This modifies the setup.py to use correct CUDA headers

import os
import re

def fix_pytorch3d_setup():
    """
    Fix PyTorch3D setup.py to use system CUDA headers instead of conda's
    """
    setup_py_path = "pytorch3d\\setup.py"
    
    if not os.path.exists(setup_py_path):
        print("setup.py not found. Make sure you're in the pytorch3d directory.")
        return False
    
    # Read setup.py
    with open(setup_py_path, 'r') as f:
        content = f.read()
    
    # Find the include directories section and modify it
    # We need to prioritize system CUDA headers over conda's
    
    # Look for CUDA include paths and reorder them
    cuda_system_include = r'"C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8\\\\include"'
    
    # Backup original
    with open(setup_py_path + '.backup', 'w') as f:
        f.write(content)
    
    # Method 1: Try to find and modify include_dirs
    if 'include_dirs' in content:
        # Add system CUDA include at the beginning of include_dirs
        content = re.sub(
            r'(include_dirs\s*=\s*\[)',
            f'\\1\n        {cuda_system_include},',
            content
        )
    
    # Method 2: Look for CUDAExtension and modify its include_dirs
    content = re.sub(
        r'(CUDAExtension\([^)]*include_dirs\s*=\s*\[)',
        f'\\1\n            {cuda_system_include},',
        content,
        flags=re.DOTALL
    )
    
    # Write modified setup.py
    with open(setup_py_path, 'w') as f:
        f.write(content)
    
    print("setup.py modified to prioritize system CUDA headers")
    return True

def create_custom_build_script():
    """
    Create a custom build script that sets proper environment
    """
    script_content = '''
import os
import subprocess
import sys

# Set environment variables
os.environ['CUDA_PATH'] = r'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8'
os.environ['CUDA_HOME'] = r'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8'
os.environ['FORCE_CUDA'] = '1'
os.environ['PYTORCH3D_NO_NINJA'] = '1'
os.environ['DISTUTILS_USE_SDK'] = '1'
os.environ['MAX_JOBS'] = '4'

# Remove conda CUDA from include path temporarily
conda_include = r'C:\\\\Users\\\\clint\\\\anaconda3\\\\envs\\\\augenblick\\\\include'
if conda_include in os.environ.get('INCLUDE', ''):
    includes = os.environ['INCLUDE'].split(';')
    includes = [inc for inc in includes if 'anaconda3' not in inc or 'crt' not in inc]
    os.environ['INCLUDE'] = ';'.join(includes)

# Add system CUDA include at the front
system_cuda_include = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include'
current_include = os.environ.get('INCLUDE', '')
os.environ['INCLUDE'] = system_cuda_include + ';' + current_include

print("Environment set, building PyTorch3D...")
print("CUDA_PATH:", os.environ.get('CUDA_PATH'))
print("INCLUDE (first 200 chars):", os.environ.get('INCLUDE', '')[:200])

# Run the build
result = subprocess.run([sys.executable, 'setup.py', 'install'], 
                       cwd=os.getcwd(),
                       capture_output=False)

sys.exit(result.returncode)
'''
    
    with open('build_pytorch3d.py', 'w') as f:
        f.write(script_content)
    
    print("Created build_pytorch3d.py - run this instead of setup.py")

if __name__ == "__main__":
    print("Fixing PyTorch3D CUDA header issues...")
    
    if fix_pytorch3d_setup():
        print("✓ setup.py modified")
    
    create_custom_build_script()
    print("✓ Custom build script created")
    
    print("\nNext steps:")
    print("1. cd to your pytorch3d directory")
    print("2. Run: python build_pytorch3d.py (or setup.py)")
    print("3. Or run the batch script for full environment setup")