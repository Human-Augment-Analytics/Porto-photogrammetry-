# PyTorch3D Installation Without Pulsar Renderer
# This bypasses the CUB dispatch_segmented_sort issue entirely

import os
import shutil

def disable_pulsar_renderer():
    """
    Disable pulsar renderer compilation to avoid CUB issues
    """
    print("Disabling pulsar renderer to bypass CUDA 11.8 CUB issues...")
    
    # Path to the pulsar CUDA source files
    pulsar_cuda_dir = "pytorch3d\\csrc\\pulsar\\cuda"
    pulsar_gpu_dir = "pytorch3d\\csrc\\pulsar\\gpu"
    
    # Create backup if it doesn't exist
    if os.path.exists(pulsar_cuda_dir) and not os.path.exists(pulsar_cuda_dir + "_backup"):
        shutil.copytree(pulsar_cuda_dir, pulsar_cuda_dir + "_backup")
        print(f" Backed up {pulsar_cuda_dir}")
    
    if os.path.exists(pulsar_gpu_dir) and not os.path.exists(pulsar_gpu_dir + "_backup"):
        shutil.copytree(pulsar_gpu_dir, pulsar_gpu_dir + "_backup")
        print(f" Backed up {pulsar_gpu_dir}")
    
    # Remove pulsar CUDA directories to prevent compilation
    if os.path.exists(pulsar_cuda_dir):
        shutil.rmtree(pulsar_cuda_dir)
        print(f" Removed {pulsar_cuda_dir}")
    
    if os.path.exists(pulsar_gpu_dir):
        shutil.rmtree(pulsar_gpu_dir)
        print(f" Removed {pulsar_gpu_dir}")
    
    # Modify setup.py to exclude pulsar
    setup_py_path = "setup.py"
    if os.path.exists(setup_py_path):
        with open(setup_py_path, 'r') as f:
            content = f.read()
        
        # Backup original
        if not os.path.exists(setup_py_path + '.backup'):
            with open(setup_py_path + '.backup', 'w') as f:
                f.write(content)
        
        # Remove pulsar-related extensions
        content = content.replace('"pulsar"', '# "pulsar"  # Disabled for CUDA 11.8 compatibility')
        content = content.replace('"pulsar/cuda"', '# "pulsar/cuda"  # Disabled for CUDA 11.8 compatibility')
        
        # Comment out pulsar extensions in the extensions list
        lines = content.split('\n')
        new_lines = []
        in_extensions = False
        
        for line in lines:
            if 'pulsar' in line.lower() and ('extension' in line.lower() or 'sources' in line.lower()):
                new_lines.append(f'        # {line.strip()}  # Disabled for CUDA 11.8')
            else:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
        
        with open(setup_py_path, 'w') as f:
            f.write(content)
        
        print(" Modified setup.py to exclude pulsar")

def create_no_pulsar_build_script():
    """
    Create a build script that installs PyTorch3D without pulsar
    """
    script_content = '''
import os
import subprocess
import sys

# Set environment variables for successful build
os.environ['CUDA_PATH'] = r'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8'
os.environ['CUDA_HOME'] = r'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v11.8'
os.environ['FORCE_CUDA'] = '1'
os.environ['PYTORCH3D_NO_NINJA'] = '1'
os.environ['DISTUTILS_USE_SDK'] = '1'
os.environ['MAX_JOBS'] = '4'

print("Building PyTorch3D without pulsar renderer...")
print("Environment configured for CUDA 11.8")

# Run the build
result = subprocess.run([sys.executable, 'setup.py', 'install'], 
                       cwd=os.getcwd(),
                       capture_output=False)

if result.returncode == 0:
    print("\\n PyTorch3D installed successfully without pulsar!")
    print("Testing installation...")
    
    test_result = subprocess.run([sys.executable, '-c', """
import pytorch3d
import torch
print("PyTorch3D version:", pytorch3d.__version__)
print("CUDA available:", torch.cuda.is_available())

# Test core functionality (non-pulsar)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras
print("Core PyTorch3D functionality available")

# Note: Pulsar renderer will not be available
try:
    from pytorch3d.renderer.points.pulsar import PulsarPointsRenderer
    print("Pulsar renderer available")
except ImportError:
    print("Pulsar renderer disabled (expected for CUDA 11.8 compatibility)")
"""], capture_output=True, text=True)
    
    print(test_result.stdout)
    if test_result.stderr:
        print("Warnings:", test_result.stderr)
else:
    print("Build failed with exit code:", result.returncode)

sys.exit(result.returncode)
'''
    
    with open('pytorch3d\\build_pytorch3d_no_pulsar.py', 'w') as f:
        f.write(script_content)
    
    print(" Created build_pytorch3d_no_pulsar.py")

def restore_pulsar():
    """
    Restore pulsar files if needed
    """
    pulsar_cuda_dir = "pytorch3d\\csrc\\pulsar\\cuda"
    pulsar_gpu_dir = "pytorch3d\\csrc\\pulsar\\gpu"
    
    if os.path.exists(pulsar_cuda_dir + "_backup"):
        if os.path.exists(pulsar_cuda_dir):
            shutil.rmtree(pulsar_cuda_dir)
        shutil.copytree(pulsar_cuda_dir + "_backup", pulsar_cuda_dir)
        print(f" Restored {pulsar_cuda_dir}")
    
    if os.path.exists(pulsar_gpu_dir + "_backup"):
        if os.path.exists(pulsar_gpu_dir):
            shutil.rmtree(pulsar_gpu_dir)
        shutil.copytree(pulsar_gpu_dir + "_backup", pulsar_gpu_dir)
        print(f" Restored {pulsar_gpu_dir}")
    
    # Restore setup.py
    if os.path.exists("setup.py.backup"):
        shutil.copy2("setup.py.backup", "setup.py")
        print(" Restored setup.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_pulsar()
    else:
        disable_pulsar_renderer()
        create_no_pulsar_build_script()
        
        print("\n" + "="*50)
        print("PyTorch3D No-Pulsar Setup Complete!")
        print("="*50)
        print("Next steps:")
        print("1. Run: python build_pytorch3d_no_pulsar.py")
        print("2. This will install PyTorch3D without the problematic pulsar renderer")
        print("3. All other PyTorch3D functionality will be available")
        print("\nNote: NeuS2 should work fine without the pulsar renderer")
        print("To restore pulsar files: python this_script.py restore")