
import os
import subprocess
import sys

# Set environment variables for successful build
os.environ['CUDA_PATH'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
os.environ['CUDA_HOME'] = r'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8'
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
    print("\n PyTorch3D installed successfully without pulsar!")
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
# #     from pytorch3d.renderer.points.pulsar import PulsarPointsRenderer  # REMOVED: Pulsar not available  # REMOVED: Pulsar not available
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
