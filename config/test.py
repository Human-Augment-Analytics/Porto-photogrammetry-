# Clear module cache and test
import sys
import importlib

# Remove all pytorch3d modules from cache
modules_to_remove = [name for name in sys.modules.keys() if name.startswith('pytorch3d')]
for module_name in modules_to_remove:
    del sys.modules[module_name]

print(f"Cleared {len(modules_to_remove)} pytorch3d modules from cache")

# Now test the imports
try:
    import pytorch3d
    print("✅ PyTorch3D base import")
    
    import pytorch3d.ops
    print("✅ pytorch3d.ops import")
    
    from pytorch3d import _C
    print("✅ SUCCESS! _C import working!")
    print("Available _C functions:", [x for x in dir(_C) if not x.startswith('_')][:10])
    
except Exception as e:
    print(f"❌ Still failing: {e}")
    import traceback
    traceback.print_exc()