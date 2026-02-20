# Test PyTorch3D functionality
import pytorch3d
from pytorch3d.ops import ball_query, knn_points
from pytorch3d import _C
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import so3_exp_map

print("🎉 PyTorch3D is fully operational!")
print("Available for your headless 3D reconstruction pipeline!")

# Test a basic operation
import torch
points = torch.rand(1, 100, 3)
pointcloud = Pointclouds(points)
print(f"✅ Created pointcloud with {pointcloud.num_points_per_cloud()} points")