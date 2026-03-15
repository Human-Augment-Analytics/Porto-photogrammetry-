import argparse
import os
import torch
import numpy as np
from PIL import Image
from sugar_utils.general_utils import str2bool
from blender.sugar_utils import (
    load_blender_package, 
    load_sugar_models_from_blender_package, 
    load_cameras_from_blender_package, 
    build_composite_scene,
    render_composited_image,
) 
from rich.console import Console

if __name__ == "__main__":
    print_every_n_frames = 5
    
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to render SuGaR meshes edited or animated with Blender.')
    
    parser.add_argument('-p', '--package_path',
                        type=str, 
                        help='(Required) path to the Blender data package to use for rendering.')
    
    parser.add_argument('-o', '--output_path',
                        type=str, 
                        default=None,
                        help='Path to the output folder where to save the rendered images. \
                        If None, images will be saved in ./output/blender/renders/{package_name}.')
       
    parser.add_argument('--sh_degree', type=int, default=3, help='SH degree to use.')
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False, help='Use a white background instead of black.')
    
    parser.add_argument('--deformation_threshold', type=float, default=2., 
                        help='Threshold for the deformation of the mesh. A face is considered too much deformed if its size increases by a ratio greater than this threshold.')
    
    parser.add_argument('--export_frame_as_ply', type=int, default=0, 
                        help='Export the Frosting representation of the scene at the specified frame as a PLY file. '
                        'If 0, no PLY file will be exported and all frames will be rendered.')
    
    CONSOLE = Console(width=120)
    
    args = parser.parse_args()
    scene_name = os.path.splitext(os.path.basename(args.package_path))[0]
    output_path = args.output_path if args.output_path else f"./output/blender/renders/{scene_name}"
    sh_degree = args.sh_degree
    deformation_threshold = args.deformation_threshold
    frame_to_export_as_ply = args.export_frame_as_ply - 1
    
    # ----- Setup -----
    torch.cuda.set_device(args.gpu)
    device = torch.device(torch.cuda.current_device())
    CONSOLE.print("[INFO] Using device: ", device)
    CONSOLE.print("[INFO] Images will be saved in: ", output_path)
    
    # ----- Load Blender package -----
    CONSOLE.print("\nLoading Blender package...")
    package = load_blender_package(args.package_path, device)
    CONSOLE.print("Blender package loaded.")
    
    # ----- Load SuGaR models -----
    CONSOLE.print("Loading SuGaR models...")
    sugar_models, scene_paths = load_sugar_models_from_blender_package(package, device)
    
    # ----- Build composite scene -----
    CONSOLE.print("\nBuilding composite scene...")
    sugar_comp = build_composite_scene(sugar_models, scene_paths, package)
    
    # ----- Build cameras -----
    CONSOLE.print("Loading cameras...")
    render_cameras = load_cameras_from_blender_package(package, device=device)
    
    # ----- Render and saving images -----
    CONSOLE.print("\nLoading successful. Rendering and saving images...")
    n_frames = len(package['camera']['lens'])
    os.makedirs(output_path, exist_ok=True)
    
    sugar_comp.eval()
    sugar_comp.adapt_to_cameras(render_cameras)
    
    with torch.no_grad():
        if frame_to_export_as_ply == -1:
            for i_frame in range(n_frames):
                rgb_render = render_composited_image(
                    package=package,
                    sugar=sugar_comp, 
                    render_cameras=render_cameras, 
                    i_frame=i_frame,
                    sh_degree=sh_degree,
                    deformation_threshold=deformation_threshold,
                )
            
                # Save image
                save_path = os.path.join(output_path, f"{i_frame+1:04d}.png")
                img = Image.fromarray((rgb_render.cpu().numpy() * 255).astype(np.uint8))
                img.save(save_path)
                
                # Info
                if i_frame % print_every_n_frames == 0:
                    print(f"Saved frame {i_frame} to {save_path}")
                    
                torch.cuda.empty_cache()
        else:
            # Export PLY file
            ply_save_path = os.path.join(output_path, f"{frame_to_export_as_ply+1:04d}.ply")
            render_composited_image(
                package=package,
                sugar=sugar_comp, 
                render_cameras=render_cameras, 
                i_frame=frame_to_export_as_ply,
                sh_degree=sh_degree,
                deformation_threshold=deformation_threshold,
                return_GS_model=True,
            ).save_ply(ply_save_path)
            CONSOLE.print(f"Exported PLY file of frame {frame_to_export_as_ply+1} to {ply_save_path}")
            
CONSOLE.print("Rendering completed.")
