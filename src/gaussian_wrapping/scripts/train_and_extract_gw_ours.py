import argparse
from argparse import BooleanOptionalAction
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")
EXTRACT_SCRIPT = os.path.join(BASE_DIR, "pivot_based_mesh_extraction.py")
TEXTURE_SCRIPT = os.path.join(BASE_DIR, "texture_mesh.py")

DEFAULT_TRAIN_FEATURE_DC_LR = 0.0013
DEFAULT_TRAIN_FEATURE_REST_LR = 0.00011
DEFAULT_TRAIN_POSITION_LR_INIT = 0.00016
DEFAULT_TRAIN_POSITION_LR_FINAL = 0.0000016
DEFAULT_TRAIN_POSITION_LR_DELAY_MULT = 0.01
DEFAULT_TRAIN_POSITION_LR_MAX_STEPS = 30_000
DEFAULT_TRAIN_OPACITY_LR = 0.05
DEFAULT_TRAIN_SCALING_LR = 0.005
DEFAULT_TRAIN_ROTATION_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_NETWORK_LR = 0.001
DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR = 0.05 / 2.0
DEFAULT_TRAIN_PGSR_APPEARANCE_LR = 0.001
DEFAULT_MAX_GAUSSIANS = 6_000_000

DEFAULT_EXTRACT_N_PIVOTS = 2
DEFAULT_EXTRACT_N_BINARY_STEPS = 10
DEFAULT_EXTRACT_ISOSURFACE_VALUE = 0.0

DEFAULT_TEXTURE_N_ITER = 1000
DEFAULT_TEXTURE_LAMBDA_DSSIM = 0.2
DEFAULT_TEXTURE_LR = 0.0025
DEFAULT_TEXTURE_SH_DEGREE = 0


def add_flag(args_list, flag, value):
    if value is None:
        return
    args_list.extend([flag, str(value)])


def add_shared_args(args_list, args):
    add_flag(args_list, "-s", args.source_path)
    add_flag(args_list, "-m", args.model_path)
    add_flag(args_list, "-r", args.resolution)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Train Gaussian Wrapping with the 'ours' rasterizer, extract a mesh, "
            "and refine its texture with a small set of fidelity-oriented knobs."
        )
    )

    data_group = parser.add_argument_group("Shared Data Arguments")
    data_group.add_argument("-s", "--source_path")
    data_group.add_argument("-m", "--model_path")
    data_group.add_argument("-r", "--resolution")

    train_group = parser.add_argument_group("Training Fidelity")
    train_group.add_argument("--iterations", type=int, default=30_000)
    train_group.add_argument("--sh_degree", type=int, default=3)
    train_group.add_argument("--max_gaussians", type=int, default=DEFAULT_MAX_GAUSSIANS)
    train_group.add_argument("--densify_until_iter", type=int, default=15_000)
    train_group.add_argument("--densify_grad_threshold", type=float, default=0.0002)
    train_group.add_argument("--lambda_depth_normal", type=float, default=0.05)
    train_group.add_argument("--multiview_factor", type=float, default=1.0)

    lr_group = parser.add_argument_group("Training Learning Rates")
    lr_group.add_argument("--position_lr_init", type=float, default=DEFAULT_TRAIN_POSITION_LR_INIT)
    lr_group.add_argument("--position_lr_final", type=float, default=DEFAULT_TRAIN_POSITION_LR_FINAL)
    lr_group.add_argument("--position_lr_delay_mult", type=float, default=DEFAULT_TRAIN_POSITION_LR_DELAY_MULT)
    lr_group.add_argument("--position_lr_max_steps", type=int, default=DEFAULT_TRAIN_POSITION_LR_MAX_STEPS)
    lr_group.add_argument("--feature_dc_lr", type=float, default=DEFAULT_TRAIN_FEATURE_DC_LR)
    lr_group.add_argument("--feature_rest_lr", type=float, default=DEFAULT_TRAIN_FEATURE_REST_LR)
    lr_group.add_argument("--opacity_lr", type=float, default=DEFAULT_TRAIN_OPACITY_LR)
    lr_group.add_argument("--scaling_lr", type=float, default=DEFAULT_TRAIN_SCALING_LR)
    lr_group.add_argument("--rotation_lr", type=float, default=DEFAULT_TRAIN_ROTATION_LR)
    lr_group.add_argument("--appearance_embeddings_lr", type=float, default=DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR)
    lr_group.add_argument("--appearance_network_lr", type=float, default=DEFAULT_TRAIN_APPEARANCE_NETWORK_LR)
    lr_group.add_argument("--gaussian_features_lr", type=float, default=DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR)
    lr_group.add_argument("--pgsr_appearance_lr", type=float, default=DEFAULT_TRAIN_PGSR_APPEARANCE_LR)

    extract_group = parser.add_argument_group("Mesh Extraction Fidelity")
    extract_group.add_argument(
        "--extract_iteration",
        type=int,
        default=None,
        help="Checkpoint iteration to use for extraction and texture refinement. Defaults to --iterations.",
    )
    extract_group.add_argument("--n_pivots", type=int, default=DEFAULT_EXTRACT_N_PIVOTS)
    extract_group.add_argument("--n_binary_steps", type=int, default=DEFAULT_EXTRACT_N_BINARY_STEPS)
    extract_group.add_argument("--isosurface_value", type=float, default=DEFAULT_EXTRACT_ISOSURFACE_VALUE)
    extract_group.add_argument("--postprocess", action=BooleanOptionalAction, default=True)
    extract_group.add_argument("--filter_large_edges", action=BooleanOptionalAction, default=True)

    texture_group = parser.add_argument_group("Texture Refinement Fidelity")
    texture_group.add_argument("--texture_n_iter", type=int, default=DEFAULT_TEXTURE_N_ITER)
    texture_group.add_argument("--texture_lambda_dssim", type=float, default=DEFAULT_TEXTURE_LAMBDA_DSSIM)
    texture_group.add_argument("--texture_lr", type=float, default=DEFAULT_TEXTURE_LR)
    texture_group.add_argument("--texture_sh_degree", type=int, default=DEFAULT_TEXTURE_SH_DEGREE)

    return parser


def build_train_args(args, passthrough_args):
    train_args = [
        sys.executable,
        TRAIN_SCRIPT,
        "--rasterizer",
        "ours",
        "--feature_dc_lr",
        str(args.feature_dc_lr),
        "--feature_rest_lr",
        str(args.feature_rest_lr),
        "--position_lr_init",
        str(args.position_lr_init),
        "--position_lr_final",
        str(args.position_lr_final),
        "--position_lr_delay_mult",
        str(args.position_lr_delay_mult),
        "--position_lr_max_steps",
        str(args.position_lr_max_steps),
        "--opacity_lr",
        str(args.opacity_lr),
        "--scaling_lr",
        str(args.scaling_lr),
        "--rotation_lr",
        str(args.rotation_lr),
        "--appearance_embeddings_lr",
        str(args.appearance_embeddings_lr),
        "--appearance_network_lr",
        str(args.appearance_network_lr),
        "--gaussian_features_lr",
        str(args.gaussian_features_lr),
        "--pgsr_appearance_lr",
        str(args.pgsr_appearance_lr),
        "--exposure_compensation",
        "--data_device",
        "cpu",
        "--iterations",
        str(args.iterations),
        "--sh_degree",
        str(args.sh_degree),
        "--N_max_gaussians",
        str(args.max_gaussians),
        "--densify_until_iter",
        str(args.densify_until_iter),
        "--densify_grad_threshold",
        str(args.densify_grad_threshold),
        "--lambda_depth_normal",
        str(args.lambda_depth_normal),
        "--multiview_factor",
        str(args.multiview_factor),
    ]

    add_shared_args(train_args, args)
    train_args.extend(passthrough_args)
    return train_args


def build_extract_args(args, extract_iteration):
    extract_args = [
        sys.executable,
        EXTRACT_SCRIPT,
        "--sdf_mode",
        "ours",
        "--rasterizer",
        "ours",
        "--dtype",
        "int32",
        "--n_pivots",
        str(args.n_pivots),
        "--n_binary_steps",
        str(args.n_binary_steps),
        "--isosurface_value",
        str(args.isosurface_value),
        "--iteration",
        str(extract_iteration),
        "--use_valid_mask",
        "--data_device",
        "cpu",
    ]
    if args.postprocess:
        extract_args.append("--postprocess")
    if args.filter_large_edges:
        extract_args.append("--filter_large_edges")

    add_shared_args(extract_args, args)
    return extract_args


def build_texture_args(args, mesh_path, extract_iteration):
    texture_args = [
        sys.executable,
        TEXTURE_SCRIPT,
        "--rasterizer",
        "ours",
        "--mesh",
        mesh_path,
        "--iteration",
        str(extract_iteration),
        "--n_iter",
        str(args.texture_n_iter),
        "--lambda_dssim",
        str(args.texture_lambda_dssim),
        "--lr",
        str(args.texture_lr),
        "--sh_degree_for_texturing",
        str(args.texture_sh_degree),
    ]
    add_shared_args(texture_args, args)
    return texture_args


def get_mesh_path(model_path, n_pivots, postprocess):
    mesh_name = f"mesh_ours_{n_pivots}pivots"
    if postprocess:
        mesh_name += "_post"
    mesh_name += ".ply"
    if model_path:
        return os.path.join(model_path, mesh_name)
    return mesh_name


def main():
    parser = build_parser()
    args, passthrough_args = parser.parse_known_args(sys.argv[1:])

    if passthrough_args:
        print(
            "[INFO] Forwarding additional arguments to train.py only: "
            + " ".join(passthrough_args)
        )

    extract_iteration = args.extract_iteration or args.iterations
    mesh_path = get_mesh_path(args.model_path, args.n_pivots, args.postprocess)

    print("[INFO] Step 1/3: Training...")
    train_result = subprocess.run(build_train_args(args, passthrough_args))
    if train_result.returncode != 0:
        print("[ERROR] Training failed. Aborting extraction.")
        sys.exit(train_result.returncode)

    print("[INFO] Step 2/3: Extracting mesh...")
    extract_result = subprocess.run(build_extract_args(args, extract_iteration))
    if extract_result.returncode != 0:
        print("[ERROR] Mesh extraction failed. Aborting texture refinement.")
        sys.exit(extract_result.returncode)

    print("[INFO] Step 3/3: Refining texture...")
    texture_result = subprocess.run(build_texture_args(args, mesh_path, extract_iteration))
    if texture_result.returncode != 0:
        print("[ERROR] Texture refinement failed.")
        sys.exit(texture_result.returncode)


if __name__ == "__main__":
    main()
