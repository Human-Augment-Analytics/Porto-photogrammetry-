"""Score a trained 2DGS model on its held-out views, masked to the specimen.

Renders the evaluation cameras, masks both render and photograph to the object, and writes
PSNR / SSIM / LPIPS over that region to nvs_metrics.json. Background pixels are excluded
because every scene here is a masked turntable capture, where scoring them would mostly
reward reproducing the black surround.

Usage:  python nvs_eval.py -m <model_dir> [--iteration 30000]
"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.image_utils import psnr as psnr_fn
from utils.loss_utils import ssim as ssim_fn


def load_mask(masks_dir, image_name, height, width, device):
    """Return the view's object mask as a [1, H, W] tensor in {0, 1}, or None if absent."""
    path = os.path.join(masks_dir, os.path.splitext(image_name)[0] + ".png")
    if not os.path.exists(path):
        return None
    mask = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    return torch.from_numpy((np.asarray(mask) > 127).astype(np.float32)).to(device).unsqueeze(0)


if __name__ == "__main__":
    parser = ArgumentParser(description="Held-out novel-view metrics for a 2DGS model")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--out", default=None)
    args = get_combined_args(parser)

    dataset, pipe = model.extract(args), pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cameras = scene.getTestCameras()
    if len(cameras) == 0:
        raise SystemExit("no held-out cameras: this model was trained without --eval")

    bg = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg, dtype=torch.float32, device="cuda")
    masks_dir = os.path.join(dataset.source_path, "masks")

    import lpips
    lpips_gpu = lpips.LPIPS(net="vgg").cuda()
    lpips_cpu = []

    def lpips_score(pred, gt):
        """LPIPS on the GPU, falling back to CPU when a 3120 px view does not fit in VRAM."""
        a, b = pred.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1
        try:
            return lpips_gpu(a, b).item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if not lpips_cpu:
                lpips_cpu.append(lpips.LPIPS(net="vgg"))
            return lpips_cpu[0](a.cpu(), b.cpu()).item()

    psnrs, ssims, lpipss, n_masked = [], [], [], 0
    with torch.no_grad():
        for cam in cameras:
            rendered = torch.clamp(render(cam, gaussians, pipe, background)["render"], 0.0, 1.0)
            truth = torch.clamp(cam.original_image.to(rendered.device), 0.0, 1.0)

            mask = load_mask(masks_dir, cam.image_name, rendered.shape[1], rendered.shape[2],
                             rendered.device)
            if mask is not None:
                rendered, truth = rendered * mask, truth * mask
                n_masked += 1

            psnrs.append(psnr_fn(rendered, truth).mean().item())
            ssims.append(ssim_fn(rendered, truth).item())
            lpipss.append(lpips_score(rendered, truth))
            del rendered, truth, mask

    metrics = {
        "model": dataset.model_path,
        "iteration": scene.loaded_iter,
        "resolution": dataset.resolution,
        "n_test": len(cameras),
        "n_masked": n_masked,
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
    }
    # get_combined_args rebuilds the namespace from the model's cfg_args, which never saw --out.
    out_path = getattr(args, "out", None) or os.path.join(dataset.model_path, "nvs_metrics.json")
    with open(out_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    print("NVS_EVAL_DONE", out_path)
