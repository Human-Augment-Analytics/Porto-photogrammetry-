"""PSNR, SSIM and LPIPS, matching the definitions the 3DGS-derived backends already use.

The formulas are reproduced here rather than imported so that scoring does not depend on any
one backend's vendored copy. They are deliberately identical to 2DGS's utils.image_utils.psnr
and utils.loss_utils.ssim, because the numbers they produce are already reported.
"""
import logging
from math import exp

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


def psnr(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Per-channel MSE averaged into one PSNR, as 3DGS's utils.image_utils.psnr does."""
    mse = ((pred - truth) ** 2).view(pred.shape[0], -1).mean(1, keepdim=True)
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).mean())


def _gaussian_window(window_size: int, channels: int) -> torch.Tensor:
    """Build the separable Gaussian window 3DGS's create_window produces."""
    gauss = torch.tensor([exp(-((x - window_size // 2) ** 2) / float(2 * SSIM_SIGMA ** 2))
                          for x in range(window_size)])
    gauss = gauss / gauss.sum()
    window_1d = gauss.unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(pred: torch.Tensor, truth: torch.Tensor, window_size: int = SSIM_WINDOW_SIZE) -> float:
    """Gaussian-windowed SSIM, identical to 3DGS's utils.loss_utils.ssim."""
    channels = pred.size(-3)
    window = _gaussian_window(window_size, channels).to(device=pred.device, dtype=pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, window, padding=pad, groups=channels)
    mu2 = F.conv2d(truth, window, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(truth * truth, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred * truth, window, padding=pad, groups=channels) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
                / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)))
    return float(ssim_map.mean())


class Lpips:
    """LPIPS scorer that falls back to the CPU when a high-resolution view exhausts VRAM.

    Args:
        net: Backbone passed to the lpips package; vgg is what the reported numbers use.
    """

    def __init__(self, net: str = "vgg"):
        # The pip lpips package, not 2DGS's vendored lpipsPyTorch: the two normalise their
        # inputs differently, so swapping them would silently shift every reported value.
        import lpips

        self._package = lpips
        self._net = net
        self._gpu = lpips.LPIPS(net=net).cuda() if torch.cuda.is_available() else None
        self._cpu = None

    def _cpu_model(self):
        """Build the CPU model on first use, since it is usually never needed."""
        if self._cpu is None:
            logger.info("LPIPS falling back to CPU for this view")
            self._cpu = self._package.LPIPS(net=self._net)
        return self._cpu

    def __call__(self, pred: torch.Tensor, truth: torch.Tensor) -> float:
        """Score a [3, H, W] pair in [0, 1], remapped to the [-1, 1] range LPIPS expects."""
        a, b = pred.unsqueeze(0) * 2 - 1, truth.unsqueeze(0) * 2 - 1
        if self._gpu is not None:
            try:
                return float(self._gpu(a.cuda(), b.cuda()).item())
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        return float(self._cpu_model()(a.cpu(), b.cpu()).item())
