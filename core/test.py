import sys
import time
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import scipy.io as io
from torch.utils.data import DataLoader

from analytics.csv_test_logger import CSVTestLogger
from analytics.tee_logger import TeeLogger
from data import get_test_set
from core.psnr import MPSNR
from core.ssim import MSSIM

from core.core import Options, build_model, load_model


test_dir = "data/ksc"


def load_test(opt: Options):
    """
    Loads the test dataset using options provided.

    Args:
        opt (TestOptions): Parsed options containing configuration.

    Returns:
        torch.utils.data.DataLoader: The testing data loader.
    """
    print("===> Loading test datasets")
    # test_set = get_test_set(opt.upscale_factor)
    test_set = get_test_set(test_dir, 1)
    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.testBatchSize,
        shuffle=False,
        pin_memory=True,
    )
    return testing_data_loader


def extract_patches(tensor: torch.Tensor, patch_size: int, stride: int) -> torch.Tensor:
    """
    Extracts non-overlapping patches from an input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W].
        patch_size (int): Size of the square patch.
        stride (int): Stride used for unfolding.

    Returns:
        torch.Tensor: Patches of shape [N, C, patch_size, patch_size], where N is the number of patches.
    """
    C, H, W = tensor.shape
    patches = tensor.unfold(1, patch_size, stride).unfold(2, patch_size, stride)
    patches = patches.contiguous().view(C, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3)  # [num_patches, C, H, W]
    return patches


def reconstruct_from_patches(
    patches: torch.Tensor, image_shape: Tuple[int, int, int], patch_size: int, stride: int
) -> torch.Tensor:
    """
    Reconstructs an image from patches using averaging in overlapping regions.

    Args:
        patches (torch.Tensor): Tensor of patches [N, C, patch_size, patch_size].
        image_shape (Tuple[int, int, int]): Shape of the original image (C, H, W).
        patch_size (int): Size of the square patch.
        stride (int): Stride used during extraction.

    Returns:
        torch.Tensor: Reconstructed image of shape [C, H, W].
    """
    C, H, W = image_shape
    output = torch.zeros((C, H, W), device=patches.device)
    count = torch.zeros((C, H, W), device=patches.device)

    patch_idx = 0
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            output[:, y : y + patch_size, x : x + patch_size] += patches[patch_idx]
            count[:, y : y + patch_size, x : x + patch_size] += 1
            patch_idx += 1

    return output / count.clamp(min=1)


def test(model: torch.nn.Module, opt: Options, print_patch: bool = True) -> Tuple[float, float]:
    """
    Tests the model using 64x64 patches of hyperspectral images and computes PSNR and SSIM.

    Returns:
        float: Average PSNR across all test samples.
    """
    log_file_path = f"logs/test_logs/test_{opt.nEpochs}.log"
    sys.stdout = TeeLogger(log_file_path)

    testing_data_loader = load_test(opt)
    patch_size = 64
    stride = 64  # no overlap

    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_time = 0.0
    model.eval()

    with torch.no_grad():
        for batch in testing_data_loader:
            W_full, X_full, im_name = batch
            W_full = W_full.squeeze(0).cuda()  # [C, H, W]
            X_full = X_full.squeeze(0).cuda()

            _, H, W = X_full.shape
            patches_W = extract_patches(W_full, patch_size, stride)
            patches_X = extract_patches(X_full, patch_size, stride)

            recon_HX_patches: List[torch.Tensor] = []
            patch_psnrs: List[float] = []
            patch_ssims: List[float] = []

            torch.cuda.synchronize()
            start_time = time.time()

            for i in range(patches_W.shape[0]):
                W_patch = patches_W[i].unsqueeze(0).float()
                X_patch = patches_X[i].float()

                # Apply bicubic downsampling on spatial dims of W_patch before feeding to model
                C, H_patch, W_patch_ = W_patch.shape[1:]
                W_patch = F.interpolate(W_patch, scale_factor=1.0 / opt.upscale_factor, mode="bicubic", align_corners=False)

                HX_patch, *_ = model(W_patch)
                HX_patch = HX_patch.squeeze(0)

                # print(f"W_patch: {W_patch.shape}")
                # print(f"HX_patch: {HX_patch.shape}")
                # print(f"X_patch: {X_patch.shape}")

                psnr = MPSNR(HX_patch.cpu().permute(1, 2, 0).numpy(), X_patch.cpu().permute(1, 2, 0).numpy())
                ssim = MSSIM(HX_patch.cpu().permute(1, 2, 0).numpy(), X_patch.cpu().permute(1, 2, 0).numpy())

                if print_patch:
                    print(f"PSNR of Patch-{i + 1}: {round(psnr, 4)} dB")
                    print(f"SSIM of Patch-{i + 1}: {round(ssim, 4)}")
                    print("-----------------------------")

                patch_psnrs.append(psnr)
                patch_ssims.append(ssim)
                recon_HX_patches.append(HX_patch)

            torch.cuda.synchronize()
            end_time = time.time()

            recon_HX_patches_tensor = torch.stack(recon_HX_patches)
            HX_full = reconstruct_from_patches(recon_HX_patches_tensor, X_full.shape, patch_size, stride)

            avg_psnr += sum(patch_psnrs) / len(patch_psnrs)
            avg_ssim += sum(patch_ssims) / len(patch_ssims)
            avg_time += end_time - start_time

            if len(testing_data_loader) > 1:
                print(f"Analyzing {im_name[0]}")
                print(f"  - Patches: {len(patch_psnrs)}")
                print(f"  - Avg Patch PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

            io.savemat(opt.outputpath + os.path.basename(im_name[0]), {"HX": HX_full.permute(1, 2, 0).cpu().numpy()})

    avg_psnr = round(avg_psnr / len(testing_data_loader), 4)
    avg_ssim = round(avg_ssim / len(testing_data_loader), 4)
    print("===> Overall Avg. PSNR: {:.4f} dB".format(avg_psnr))
    print("===> Overall Avg. SSIM: {:.4f}".format(avg_ssim))
    print("===> Avg. time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    return avg_psnr, avg_ssim


log_path = "logs/csv/test.csv"


def batch_test(
    msi_spectral_bands: int, hsi_spectral_bands: int, opt: Options, start_epoch: int = 5, end_epoch: int = 150, step: int = 5
):
    """
    Perform batch testing over multiple epochs using saved model checkpoints.

    Parameters
    ----------
    msi_spectral_bands : int
        Number of spectral bands in the MSI input.
    hsi_spectral_bands : int
        Number of spectral bands in the HSI ground truth.
    opt : Options
        Configuration object containing testing parameters.
    start_epoch : int, optional
        Starting epoch number for testing (default is 5).
    end_epoch : int, optional
        Final epoch number for testing (default is 150).
    step : int, optional
        Step interval between tested epochs (default is 5).

    Returns
    -------
    None
    """
    model, optimizer, scheduler, criterion = build_model(msi_spectral_bands, hsi_spectral_bands, opt)
    logger = CSVTestLogger(log_path)

    for epoch in range(start_epoch, end_epoch + 1, step):
        print(f"\n Testing Epoch-{epoch}")

        model, optimizer = load_model(model, optimizer, epoch, opt)
        psnr, ssim = test(model, opt, print_patch=False)
        logger.log_epoch(epoch, psnr, ssim)

        print("-----------------------------")
