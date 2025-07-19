import numpy as np
import math
from skimage.metrics import structural_similarity as ssim


def MSSIM(img1, img2) -> float:
    # Normalize both images to [0, 1]
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    ch = np.size(img1, 2)
    # Crop to smallest common size
    # h = min(img1.shape[0], img2.shape[0])
    # w = min(img1.shape[1], img2.shape[1])
    # ch = min(img1.shape[2], img2.shape[2]) if img1.ndim == 3 else 1
    #
    # img1 = img1[:h, :w, :ch] if img1.ndim == 3 else img1[:h, :w]
    # img2 = img2[:h, :w, :ch] if img2.ndim == 3 else img2[:h, :w]

    if ch == 1:
        return ssim(img1, img2, data_range=1.0)[0]
    else:
        ssim_total = 0.0
        for i in range(ch):
            ssim_score = ssim(img1[:, :, i], img2[:, :, i], data_range=1.0)
            if isinstance(ssim_score, tuple):  # Some versions of skimage return (score, ssim_map)
                ssim_score = ssim_score[0]
            ssim_total += ssim_score
        return ssim_total / ch


def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)
