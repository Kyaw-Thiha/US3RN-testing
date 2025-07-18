import numpy as np
import math


def MPSNR(img1, img2):
    # Crop the image to smallest dimension
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    ch = min(img1.shape[2], img2.shape[2]) if img1.ndim == 3 else 1
    # ch = np.size(img1, 2)

    img1 = img1[:h, :w, :ch] if img1.ndim == 3 else img1[:h, :w]
    img2 = img2[:h, :w, :ch] if img2.ndim == 3 else img2[:h, :w]

    # Calculate the PSNR
    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return s
    else:
        sum = 0
        for i in range(ch):
            mse = np.mean((img1[:, :, i] - img2[:, :, i]) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 1.0
            s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            sum = sum + s
        s = sum / ch
        return s
