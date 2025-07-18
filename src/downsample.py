from scipy.io import loadmat, savemat
import cv2
import numpy as np
from sklearn.decomposition import PCA
import os

FILE_PATH = "../data"


def load_downsample_save(
    input_dir: str,
    output_dir: str,
    key: str,
    spatial_scale: float = 4,
    spectral_scale: float = 4,
    spectral_algorithm="uniform",
    out_bands: int = -1,
):
    """
    A function that
    1. loads the mat image from input_dir,
    2. downsample the given image by given spatial & spectral scales
    3. save the image in .mat form in output_dir

    Parameters:
    - input_dir: Input directory to load a list of mat images
    - output_dir: Output directory to save the downsampled mat images
    - key: key to access the img from mat files (msi for X, and RGB for Y)
    - img: h x w x s image file
    - spatial_scale (float): how much scale to downsample on spatial level
    - spectral_scale (float): how much scale to downsample on spectral level
    - spectral_algorithm: what algorithm to use for spectral downsampling
      "uniform" - Sample every 'spectral_scale' time
      "pca" - Sample using Principal Component Analysis (Retains spectral bands with most effect on data)
    - out_bands (int): spectral band to downsample to when using pca. -1 indicates using spectral_scale instead.
      Only used for pca
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mat"):
            continue
        data = loadmat(os.path.join(input_dir, fname))
        img = data.get(key)
        img = downsample(img, spatial_scale, spectral_scale, spectral_algorithm, out_bands)
        print(f"[✓] Downsampled: {fname} ")

        savemat(
            os.path.join(output_dir, fname),
            {key: img},
        )


def downsample(img, spatial_scale: float = 4, spectral_scale: float = 4, spectral_algorithm="uniform", out_bands: int = -1):
    """
    A function that downsample the given image by given spatial & spectral scales
    Parameters:
    - img: h x w x s image file
    - spatial_scale (float): how much scale to downsample on spatial level
    - spectral_scale (float): how much scale to downsample on spectral level
    - spectral_algorithm: what algorithm to use for spectral downsampling
      "uniform" - Sample every 'spectral_scale' time
      "pca" - Sample using Principal Component Analysis (Retains spectral bands with most effect on data)
    - out_bands (int): spectral band to downsample to when using pca. -1 indicates using spectral_scale instead.
      Only used for pca
    """
    if spectral_algorithm != "uniform" and spectral_algorithm != "pca":
        return

    h, w, c = img.shape

    # Downsample spatially
    lowres = np.zeros((h // spatial_scale, w // spatial_scale, c), dtype=np.float32)
    for i in range(c):
        band = img[:, :, i]
        band = cv2.resize(
            band,
            (w // spatial_scale, h // spatial_scale),
            interpolation=cv2.INTER_CUBIC,
        )
        lowres[:, :, i] = band

    # Downsample spectrally
    if spectral_algorithm == "uniform":
        # 1. Uniformly sample every spectral_scale-th band
        lowres = lowres[:, :, ::spectral_scale]
    elif spectral_algorithm == "pca":
        # 2. PCA-based spectral downsampling
        if out_bands < 0:
            out_bands = c // spectral_scale
        print(out_bands)
        H, W, C = lowres.shape
        reshaped = lowres.reshape(-1, C)  # shape: (H*W, C)
        pca = PCA(n_components=out_bands)
        projected = pca.fit_transform(reshaped)  # shape: (H*W, out_bands)
        lowres = projected.reshape(H, W, out_bands)

    return lowres


if __name__ == "__main__":
    print(f"Downsampling the files from {FILE_PATH}")
    load_downsample_save(f"{FILE_PATH}/test/X", f"{FILE_PATH}/test/X", "msi")
    print("-------------------------------------")
