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
    spatial_scale: float = 1,
    spectral_scale: float = 1,
    spectral_algorithm="uniform",
    target_size: tuple[int, int] = (-1, -1),
    out_bands: int = -1,
    normalization: int = 1,
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
    - target_size: (target_h, target_w) to resize to directly. (-1, -1) indicates using spatial_scale instead.
    - out_bands (int): spectral band to downsample to when using pca. -1 indicates using spectral_scale instead.
      Only used for pca
    - normalization (int 1 or 255): value to which to normalize the image to
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mat"):
            continue
        data = loadmat(os.path.join(input_dir, fname))
        img = data.get(key)
        img = downsample(img, spatial_scale, spectral_scale, spectral_algorithm, target_size, out_bands)
        print(f"[✓] Downsampled: {fname} ")

        if img is not None and normalization == 1:
            img = normalize_image(img)
            print(f"[✓] Normalized to [0, 1]: {fname} ")
        elif img is not None and normalization == 255:
            img = normalize_to_uint8(img)
            print(f"[✓] Normalized to [0, 255]: {fname} ")
        else:
            print(f"[✗] Skipping Normalization due to invalid value: {normalization} ")

        savemat(
            os.path.join(output_dir, fname),
            {key: img},
        )


def downsample(
    img,
    spatial_scale: float = 1,
    spectral_scale: float = 1,
    spectral_algorithm="uniform",
    target_size: tuple[int, int] = (-1, -1),
    out_bands: int = -1,
):
    """
    A function that downsample the given image by given spatial & spectral scales
    Parameters:
    - img: h x w x s image file
    - spatial_scale (float): how much scale to downsample on spatial level
    - spectral_scale (float): how much scale to downsample on spectral level
    - spectral_algorithm: what algorithm to use for spectral downsampling
      "uniform" - Sample every 'spectral_scale' time
      "pca" - Sample using Principal Component Analysis (Retains spectral bands with most effect on data)
      "camera" - Simulate to response function of a camera (Nikon D700 expects 31 spectral bands, return 3)
    - target_size: (target_h, target_w) to resize to directly. (-1, -1) indicates using spatial_scale instead.
    - out_bands (int): spectral band to downsample to when using pca. -1 indicates using spectral_scale instead.
      Only used for pca
    """
    if spectral_algorithm != "uniform" and spectral_algorithm != "pca" and spectral_algorithm != "camera":
        print("Error: Invalid Spectral Algorithm")
        return

    h, w, c = img.shape

    # Downsample spatially
    new_h, new_w = target_size
    if new_h == -1:
        new_h = h // spatial_scale
    if new_w == -1:
        new_w = w // spatial_scale

    lowres = np.zeros((new_h, new_w, c), dtype=np.float32)
    for i in range(c):
        band = img[:, :, i]
        band = cv2.resize(
            band,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )
        lowres[:, :, i] = band

    # Downsample spectrally
    if spectral_algorithm == "uniform":
        # 1. Uniformly sample every spectral_scale-th band
        if out_bands < 0:
            # fallback to current spectral_scale
            lowres = lowres[:, :, ::spectral_scale]
        else:
            C = lowres.shape[2]
            # Compute indices spaced evenly from 0 to C-1 for out_bands
            indices = np.linspace(0, C - 1, out_bands, dtype=int)
            lowres = lowres[:, :, indices]
    elif spectral_algorithm == "pca":
        # 2. PCA-based spectral downsampling
        if out_bands < 0:
            out_bands = c // spectral_scale
        H, W, C = lowres.shape
        reshaped = lowres.reshape(-1, C)  # shape: (H*W, C)
        pca = PCA(n_components=out_bands)
        projected = pca.fit_transform(reshaped)  # shape: (H*W, out_bands)
        lowres = projected.reshape(H, W, out_bands)
    elif spectral_algorithm == "camera":
        # 3. Downsample based on response function of a camera
        lowres = simulate_msi_from_hsi(lowres)
    return lowres


NIKON_D700_RESPONSE = np.load("NIKON_D700_RESPONSE.npy")


def simulate_msi_from_hsi(hsi: np.ndarray, response: np.ndarray = NIKON_D700_RESPONSE) -> np.ndarray:
    """
    Simulate an MSI (RGB) image from an HSI image using a camera spectral response function.

    Parameters:
    - hsi (H x W x C): hyperspectral image, must have 31 spectral bands matching response
    - response (3 x C): camera spectral response matrix (default: Nikon D700)

    Returns:
    - rgb (H x W x 3): simulated RGB image normalized to [0, 1]
    """
    assert hsi.shape[2] == response.shape[1], "HSI and response function must have same number of bands"

    H, W, C = hsi.shape
    hsi_reshaped = hsi.reshape(-1, C)  # (H*W, C)
    rgb_reshaped = hsi_reshaped @ response.T  # (H*W, 3)
    rgb = rgb_reshaped.reshape(H, W, 3)

    return rgb


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image data to the range [0, 1].
    """
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)  # Avoid division by zero
    return (img - min_val) / (max_val - min_val)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image data to the range [0, 255] and convert to uint8.
    """
    img_norm = normalize_image(img)
    return (img_norm * 255).astype(np.uint8)


if __name__ == "__main__":
    print(f"Downsampling the files from {FILE_PATH}")
    load_downsample_save(f"{FILE_PATH}/test/X", f"{FILE_PATH}/test/X", "msi")
    print("-------------------------------------")
