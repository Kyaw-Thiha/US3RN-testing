from typing import Optional
from scipy.io import loadmat, savemat
import os
import tifffile
import numpy as np

FILE_PATH = "data"


def change_key(input_dir: str, output_dir: str, new_key: str):
    """
    A function that change the key of mat, npy, and tif files from the `input_dir`,
    and save the clean versions as mat file in the `output_dir`

    Operations:
    - Load the image from .mat, .npy or .tif file.
    - Transpose from (H, W, C) to (C, H, W) if 3D.
    - Save the result as .mat with `new_key` as the variable name.

    Note that
    - X should have 'msi' key
    - Y should have 'RGB' key
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        img_path = os.path.join(input_dir, fname)
        img = None

        if fname.endswith(".mat"):
            img = process_mat(img_path)
        elif fname.endswith(".npy"):
            img = process_npy(img_path)
        elif fname.endswith(".tif"):
            img = process_tif(img_path)
        else:
            continue

        if img is None:
            print(f"[!] Warning: No valid array found in {fname}")
            continue

        output_fname = os.path.splitext(fname)[0] + ".mat"
        savemat(
            os.path.join(output_dir, output_fname),
            {new_key: img},
        )
        print(f"[✓] Key Changed: {fname} to {new_key}")


def process_mat(img_path: str):
    """
    Loads a .mat file, extracts the first valid 3D NumPy array, and transposes it
    from (H, W, C) to (C, H, W) if applicable.

    Parameters:
        img_path (str): Path to the .mat file.

    Returns:
        np.ndarray or None: The transposed array, or None if not found.
    """
    data = loadmat(img_path)

    img = None
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray):
            img = data.get(key)

            if img is not None and img.ndim == 3:
                img = img.transpose(2, 0, 1)
                print(f"Transposed shape: {img.shape}")
            return img
    print(f"[❌] Error: No valid key found in {img_path}")


def process_npy(img_path: str):
    """
    Loads a .npy file and transposes the array from (H, W, C) to (C, H, W) if 3D.

    Parameters:
        img_path (str): Path to the .npy file.

    Returns:
        np.ndarray: The loaded and possibly transposed array.
    """
    img = np.load(img_path)
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)
        print(f"Transposed shape: {img.shape}")
        return img
    print(f"[❌] Error: Array shape is wrong in {img_path}")


def process_tif(img_path: str) -> Optional[np.ndarray]:
    """
    Loads a .tif HSI image and ensures the output shape is (C, H, W).

    Parameters:
        img_path (str): Path to the .tif file.

    Returns:
        np.ndarray: The loaded and shape-corrected HSI array.
    """
    img = tifffile.imread(img_path)
    if img.ndim == 3:
        # Case: (H, W, C) -> (C, H, W)
        if img.shape[2] < 1000:  # likely channels last
            img = img.transpose(2, 0, 1)
            print(f"Transposed shape: {img.shape}")
        else:
            print(f"[✅] Already in (C, H, W) shape: {img.shape}")
        return img
    else:
        print(f"[❌] Error: Unexpected array shape {img.shape} in {img_path}")


if __name__ == "__main__":
    print(f"Changing the keys of the files from {FILE_PATH}")
    change_key(f"{FILE_PATH}/raw", f"{FILE_PATH}/test/X", "msi")
    change_key(f"{FILE_PATH}/raw", f"{FILE_PATH}/test/Y", "RGB")
    print("-------------------------------------")
