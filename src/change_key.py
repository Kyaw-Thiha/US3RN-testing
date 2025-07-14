from scipy.io import loadmat, savemat
import os
import numpy as np

FILE_PATH = "../data"


def change_key(input_dir: str, output_dir: str, new_key: str):
    """
    A function that change the key of mat files from the `input_dir`,
    and save the clean versions in the `output_dir`
    Currently, it does
    Change the main key to be 'new_key' parameter

    Note that
    - X should have 'msi' key
    - Y should have 'RGB' key
    """
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".mat"):
            continue
        data = loadmat(os.path.join(input_dir, fname))

        for key, value in data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray):
                img = data.get(key)
                savemat(
                    os.path.join(output_dir, fname),
                    {new_key: img},
                )
        print(f"[✓] Key Changed: {fname} to {new_key}")


if __name__ == "__main__":
    print(f"Changing the keys of the files from {FILE_PATH}")
    change_key(f"{FILE_PATH}/raw", f"{FILE_PATH}/test/X", "msi")
    change_key(f"{FILE_PATH}/raw", f"{FILE_PATH}/test/Y", "RGB")
    print("-------------------------------------")
