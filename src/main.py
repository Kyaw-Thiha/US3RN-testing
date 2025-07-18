from preprocess import change_key
from downsample import load_downsample_save


FILE_PATH = "../data"
data_src = "indian_pines"
HSI_spectral_bands = 31
MSI_spectral_bands = 3


def main(base_dir: str):
    print(f"Changing the keys of the files from {base_dir}")
    change_key(f"{base_dir}", f"{base_dir}/test/X", "msi")
    change_key(f"{base_dir}", f"{base_dir}/test/Y", "RGB")
    print("-------------------------------------")

    print(f"Downsampling the files from {base_dir}/test/X to {HSI_spectral_bands} spectral bands")
    load_downsample_save(
        f"{base_dir}/test/X", f"{base_dir}/test/X", "msi", spatial_scale=1, spectral_algorithm="pca", out_bands=HSI_spectral_bands
    )
    print("-------------------------------------")

    print(f"Downsampling the files from {base_dir}/test/Y to {MSI_spectral_bands} spectral bands")
    load_downsample_save(
        f"{base_dir}/test/Y", f"{base_dir}/test/Y", "RGB", spatial_scale=1, spectral_algorithm="pca", out_bands=MSI_spectral_bands
    )
    print("-------------------------------------")


if __name__ == "__main__":
    main(f"{FILE_PATH}/{data_src}")
