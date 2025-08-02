from preprocess.change_key import change_key
from preprocess.downsample import load_downsample_save


FILE_PATH = "data"
data_src = "ksc"
HSI_spectral_bands = 81
MSI_spectral_bands = 40
spatial_factor = 64


def main(base_dir: str):
    print(f"Changing the keys of the files from {base_dir}")
    change_key(f"{base_dir}", f"{base_dir}/test/X", "msi")
    change_key(f"{base_dir}", f"{base_dir}/test/Y", "RGB")
    print("-------------------------------------")

    print(
        f"Downsampling the files from {base_dir}/test/X to {HSI_spectral_bands} spectral bands and {spatial_factor} spatial factor"
    )
    load_downsample_save(
        f"{base_dir}/test/X",
        f"{base_dir}/test/X",
        "msi",
        spatial_factor=spatial_factor,
        spectral_algorithm="pca",
        out_bands=HSI_spectral_bands,
    )
    print("-------------------------------------")

    print(
        f"Downsampling the files from {base_dir}/test/Y to {MSI_spectral_bands} spectral bands and {spatial_factor} spatial factor"
    )
    load_downsample_save(
        f"{base_dir}/test/Y",
        f"{base_dir}/test/Y",
        "RGB",
        spatial_factor=spatial_factor,
        spectral_algorithm="pca",
        out_bands=MSI_spectral_bands,
    )
    print("-------------------------------------")


if __name__ == "__main__":
    main(f"{FILE_PATH}/{data_src}")
