from change_key import change_key
from downsample import load_downsample_save


FILE_PATH = "../data"
data_src = "indian_pines"


def preprocess(base_dir: str):
    print(f"Changing the keys of the files from {base_dir}")
    change_key(f"{base_dir}", f"{base_dir}/test/X", "msi")
    change_key(f"{base_dir}", f"{base_dir}/test/Y", "RGB")
    print("-------------------------------------")

    # print(f"Downsampling the files from {base_dir}")
    # load_downsample_save(f"{base_dir}/test/X", f"{base_dir}/test/X", "msi")
    # print("-------------------------------------")


if __name__ == "__main__":
    preprocess(f"{FILE_PATH}/{data_src}")
