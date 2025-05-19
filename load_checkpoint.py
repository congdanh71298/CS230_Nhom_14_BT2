import os
import gdown


def load_checkpoint_folder_from_drive():
    folder_url = "https://drive.google.com/drive/folders/1k_c1oYxl3wpwHqiwNZOGv5x_gNZ5IKKv?usp=drive_link"

    gdown.download_folder(url=folder_url, quiet=False, use_cookies=False)

if __name__ == "__main__":
    load_checkpoint_folder_from_drive()
