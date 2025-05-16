import os
import gdown


def load_checkpoint_folder_from_drive():
    folder_url = "https://drive.google.com/drive/folders/1fA0I43PSdS9dr_M99-MKBDhiryNCuaVh?usp=drive_link"

    gdown.download_folder(url=folder_url, quiet=False, use_cookies=False)

if __name__ == "__main__":
    load_checkpoint_folder_from_drive()
