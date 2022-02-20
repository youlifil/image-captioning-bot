import os
import gdown

class Config: pass

def download_google_file(url, output):
    if os.path.isfile(output):
        return
    gdown.download(url, output, quiet=False)


def set_base(base_path):
    Config.base_path = base_path


def path(filename):
    return os.path.join(Config.base_path, filename)
