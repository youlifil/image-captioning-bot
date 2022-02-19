import os
import gdown

base_path = 'data'


def bot_path(filename):
    return os.path.join(base_path, filename)


def download_google_file(url, output):
    if os.path.isfile(output):
        return
    gdown.download(url, output, quiet=False)