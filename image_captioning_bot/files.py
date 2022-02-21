import os
import gdown

BASE_PATH = 'data'

def download_google_file(url, output):
    if os.path.isfile(output):
        return
    gdown.download(url, output, quiet=False)


def bot_path(filename):
    return os.path.join(BASE_PATH, filename)
