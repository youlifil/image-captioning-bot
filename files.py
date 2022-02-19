import os

base_path = 'data'

def bot_path(filename):
    return os.path.join(base_path, filename)