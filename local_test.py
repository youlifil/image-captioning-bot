import matplotlib.pyplot as plt
from files import bot_path
from image_captions import generate_captions, init_model


def load_image(path):
    return plt.imread(path)


def run_local_test():
    print("Initializing context...")
    init_model()

    print("Loading image...")
    image = load_image(bot_path('test.jpg'))
    
    print("Generating captions...")
    captions = generate_captions(image)

    print('\n'.join(captions))
