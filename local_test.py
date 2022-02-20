import PIL
import numpy as np
import image_captions as imcap


def load_image(path):
    # return plt.imread(path)
    return np.array(PIL.Image.open(path))


def run_local_test():
    print("Initializing model...")
    imcap.init_model('data')

    print("Loading image...")
    image = load_image('data/test.jpg')
    
    print("Generating captions...")
    captions = imcap.generate_captions(image)

    print('\n'.join(captions))
