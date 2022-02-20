import PIL
import numpy as np
import image_captions as imcap
import log

def load_image(path):
    # return plt.imread(path)
    return np.array(PIL.Image.open(path))


def run_local_test():
    logger = log.logger()
    
    logger.info("Initializing model...")
    imcap.init_model(stuff_folder='data', logger=logger)

    logger.info("Loading image...")
    image = load_image('data/test.jpg')
    
    logger.info("Generating captions...")
    captions = imcap.generate_captions(image)

    logger.info('Generated captions are:\n' + '\n'.join(captions))
