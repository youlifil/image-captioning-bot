import PIL
import numpy as np
import image_captioning_bot.model as model
import image_captioning_bot.log as log

def load_image(path):
    return np.array(PIL.Image.open(path))


def run_local_test():
    logger = log.logger()
    
    logger.info("Initializing model...")
    model.init_model()

    logger.info("Loading image...")
    image = load_image('data/test.jpg')
    
    logger.info("Generating captions...")
    captions = model.generate_captions(image)

    logger.info('Generated captions are:\n' + '\n'.join(captions))
