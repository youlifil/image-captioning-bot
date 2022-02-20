import logging
import os

class Config: pass

def init_logger():
    logger = logging.getLogger('imcapbot')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:  %(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    if not os.path.isdir('logs'):
        os.mkdir('logs')
    f_handler = logging.FileHandler('logs/log.txt')
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:  %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)

    Config.logger = logger


def logger():
    return Config.logger
