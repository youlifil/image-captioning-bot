import logging

class Config: pass

def set_logger(logger):
    Config.logger = logger


def logger():
    return Config.logger