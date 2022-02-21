import json
from image_captioning_bot.files import bot_path

class Vocab:
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    UNK_IDX = None
    VOCAB_DIM = None
    vocabulary = None
    word_to_index = None

    def __init__(self):
        self.vocabulary = json.load(open(bot_path('imcap-vocab.json')))
        self.VOCAB_DIM = len(self.vocabulary)
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
