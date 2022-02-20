import numpy as np
from PIL import Image
from image_captions.decoder import Decoder
from image_captions.inception import ImageEncoder
from image_captions.vocab import Vocab
import torch
import torch.nn.functional as F
import image_captions.files as files

class Model: pass


def init_model(stuff_folder):
    if not hasattr(init_model, "done"): 
        files.set_base(stuff_folder)
        Model.vocab = Vocab()
        Model.image_encoder = ImageEncoder()
        Model.decoder = Decoder(Model.vocab.VOCAB_DIM)
        init_model.done = True


def _generate_caption(image, t=0):
    with torch.no_grad():
        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

        img_vec = Model.image_encoder.image_vector(image)
        caption_prefix = [Model.vocab.BOS_TOKEN]

        for _ in range(100):
            prefix_ix = [Model.vocab.word_to_index.get(word, Model.vocab.UNK_IDX) for word in caption_prefix]
            prefix_ix = torch.tensor(prefix_ix).unsqueeze(0)

            logits = Model.decoder.inference(img_vec, prefix_ix)[0, -1]
            probs = F.softmax(logits, -1).detach().numpy()
            
            probs = probs ** t / np.sum(probs ** t)
            next_token = np.random.choice(Model.vocab.vocabulary, p=probs)
            caption_prefix.append(next_token)

            if next_token == Model.vocab.EOS_TOKEN:
                break
            
    return ' '.join(caption_prefix[1:-1])


def generate_captions(image, step_callback=None):
    image = np.array(Image.fromarray(image).resize((299,299))).astype('float32') / 255.

    CAPTIONS_NUM = 10

    captions = []
    for i in range(10):
        if step_callback:
            step_callback(i, CAPTIONS_NUM)

        captions.append(_generate_caption(image=image, t=5.))

    return captions
