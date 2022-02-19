import numpy as np
from PIL import Image
from image_captions.decoder import Decoder
from image_captions.inception import ImageEncoder
from image_captions.vocab import Vocab
import torch
import torch.nn.functional as F


class Context:
    decoder = None
    image_encoder = None
    vocab = None


def init_model():
    if not hasattr(init_model, "context"): 
        ctx = Context()
        ctx.vocab = Vocab()
        ctx.image_encoder = ImageEncoder()
        ctx.decoder = Decoder(ctx.vocab.VOCAB_DIM)
        init_model.context = ctx

    return init_model.context


def generate_caption(context:Context, image, t=0):
    with torch.no_grad():
        image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

        img_vec = context.image_encoder.image_vector(image)
        caption_prefix = [context.vocab.BOS_TOKEN]

        for _ in range(100):
            prefix_ix = [context.vocab.word_to_index.get(word, context.vocab.UNK_IDX) for word in caption_prefix]
            prefix_ix = torch.tensor(prefix_ix).unsqueeze(0)

            logits = context.decoder.inference(img_vec, prefix_ix)[0, -1]
            probs = F.softmax(logits, -1).detach().numpy()
            
            probs = probs ** t / np.sum(probs ** t)
            next_token = np.random.choice(context.vocab.vocabulary, p=probs)
            caption_prefix.append(next_token)

            if next_token == context.vocab.EOS_TOKEN:
                break
            
    return ' '.join(caption_prefix[1:-1])


def generate_captions(image):
    image = np.array(Image.fromarray(image).resize((299,299))).astype('float32') / 255.

    captions = []
    for i in range(10):
        captions.append(generate_caption(context=init_model(), image=image, t=5.))

    return captions
