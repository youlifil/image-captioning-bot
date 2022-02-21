from image_captioning_bot.files import bot_path, download_google_file
import torch
import torch.nn as nn
import image_captioning_bot.log as log


class CaptionAttentionNet(nn.Module):
    def __init__(self, vocab_dim, cnn_feature_size=2048, emb_dim=256, lstm_dim=256, dropout=0):
        super(self.__class__, self).__init__()

        self.lstm_dim = lstm_dim

        self.init_h1 = nn.Linear(cnn_feature_size, lstm_dim)
        self.init_c1 = nn.Linear(cnn_feature_size, lstm_dim)

        self.init_h2 = nn.Linear(cnn_feature_size, lstm_dim)
        self.init_c2 = nn.Linear(cnn_feature_size, lstm_dim)

        self.embedding = nn.Embedding(vocab_dim, emb_dim)

        self.emb_dropout = nn.Dropout(p=dropout)
        self.lstm_dropout = nn.Dropout(p=dropout)

        self.lstm1 = nn.LSTM(emb_dim, lstm_dim, 1, batch_first=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, lstm_dim, 1, batch_first=True)

        self.attention1 = nn.MultiheadAttention(lstm_dim, num_heads=1, batch_first=True)
        self.attention2 = nn.MultiheadAttention(lstm_dim, num_heads=1, batch_first=True)

        self.logits = nn.Linear(lstm_dim * 3, vocab_dim)


    def forward(self, image_vectors, captions_ix):
        h1 = self.init_h1(image_vectors).unsqueeze(0)
        c1 = self.init_c1(image_vectors).unsqueeze(0)

        h2 = self.init_h2(image_vectors).unsqueeze(0)
        c2 = self.init_c2(image_vectors).unsqueeze(0)

        emb = self.embedding(captions_ix) 
        emb = self.emb_dropout(emb)

        outputs = []

        prefix1 = None
        prefix2 = None

        for t in range(emb.size(1)):
            input = emb[:, t, :].unsqueeze(1)

            output, (h1, c1) = self.lstm1(input, (h1, c1))

            output = self.lstm_dropout(output)

            key = h1.permute(1,0,2)
            if prefix1 is None:
                prefix1 = key
            else:
                prefix1 = torch.cat((prefix1, key), dim=1)

            attn1, _ = self.attention1(query=prefix1, key=key, value=key, need_weights=False)
            attn1 = attn1[:,-1,:].unsqueeze(1)
            output = torch.cat((output, attn1), dim=2)

            output, (h2, c2) = self.lstm2(output, (h2, c2))

            key = h2.permute(1,0,2)
            if prefix2 is None:
                prefix2 = key
            else:
                prefix2 = torch.cat((prefix2, key), dim=1)


            attn2, _ = self.attention2(query=prefix2, key=key, value=key, need_weights=False)
            attn2 = attn2[:,-1,:].unsqueeze(1)
            output = torch.cat((output, attn1, attn2), dim=2)

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1)
        logits = self.logits(outputs)

        return logits


class Decoder():
    def __init__(self, vocab_dim):
        logger = log.logger()

        logger.info("Making CaptionAttentionNet model...")
        self.model = CaptionAttentionNet(vocab_dim, lstm_dim=512, emb_dim=256, dropout=0.3)

        logger.info("Loading pretrained weights for CaptionAttentionNet")
        weights_file = bot_path("attnet-250e-30trainbatch.pt")
        weights_url = 'https://drive.google.com/uc?id=1pvyVjbwg845rqx_ERZ9uJ7AJ7fgQgz_1'
        download_google_file(weights_url, weights_file)
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def inference(self, image_vector, caption_prefix):
        return self.model(image_vector, caption_prefix)