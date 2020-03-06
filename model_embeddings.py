import torch.nn as nn
import torch

from cnn import CNN
from highway import Highway


class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        :param int embed_size: Embedding size (dimensionality) for the output
        :param VocabEntry vocab: VocabEntry object. See vocab.py for documentation.
        """
        super().__init__()

        pad_token_idx = vocab.char2id['<pad>']
        self.embeddings = nn.Embedding(len(vocab.char2id), embedding_dim=50, padding_idx=pad_token_idx)
        self.cnn = CNN(in_channels=50, out_channels=embed_size)  # Hard-coded character embedding dimensionality = 50
        self.highway = Highway(embed_size)
        self.dropout = nn.Dropout(0.3)
        self.embed_size = embed_size  # Word embeddings size

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        :param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        :return: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        x_emb = self.embeddings(input)  # Dim: (sentence_length, batch_size, max_word_length (max_w_l), char_embed_size)
        x_emb = x_emb.permute(0, 1, 3, 2)  # Dim: (sentence_length, batch_size, char_embed_size, max_w_l)

        sent_len, batch_size, char_embed_size, max_w_l = x_emb.shape
        x_emb_unrolled = torch.reshape(x_emb, (sent_len * batch_size, char_embed_size, max_w_l))

        x_conv_out = self.cnn(x_emb_unrolled)  # Dim: (sentence length * batch size, word embedding size (e_word))
        x_highway = self.highway(x_conv_out)
        x_word_emb_unrolled = self.dropout(x_highway)
        x_word_emb = torch.reshape(x_word_emb_unrolled, (sent_len, batch_size, self.embed_size))
        return x_word_emb
