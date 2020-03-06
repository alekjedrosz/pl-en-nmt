import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Initialise the Character Decoder.

        :param int hidden_size: Hidden size of the decoder LSTM
        :param int char_embedding_size: dimensionality of character embeddings
        :param VocabEntry target_vocab: vocabulary for the target language. See vocab.py for documentation.
        """
        super().__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.target_vocab = target_vocab
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        pad_token_idx = target_vocab.char2id['<pad>']
        self.decoder_char_emb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx=pad_token_idx)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder at inference time.

        :param Tensor input: tensor of integers; shape (seq_len, batch)
        :param Tensor dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two
                                  tensors of shape (1, batch, hidden_size) (i.e. the initial hidden and cell states h_0,
                                  c_0, by default set to None and therefore initialized as zeros).
        :return Tensor: scores of each hiddden state of the decoder; to be passed through a softmax to obtain a
                        probability distribution over characters; shape (seq_length, batch, char_vocab_size)
        :return Tensor: internal state of the LSTM after reading the input characters. A tuple of two
        tensors of shape (1, batch, hidden_size) (i.e. h_t, c_t for the last time step for every example in the batch)
        """
        input_embeddings = self.decoder_char_emb(input)  # Dim: (seq_len, batch, embedding_size)

        # ''dec_hidden'' is a tuple of (h_t, c_t), where t is the last time step for every sequence (word) in the batch,
        # whereas ''output'' contains all of the intermediate hidden states h_t (on each time step t).
        output, dec_hidden = self.charDecoder(input_embeddings, dec_hidden)  # output dim: (seq_len, batch, hidden_size)

        scores = self.char_output_projection(output)

        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        :param Tensor char_sequence: tensor of integers, shape (seq_length, batch)
        :param tuple(Tensor, Tensor) dec_hidden: initial internal state of the LSTM, obtained from the output of the
                                                 word-level decoder. A tuple of two tensors of
                                                 shape (1, batch, hidden_size)
        :return float: The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the
                       words in the batch.
        """
        # Removing <s> and </s> tokens appropriately
        target_sequence = char_sequence[1:]
        source_sequence = char_sequence[:-1]

        scores, dec_hidden = self.forward(source_sequence, dec_hidden)

        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        scores = scores.permute(1, 2, 0)
        target_sequence = torch.t(target_sequence)

        return loss(scores, target_sequence)

    def decode_greedy(self, initial_states, device, max_length=21):
        """ Greedy decoding
        :param tuple(Tensor, Tensor) initial_states: initial internal state of the LSTM, a tuple of two tensors
                                                     of size (1, batch, hidden_size)
        :param torch.device device: indicates whether the model is on CPU or GPU
        :param int max_length: maximum length of the words in the batch to decode
        :return List[str]: a list (of length batch) of strings, each of which has length <= max_length.
        """
        _, batch_size, _ = initial_states[0].shape
        current_chars = torch.full((1, batch_size), self.target_vocab.start_of_word, device=device, dtype=torch.long)
        output_words = ['' for _ in range(batch_size)]
        dec_hidden = initial_states
        for t in range(max_length):
            scores, dec_hidden = self.forward(current_chars, dec_hidden)  # scores dim: (1, batch, char_vocab_size)
            probability_dist = self.softmax(scores)
            current_chars = torch.argmax(probability_dist, dim=2)  # current_chars dim: (1, batch), char indices

            char_indices = [element.item() for element in current_chars.flatten()]
            output_words = [word + self.target_vocab.id2char[char_index]
                            for word, char_index in zip(output_words, char_indices)]
        output_words = [word.split(self.target_vocab.id2char[self.target_vocab.end_of_word])[0] for word in output_words]
        return output_words
