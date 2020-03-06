from collections import namedtuple
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings
from char_decoder import CharDecoder

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model
    """

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, no_char_decoder=False):
        """ Initalize the NMT Model.

        :param int embed_size: Embedding size (dimensionality)
        :param int hidden_size: Hidden Size (dimensionality)
        :param Vocab vocab: Vocabulary object containing src and tgt languages
                             See vocab.py for documentation.
        :param float dropout_rate: Dropout probability, for the attention combination layer
        """
        super(NMT, self).__init__()

        self.model_embeddings_source = ModelEmbeddings(embed_size, vocab.src)
        self.model_embeddings_target = ModelEmbeddings(embed_size, vocab.tgt)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        # Need to feed in transpose of [h_enc(1)(<-) ; h_enc(m)(->)], and output is 1xh
        self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # Need to feed in transpose of [c_enc(1)(<-); c_enc(m)(->)], and output is 1xh
        self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        # Need to feed in transpose of u(t), and output is 1xh (v(t))
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)

        # Need to feed in transpose of o(t), and output is 1x|Vtg|
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        if not no_char_decoder:
            self.charDecoder = CharDecoder(hidden_size, target_vocab=vocab.tgt)
        else:
            self.charDecoder = None

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        :param List[List[str]] source: list of source sentence tokens
        :param List[List[str]] target: list of target sentence tokens, wrapped by `<s>` and `</s>`

        :return Tensor: a variable/tensor of shape (b, ) representing the
                        log-likelihood of generating the gold-standard target sentence for
                        each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)
        target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device=self.device)

        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded_chars)

        # Compute the softmax scores for all hidden states from the decoder (all in the batch, including masked ones)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text (we get zeros for pad tokens)
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words (ignoring the start token)
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(
            -1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum()

        if self.charDecoder is not None:
            max_word_len = target_padded_chars.shape[-1]

            target_words = target_padded[1:].contiguous().view(-1)
            target_chars = target_padded_chars[1:].reshape(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, 256)

            target_chars_oov = target_chars  # torch.index_select(target_chars, dim=0, index=oovIndices)
            rnn_states_oov = target_outputs  # torch.index_select(target_outputs, dim=0, index=oovIndices)
            oovs_losses = self.charDecoder.train_forward(target_chars_oov.t(),
                                                         (rnn_states_oov.unsqueeze(0), rnn_states_oov.unsqueeze(0)))
            scores = scores - oovs_losses

        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.
        :param Tensor source_padded: Tensor of padded source sentences with shape (src_len, b, max_word_length), where
                                        b = batch_size, src_len = maximum source sentence length (already sorted in
                                         order of longest to shortest sentence).
        :param List[int] source_lengths: List of actual lengths for each of the source sentences in the batch.
        :return Tensor: Tensor of hidden units with shape (b, src_len, h*2), where
                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        :return tuple(Tensor, Tensor): Tuple of tensors representing the decoder's initial
                                        hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        X = self.model_embeddings_source(source_padded)
        X = nn.utils.rnn.pack_padded_sequence(X, source_lengths)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        enc_hiddens, _ = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)

        init_decoder_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        init_decoder_hidden = self.h_projection(init_decoder_hidden)

        init_decoder_cell = torch.cat((last_cell[0], last_cell[1]), 1)
        init_decoder_cell = self.c_projection(init_decoder_cell)
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.
        :param Tensor enc_hiddens: Hidden states (b, src_len, h*2), where
                                    b = batch size, src_len = maximum source sentence length, h = hidden size.
        :param Tensor enc_masks: Tensor of sentence masks (b, src_len), where
                                  b = batch size, src_len = maximum source sentence length.
        :param tuple(Tensor, Tensor) dec_init_state: Initial state and cell for decoder
        :param Tensor target_padded: Gold-standard padded target sentences (tgt_len, b, max_word_length), where
                                     tgt_len = maximum target sentence length, b = batch size.
        :return Tensor: combined output tensor  (tgt_len, b,  h), where
                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Remove the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step (output of each decoder step)
        combined_outputs = []

        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_embeddings_target(target_padded)
        for Y_t in torch.split(Y, 1, 0):
            Y_t = torch.squeeze(Y_t, dim=0)
            Ybar_t = torch.cat((Y_t, o_prev), dim=1)
            dec_state, o_t, _ = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t

        combined_outputs = torch.stack(combined_outputs, dim=0)
        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.
        :param Tensor Ybar_t: Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                              where b = batch size, e = embedding size, h = hidden size.
        :param tuple(Tensor, Tensor) dec_state: Tuple of tensors both with shape (b, h), where b = batch size,
                                                h = hidden size. First tensor is decoder's prev hidden state,
                                                second tensor is decoder's prev cell.
        :param Tensor enc_hiddens: Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                   src_len = maximum source length, h = hidden size.
        :param Tensor enc_hiddens_proj: Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is of shape
                                        (b, src_len, h), where b = batch size, src_len = maximum source length,
                                        h = hidden size.
        :param Tensor enc_masks: Tensor of sentence masks shape (b, src_len),
                                 where b = batch size, src_len is maximum source length.
        :return tuple(Tensor, Tensor): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                                       First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        :return Tensor: Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        :return Tensor: Tensor of shape (b, src_len). It is attention scores distribution.
        """

        combined_output = None

        dec_state = self.decoder(Ybar_t, dec_state)
        dec_hidden, dec_cell = dec_state
        batch2 = torch.unsqueeze(dec_hidden, 2)
        e_t = torch.bmm(enc_hiddens_proj, batch2)
        e_t = torch.squeeze(e_t, dim=2)

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        alpha_t = nn.functional.softmax(e_t, 1)
        alpha_t = torch.unsqueeze(alpha_t, dim=1)
        a_t = torch.bmm(alpha_t, enc_hiddens)
        a_t = torch.squeeze(a_t, dim=1)

        U_t = torch.cat((a_t, dec_hidden), dim=1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        :param Tensor enc_hiddens: encodings of shape (b, src_len, 2*h), where b = batch size,
                                    src_len = max source length, h = hidden size.
        :param List[int] source_lengths: List of actual lengths for each of the sentences in the batch.

        :return Tensor: Tensor of sentence masks of shape (b, src_len),
                        where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[
        Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        :param List[str] src_sent: a single source sentence (words)
        :param int beam_size: beam size
        :param int max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        :return List[Hypothesis]: a list of hypothesis, each hypothesis has two fields:
                                  value List[str]: the decoded target sentence, represented as a list of words
                                  score float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor_char([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = self.vocab.tgt.to_input_tensor_char(list([hyp[-1]] for hyp in hypotheses), device=self.device)
            y_t_embed = self.model_embeddings_target(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            decoder_states_for_unks = []
            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]

                # Record output layer in case UNK was generated
                if hyp_word == "<unk>":
                    hyp_word = "<unk>" + str(len(decoder_states_for_unks))
                    decoder_states_for_unks.append(att_t[prev_hyp_id])

                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(decoder_states_for_unks) > 0 and self.charDecoder is not None:  # decode UNKs
                decoder_states_for_unks = torch.stack(decoder_states_for_unks, dim=0)
                decoded_words = self.charDecoder.decode_greedy((decoder_states_for_unks.unsqueeze(0),
                                                                decoder_states_for_unks.unsqueeze(0)),
                                                               max_length=21, device=self.device)
                assert len(decoded_words) == decoder_states_for_unks.size()[0], "Incorrect number of decoded words"
                for hyp in new_hypotheses:
                    if hyp[-1].startswith("<unk>"):
                        hyp[-1] = decoded_words[int(hyp[-1][5:])]

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.att_projection.weight.device

    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        :param str model_path: path to model
        :param boolean no_char_decoder: whether the char-level decoder is also used
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], no_char_decoder=no_char_decoder, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the model to a file.
        :param str path: path to the model parameters
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings_source.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
