"""
Usage:
    vocab.py --train-src=<file> --train-tgt=<file> [options] VOCAB_FILE

Options:
    -h --help                  Show this screen.
    --train-src=<file>         File of training source sentences
    --train-tgt=<file>         File of training target sentences
    --size=<int>               vocab size [default: 50000]
    --freq-cutoff=<int>        frequency cutoff [default: 2]
"""

from collections import Counter
from docopt import docopt
from itertools import chain
import json
import torch
from typing import List
from utils import read_corpus, pad_sents, pad_sents_char


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self, word2id=None):
        """ Initialize a VocabEntry instance.
        :param dict word2id: Dictionary mapping words to indices.
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0  # Pad Token
            self.word2id['<s>'] = 1  # Start Token
            self.word2id['</s>'] = 2  # End Token
            self.word2id['<unk>'] = 3  # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.char_list = list(
            """AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŻŹaąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")

        self.char2id = dict()  # Converts characters to integers
        self.char2id['<pad>'] = 0
        self.char2id['{'] = 1
        self.char2id['}'] = 2
        self.char2id['<unk>'] = 3
        for i, c in enumerate(self.char_list):
            self.char2id[c] = len(self.char2id)
        self.char_unk = self.char2id['<unk>']
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]
        assert self.start_of_word + 1 == self.end_of_word

        self.id2char = {v: k for k, v in self.char2id.items()}  # Converts integers to characters

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        :param str word: Word to look up.
        :return int: Index of the word.
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if the word is captured by the VocabEntry.
        :param str word: Word to look up.
        :return bool: Whether the word is in the VocabEntry.
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in VocabEntry.
        :return int: Number of words in VocabEntry.
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        :param int wid: word index.
        :return str: word corresponding to index.

        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        :param str word: Word to add to VocabEntry.
        :return int: Index that the word has been assigned.
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2charindices(self, sents):
        """ Convert list of sentences of words into list of list of list of character indices.
        :param list[list[str]] sents: sentence(s) comprising of words.
        :return list[list[list[int]]]: sentence(s) at the character level.
        """
        return [[[self.start_of_word] + [self.char2id[char] for char in word] + [self.end_of_word]
                 for word in sent] for sent in sents]

    def words2indices(self, sents):
        """ Convert list of sentences of words into list of list of indices.
        :param list[list[str]] sents: sentence(s) comprising of words.
        :return list[list[int]] word_ids: sentence(s) comprising of word indices.
        """
        return [[self[w] for w in s] for s in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        :param list[int] word_ids: list of word indices.
        :return list[str]: list of words.
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor_char(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        :param List[List[str]] sents: list of sentences comprising of words.
        :param device: device on which to load the tensor, i.e. CPU or GPU.

        :return sents_var: padded tensor of sentences at the character level,
        shaped (max_sentence_length, batch_size, max_word_length).
        """
        char_indices = self.words2charindices(sents)
        sents_var = pad_sents_char(char_indices, self.char2id['<pad>'])
        sents_var = torch.tensor(sents_var, device=device)
        sents_var = sents_var.permute(1, 0, 2)

        return sents_var

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        :param List[List[str]] sents: list of sentences (words).
        :param device: device on which to load the tensor, i.e. CPU or GPU.

        :return: tensor of (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab Entry.
        :param list[str] corpus: corpus of text produced by read_corpus function
        :param int size: Number of words in vocabulary
        :param int freq_cutoff: if word occurs n < freq_cutoff times, drop the word
        :return VocabEntry: VocabEntry instance produced from provided corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """

    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        @param tgt_vocab (VocabEntry): VocabEntry for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents, tgt_sents, vocab_size, freq_cutoff) -> 'Vocab':
        """ Build Vocabulary.
        :param list[str] src_sents: Source sentences provided by read_corpus() function
        :param list[str] tgt_sents: Target sentences provided by read_corpus() function
        :param int vocab_size: Size of vocabulary for both source and target languages
        :param int freq_cutoff: if word occurs n < freq_cutoff times, drop the word.
        """
        assert len(src_sents) == len(tgt_sents)

        print('initialize source vocabulary ..')
        src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)

        print('initialize target vocabulary ..')
        tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)

        return Vocab(src, tgt)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        :param str file_path: file path to vocab file
        """
        json.dump(dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id),
                  open(file_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        :param str file_path: file path to vocab file
        :return Vocab: Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, 'r', encoding='utf-8'))
        src_word2id = entry['src_word2id']
        tgt_word2id = entry['tgt_word2id']

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocab(source %d words, target %d words)' % (len(self.src), len(self.tgt))


if __name__ == '__main__':
    args = docopt(__doc__)

    print('read in source sentences: %s' % args['--train-src'])
    print('read in target sentences: %s' % args['--train-tgt'])

    src_sents = read_corpus(args['--train-src'], source='src')
    tgt_sents = read_corpus(args['--train-tgt'], source='tgt')

    vocab = Vocab.build(src_sents, tgt_sents, int(args['--size']), int(args['--freq-cutoff']))
    print('generated vocabulary, source %d words, target %d words' % (len(vocab.src), len(vocab.tgt)))

    vocab.save(args['VOCAB_FILE'])
    print('vocabulary saved to %s' % args['VOCAB_FILE'])
