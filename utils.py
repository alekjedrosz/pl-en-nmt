import math

import numpy as np


def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    :param list[list[list[int]]] sents: list of sentences, result of `words2charindices()`
        from `vocab.py`
    :param int char_pad_token: index of the character-padding token
    :return list[list[list[int]]]: list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    max_sent_length = max([len(sent) for sent in sents])
    padding_word = [char_pad_token] * max_word_length

    sents_padded = [[sent[i] if i < len(sent) else padding_word for i in range(max_sent_length)] for sent in sents]
    sents_padded = [[[word[i] if i < len(word) else char_pad_token for i in range(max_word_length)]
                    for word in sent] for sent in sents_padded]

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    :param list[list[int]] sents: list of sentences, where each sentence
                                    is represented as a list of words
    :param int pad_token: padding token
    :return list[list[int]]: list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_length = max([len(sent) for sent in sents])
    sents_padded = [[sent[i] if i < len(sent) else pad_token for i in range(max_length)] for sent in sents]

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is delineated by a `\n`.
    :param str file_path: path to file containing corpus
    :param str source: "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path, encoding='utf-8'):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    :param list((src_sent, tgt_sent)) data: list of tuples containing source and target sentence
    :param int batch_size: batch size
    :param boolean shuffle : whether to randomly shuffle the data set
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
