import json
import os
from urllib.request import urlopen
from collections import Counter


def preprocess_data():
    with urlopen('http://pelcra.clarin-pl.eu:9893/paraligner/index/segments?query=source:CORDIS&start=0&rows=50000') \
            as response:
        source_1 = response.read()

    with urlopen('http://pelcra.clarin-pl.eu:9893/paraligner/index/segments?query=source:CORDIS&start=50000&rows=50000') \
            as response:
        source_2 = response.read()

    data_1 = json.loads(source_1)
    data_2 = json.loads(source_2)
    data = {**data_1, **data_2}

    with open('./pl_en_data/pl_dirty.txt', 'w+', encoding='utf-8') as pl_dirty, \
            open('./pl_en_data/en_dirty.txt', 'w+', encoding='utf-8') as en_dirty:
        for key, val in data.items():
            pl_dirty.write(val['seg_pl_txt'].replace('\n', '') + '\n')
            en_dirty.write(val['seg_en_txt'].replace('\n', '') + '\n')

        pl_dirty.seek(0)
        pl_data = pl_dirty.read()
        total_nr_chars_pl = len(pl_data)

        en_dirty.seek(0)
        en_data = en_dirty.read()
        total_nr_chars_en = len(en_data)

        char_counts_pl = Counter(pl_data)
        char_counts_en = Counter(en_data)

        all_char_counts = {**char_counts_en, **char_counts_pl}
        all_chars = set(all_char_counts.keys())

        chars_to_keep = set(
            """AĄBCĆDEĘFGHIJKLŁMNŃOÓPQRSŚTUVWXYZŻŹaąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]""")

        chars_to_keep_from_counts = set([k for k, v in all_char_counts.items() if v > 1000 and k not in chars_to_keep])
        chars_to_keep.update(chars_to_keep_from_counts)

        chars_to_remove = all_chars - chars_to_keep
        print('Removing chars:', chars_to_remove)

        for char in chars_to_remove:
            if char in pl_data:
                pl_data = pl_data.replace(char, '')
            if char in en_data:
                en_data = en_data.replace(char, '')

        chars_pl = set(pl_data)
        chars_en = set(en_data)
        all_chars = chars_pl | chars_en
        assert all_chars <= chars_to_keep, 'Not all invalid characters removed.'

        pl_data = pl_data.split('\n')[:-1]
        en_data = en_data.split('\n')[:-1]
        assert len(pl_data) == len(en_data), 'Invalid alignment.'

        with open('./pl_en_data/pl_train.txt', 'w', encoding='utf-8') as pl_train, \
                open('./pl_en_data/en_train.txt', 'w', encoding='utf-8') as en_train, \
                open('./pl_en_data/pl_test.txt', 'w', encoding='utf-8') as pl_test, \
                open('./pl_en_data/en_test.txt', 'w', encoding='utf-8') as en_test, \
                open('./pl_en_data/pl_dev.txt', 'w', encoding='utf-8') as pl_dev, \
                open('./pl_en_data/en_dev.txt', 'w', encoding='utf-8') as en_dev:

            nr_segments = len(pl_data)
            train_size = 0.9 * nr_segments
            dev_size = 0.008 * nr_segments
            train_and_dev = train_size + dev_size

            for i, (line_pl, line_en) in enumerate(zip(pl_data, en_data)):
                if i < train_size:
                    pl_train.write(line_pl + '\n')
                    en_train.write(line_en + '\n')
                elif i < train_and_dev:
                    pl_dev.write(line_pl + '\n')
                    en_dev.write(line_en + '\n')
                else:
                    pl_test.write(line_pl + '\n')
                    en_test.write(line_en + '\n')

    os.remove('./pl_en_data/pl_dirty.txt')
    os.remove('./pl_en_data/en_dirty.txt')


if __name__ == '__main__':
    preprocess_data()
