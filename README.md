# Polish-English character-level Neural Machine Translation system
A sequence-to-sequence (seq2seq) machine translation model with global attention. Uses a character-level convolutional encoder to create word embeddings, thus capturing the complex morphology of the Polish language better and enabling informally spelled words (e.g. social media jargon) to be represented. Additionally, uses a character-level LSTM decoder for out-of-vocabulary words, allowing transliteration and rare-word reconstruction.

## Installing dependencies
This program is written in Python 3.7. Please use [pip](https://pip.pypa.io/en/stable/) package manager to install the necessary dependencies (it is recommended to install them in a virtual environment like venv).
It is best to execute the following commands sequentially.


Pytorch installation, together with its required dependencies:
```bash
pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Installing the other requirements:
```bash
pip install -r requirements.txt
```

## Retrieve and preprocess data:
```bash
python preprocessing.py
```
## Vocabulary generation:
```bash
python vocab.py --train-src=./pl_en_data/pl_train.txt --train-tgt=./pl_en_data/en_train.txt vocab.json
```

## Running tests (remove --cuda if no GPU is available; requires pre-trained parameters):
```bash
python run.py decode model.bin ./pl_en_data/pl_test.txt ./pl_en_data/en_test.txt outputs/test_outputs.txt --cuda
```

## Training the model:
```bash
python run.py train --train-src=./pl_en_data/pl_train.txt --train-tgt=./pl_en_data/en_train.txt --dev-src=./pl_en_data/pl_dev.txt --dev-tgt=./pl_en_data/en_dev.txt --vocab=vocab.json --cuda
```


Note: This code is in part adapted from course assignments. 