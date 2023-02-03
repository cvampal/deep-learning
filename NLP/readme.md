# Readme

Here we use [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network) to do [POS Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging) of Hindi sentences. [Pytorch](https://pytorch.org/) deeplearning library is used for this purpose.


## Dataset

We use **Hindi Dependency Treebank (HDTB)** dataset. It can be downloaded from [here](http://ltrc.iiit.ac.in/treebank_H2014/index.html).

## Tips to play with code

- Make sure that data is in correct format. (see data/ dir.)
- First run ``build_vocab.py`` that will create ``tag.txt`` and ``word.txt`` files.
- Then run ``train.py`` followed by ``evaluate.py``.
-  You can play with different models e.g. LSTM , GRU in ``model/net.py`` file.
- Don't use this tip.

##### Credit: [CS230 Deep Learning](https://github.com/cs230-stanford/cs230-code-examples)
