# NLP_Transformer

This is a Transformer based implementation of the German-English translator in pyTorch.

This script was developed using the implementation in https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec. Hence, big credit goes to Samuel Lynn-Evans.

Few corrections mentioned in the comments sections (for e.g. positional encodings) of the medium article are implemented in this code.

The repository contains mainly two scripts:
1) TorchText_Spacy.ipynb : A jupyter notebook about basic spacy (language )library usage
2) De_En_translate.py : Transformer implementation of the translate function

The dataset used to train the network is from the "European Parliament Proceedings Parallel Corpus" available at http://www.statmt.org/europarl/
