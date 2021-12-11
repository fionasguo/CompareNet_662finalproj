# Improve the Model: replacing LSTM encoder with sentence-BERT

Additional Experiments in the reproducibility work of the paper Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge (ACL 2021).

In the original paper, the news text (TextEncoder) and entity description (EntityEncoder) encoder uses LSTM for representation learning. We replace it with pretrained sentence-BERT.

## The following files have modifications made by us:

(The modifications are boxed and labeled by "Modified by Fiona Guo")

data_loader.py - the input data structure needs to be changed so it specifically fits sentence-BERT encoder.

models/classifier.py - this file defines the overall model structure. Whenever text and entity encoders are called, we replace with the BERT version encoders.

models/model.py - this file includes major modules in the model. We change the text encoder and entity encoder to using BERT.

models/layers.py - this file includes layers in the NN, we add the BERT encoder here.

## Run Command
python main.py --mode 0 --bert_encoder 1

## Result

We tested the improved model for the in-domain 4-way classification task, with different seeds.

The results show that using sentence-BERT significantly improves the model.

see file bert_encoder.log for recorded F1 scores of each repeat.
