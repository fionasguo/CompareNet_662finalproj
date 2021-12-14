# Reproduction work of CompareNet

We reproduced the ACL 2021 paper "[Compare to The Knowledge: Graph Neural Fake News Detection with External Knowledge](https://aclanthology.org/2021.acl-long.62/)"


## This branch is for replacing LSTM to be GRU for the encoder. 



## Data

raw_data.zip: https://github.com/BUPT-GAMMA/CompareNet_FakeNewsDetection/releases/tag/dataset


Make sure the following files are present as per the directory structure before running the code,
```
FakeNewsDetection
├── README.md
├── *.py
└───models
|   └── *.py 
└───data
    ├── fakeNews
    │   ├── adjs
    │   │   ├── train
    │   │   ├── dev
    │   │   └── test
    │   ├── fulltrain.csv (LUN-train)
    │   ├── balancedtest.csv (LUN-test)
    │   ├── test.xlsx (SLN)
    │   ├── entityDescCorpus.pkl
    │   └── entity_feature_transE.pkl
    └── stopwords_en.txt
```

## Dependencies

We reproduce the work using NVIDIA Tesla V100 GPU, with the following packages installed:
```
python 3.7
torch 1.3.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
nltk
```

## Run

Preprocessing of data is automatically included in main.py.

To train and test,
```
python main.py --mode 0
```

To test only,
```
python main.py --mode 1 --model_file MODELNAME
```

We repeat all experiments with 5 seeds, add the following command line arguments:
```
python main.py --mode 0 --repeat 5 --seed 42 91 30 72 5
```

The following are all experiments we ran:

<img src="https://user-images.githubusercontent.com/44278097/145665176-6a936fd9-95f8-4838-b7b1-1aab0d8d077b.png" width="600" height="300">

## Results

The logs of all experiments are in the logs foler. 

The result_summary.xlsx is a spread sheet with all final scores aggregated.
