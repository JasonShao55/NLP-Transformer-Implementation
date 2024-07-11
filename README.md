# Transformer Implementation

## Overview
This project is completed as a self-chosen research assignment for the CSE 256 NLP course (Spring 2024) at the University of California, San Diego. The goal is to experiment with different parts (i.e. the encoder and the decoder) of the transformer architecture from scratch ***without using any existing transformer-related libraries (such as nn.MultiheadAttention)*** and complete the following tasks:

- Part 1. **Encoder with Classifier**: Implement a transformer encoder and train it jointly with a feedforward classifier for a downstream task of predicting the speaker of a given speech segment.
- Part 2. **Decoder for Language Modeling**: Implement a GPT-like transformer decoder, pretrain it on an autoregressive language modeling task, and report perplexity on speeches from different politicians.
- Part 3. **Exploration**: Experiment with different parts of the architecture, such as positional encoding and sparse attention patterns, to improve the performance of the classifier or the decoder.

## Directory Structure
```
PA2_code/
│
├── speechesdataset/
│ ├── test_CLS.tsv
│ ├── test_LM_hbush.txt
│ ├── test_LM_obama.txt
│ ├── test_LM_wbush.txt
│ ├── train_CLS.tsv
│ └── train_LM.txt
│
├── Attention maps (PDF)
│
├── dataset.py
├── main.py
├── tokenizer.py
├── transformer.py
└── utilities.py
```


## Instructions to Run the Code

### Prerequisites
Ensure you have the following libraries installed:
- Python 3.7+
- PyTorch
- NLTK
- NumPy
- Matplotlib


### Running the Code
**Part 1: Encoder with Classifier**
```bash
python main.py part1
```

**Part 2: Decoder for Language Modeling**
Choose the intended test set by changing the input path before running
```bash
python main.py part2
```

**Part 3: Exploration**
```bash
python main.py part3
```


### Code Implementation
Model initialization with hyperparameters, optimizer initialization, pretraining and evaluation are completed in `main.py`.  

The implementation of the required tranformer and improved models are completed in `transformer.py`.








