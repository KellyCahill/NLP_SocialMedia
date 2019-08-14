# NLP SocialMedia

## Motivation
summarize topics of social media data 

## Model Training 
Import the following packages into the python workspace: 
```python
##!pip install pytorch-pretrained-bert
import pandas as pd
import torch

import csv
import os
import sys
import logging

import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm_notebook, trange
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
import multiprocessing
```

The model is trained using BERT on the twitter paraphrase corpus (Twitter_Corpus_train.csv). The data needs to be prepared into a tsv as follows: 

```python
train_df = pd.read_csv('data/train.csv', header=None)
# convert to BERT friendly structure
train_df_bert = pd.DataFrame({
    'id': range(len(train_df)),
    'label': train_df[0],
    'alpha': ['a'] * train_df.shape[0],
    'text_a': train_df[1].replace(r'\n', ' ', regex=True),
    'text_b': train_df[1].replace(r'\n', ' ', regex=True)
})

train_df_bert.to_csv('data/train.tsv', sep='\t', index=False, header=False)
```

Set Logging: 
```python
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Load all classes and functions found in the BertTrain.py script (lines 27-178) 

Set directories and model parameters: 

```python
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-base-cased'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'twitter'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

OUTPUT_MODE = 'classification'
output_mode = OUTPUT_MODE
cache_dir = CACHE_DIR

if os.path.exists(REPORTS_DIR) and os.listdir(REPORTS_DIR):
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)
    REPORTS_DIR += f'/report_{len(os.listdir(REPORTS_DIR))}'
    os.makedirs(REPORTS_DIR)

if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
    raise ValueError("Output directory ({}) already exists and is not empty.".format(OUTPUT_DIR))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

num_labels = len(label_list)

# Set training steps based on the number of epochs
num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
MAX_SEQ_LENGTH = 128

TRAIN_BATCH_SIZE = 24
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
```
Train model and output epoch statistics (BertTrain.py lines 258-end)
## Model Evaluating 

## Visualization
