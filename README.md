# NLP SocialMedia

## Motivation
Summarize text from social media (or other sources) into an interactive network graph
![Network Screen shot using college, diet, and anxiety tweets](https://github.com/KellyCahill/NLP_SocialMedia/blob/master/network%20graph%20Twitter/network_screenshot.PNG?raw=true)

## Bert Background 
* BERT (Bidirectional Encoding Representations from Transformers) is a recent huge and complex deep learning model developed at Google
* Previous unsupervised NLP models (LDA/TF-IDF) work well in matching text according to keyword searches where the key words are identical in each sentence
* BERT is able to capture the full context of the entire sequence of words that precede and follow a word in a sentence(bidirectional context)

## Modeling approach
* Twitter paraphrase corpus was used to train the model. Paraphrasing is a difficult task that allows for a conservative prediction in a testing set. 
* When applying new unlabeled text pairs to the model, we can use the probability of of paraphrased as a distance metric or similarity score for network building. 
![Methods](https://github.com/KellyCahill/NLP_SocialMedia/blob/master/img/steps.PNG)
![paraphrase example](https://github.com/KellyCahill/NLP_SocialMedia/blob/master/img/dataex.PNG)
## Model Training 
### Required imports
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

### Data prep

The model is trained using BERT on the twitter paraphrase corpus (Twitter_Corpus_train.csv). The data needs to be prepared into a tsv as follows: 

```python
train_df = pd.read_csv('data/Twitter_Corpus_train.csv', header=None)
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

Set Logging so that the user can follow along with the training process: 
```python
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
### Load all classes and functions found in the BertTrain.py script (lines 27-178) 

### Set directories and model parameters 
DATA_DIR is where the tsv data is stored 
BERT_MODEL can take on varying forms and sizes. Bert-large-uncased ignores cases and has 24 layers (as compared to bert base which has 12 layers)

```python
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
BERT_MODEL = 'bert-large-uncased'

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
MAX_SEQ_LENGTH = 512 #max number of characters 

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
### Train model and output epoch statistics (BertTrain.py lines 258-end)

## Model Evaluating with Test set (Twitter_Corpus_test.csv)
Model evaluation using the twitter paraphrase corpus test set which is already labeled follows similar method above, using all functions from above. Adjust the files as shown in BertEval.py
```python
# The input data dir. Should contain the .tsv files (or other data files) for the task.
DATA_DIR = "data/"

# Bert pre-trained model selected in the list: bert-base-uncased,
# bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased,
# bert-base-multilingual-cased, bert-base-chinese.
#SHOULD BE THE NAME OF THE TRAINED MODEL
BERT_MODEL = 'twitter_model.tar.gz'

# The name of the task to train.I'm going to name this 'yelp'.
TASK_NAME = 'text_similarity'

# The output directory where the fine-tuned model and checkpoints will be written.
OUTPUT_DIR = f'outputs/{TASK_NAME}/'

# The directory where the evaluation reports will be written to.
REPORTS_DIR = f'reports/{TASK_NAME}_evaluation_report/'

# This is where BERT will look for pre-trained models to load parameters from.
CACHE_DIR = 'cache/'

# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.
MAX_SEQ_LENGTH = 512

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1
OUTPUT_MODE = 'classification'

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"

dev_filename = "dev.tsv"
```
### Get model evaluation statistics (AUC, MCC, etc...) need functions: 
```python 
def get_eval_report(task_name, labels, preds, positives):
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    auc = roc_auc_score(labels,positives)
    accuracy = accuracy_score(labels,preds)
    f1 = f1_score(labels,preds)
    return {
        "task": task_name,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "auc": auc,
        'accuracy':accuracy,
        'f1':f1,
    }


def compute_metrics(task_name, labels, preds, positives):
    assert len(preds) == len(labels)
    assert len(positives) == len(labels)
    return get_eval_report(task_name, labels, preds, positives)
```

### Load pretrained model (BertEval.py lines 284 - 310) 

### Run prediction on labeled test set
```python
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
model.to(device)
model.eval()

eval_loss = 0
nb_eval_steps = 0
preds = []
positives = []
positives_raw =[]
for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    if OUTPUT_MODE == "classification":
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    elif OUTPUT_MODE == "regression":
        loss_fct = MSELoss()
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
    
    positives.append(logits.detach().cpu().numpy()[0][1])
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    positives_raw.append(probabilities.detach().cpu().numpy()[0][1])
```

### Output model evalutation statistics: 
```python 
eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
if OUTPUT_MODE == "classification":
    preds = np.argmax(preds, axis=1)
elif OUTPUT_MODE == "regression":
    preds = np.squeeze(preds)
result = compute_metrics(TASK_NAME, all_label_ids.numpy(), preds, positives)
result['eval_loss'] = eval_loss
output_eval_file = os.path.join(REPORTS_DIR, "eval_results.txt")
with open(output_eval_file, "w") as writer:
    logger.info("***** Eval results *****")
    for key in (result.keys()):
        logger.info("  %s = %s", key, str(r
        esult[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
```

## Model Evaluation without test set

Without a test set the evaluation is the same as above, except there are no performance statistics to calculate. Instead the probability of classification is used for each sentence pair. The probabilities (\bold{positives_raw}) are then used as a distance metric for further downstream analysis such as cluster or network analyses. 

```python
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(CACHE_DIR + BERT_MODEL, cache_dir=CACHE_DIR, num_labels=len(label_list))
model.to(device)
model.eval()

eval_loss = 0
nb_eval_steps = 0
preds = []
positives = []
positives_raw =[]
for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    if OUTPUT_MODE == "classification":
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    elif OUTPUT_MODE == "regression":
        loss_fct = MSELoss()
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
    
    positives.append(logits.detach().cpu().numpy()[0][1])
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    positives_raw.append(probabilities.detach().cpu().numpy()[0][1]) ###Probabilities to be used for further analyses 
```
## Visualization
An interactive network graph of social media topics is located in the network graph twitter folder. The data needs to prepared as a JSON file. We use d3.js to build the network. 

### JSON data example (full data found in network graph twitter folder):
1) Nodes= actual tweets (given by an ID number)
2) group is pre-determined clustering group (optional, can set to 1)
3) body is tweet text to be shown when shift is held and curser is dragged over node
4) Source is the source node (by id) 
5) target  the target node (by id). Not all pairwise links need to be included
```javascript
var data = {
  "nodes": [
 {
   "id": 21,
   "group": 1,
   "body": "@oseseo same, its helped w my anxiety a lot. &amp; i was generally eating pretty healthy so i havent changed too much of my diet. did you?"
 },
 {
   "id": 22,
   "group": 1,
   "body": "@mynameis152 ugh bby you should see a GI if you can! there are ways you adjust your diet based on what a doctor says.. i have anxiety induced ibs but after i saw a doctor it rly helped"
 }
 ],
 "links":[
 {
   "source": 21,
   "target": 59,
   "value": 0.35208
 },
 {
   "source": 22,
   "target": 67,
   "value": 0.41684
 }
 ]
 }
 
```
