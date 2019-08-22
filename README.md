# Using BERT for social media topic clustering and analysis 

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
Import the following packages into the python workspace as well as the pre-trained bert model: 
```python
!pip install pytorch-pretrained-bert # use !pip in jupyter notebook or ust pip in linux
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

* The model is trained using BERT on the twitter paraphrase corpus (Twitter_Corpus_train.csv).
* The data needs to be prepared into a tsv as follows: 

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
### Set classes for input data and features 
```python 
class InputExample(object):
    """A single training/test example for sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels (output paraphrase indicator) for this data set."""
        raise NotImplementedError()
    
    #read in data
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
```
### Build class to handle binary classification 
```python
class BinaryClassificationProcessor(DataProcessor):
    """Processor for binary classification dataset."""

    #load training tsv    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    #load testing tsv (should be the appropriate labeled set for evaluation)
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
        
    #pull out text a, textb, abd label into a list object for each pair of text
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=label))
        return examples
     
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
            
def convert_example_to_feature(example_row): #prepare data object for BERT model
    # return example_row
    example, label_map, max_seq_length, tokenizer, output_mode = example_row

    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if output_mode == "classification":
        label_ids = label_map[example.labels]
    elif output_mode == "regression":
        label_ids = float(example.label)
    else:
        raise KeyError(output_mode)

    return InputFeatures(input_ids=input_ids,
                         input_mask=input_mask,
                         segment_ids=segment_ids,
                         label_ids=label_ids)
```

### Set directories and model parameters 
* DATA_DIR is where the tsv data is stored 
* BERT_MODEL can take on varying forms and sizes. Bert-large-uncased ignores cases and has 24 layers (as compared to bert base which has 12 layers and considers case)

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
    
### Set training parameters and load pre-trained model tokenizer
```python
# Training
# The maximum total input sequence length after WordPiece tokenization.
# Sequences longer than this will be truncated, and sequences shorter than this will be padded.

num_labels = len(label_list)

# Set training steps based on the number of epochs
num_train_optimization_steps = int(train_examples_len / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
MAX_SEQ_LENGTH = 512 #maximum number of characters BERT can accept 

TRAIN_BATCH_SIZE = 24 #takes 24 lines at a time in training
EVAL_BATCH_SIZE = 32 #takes 32 lines at a time in testing
LEARNING_RATE = 2e-5 
NUM_TRAIN_EPOCHS = 3
RANDOM_SEED = 42
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_PROPORTION = 0.1

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
```

## Run data through above functions to prepare for BERT
```python
processor = BinaryClassificationProcessor()

# Get the training examples
train_examples = processor.get_train_examples(DATA_DIR)
train_examples_len = len(train_examples)

# Get the label list
label_list = processor.get_labels()  # [0, 1] for binary classification
label_map = {label: i for i, label in enumerate(label_list)}
train_examples_for_processing = [(example, label_map, MAX_SEQ_LENGTH, tokenizer, OUTPUT_MODE) for example in train_examples]

# Use muiltiprocessing to process all the examples
process_count = multiprocessing.cpu_count()
print(f'Preparing to convert {train_examples_len} examples..')
print(f'Spawning {process_count} processes..')
pool = multiprocessing.Pool(processes=process_count)
train_features = pool.map(convert_example_to_feature, train_examples_for_processing)
pool.close()
pool.join()
pool.terminate()
#Save BERT ready data with Pickle
with open(DATA_DIR + "train_features.pkl", "wb") as f:
    pickle.dump(train_features, f)
```
### Load pre-trained BERT model and set optimizer  
```python
# Load pre-trained model weights
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, cache_dir=CACHE_DIR, num_labels=num_labels)
model.to(device)

## Set optimization parameters for backpropagation
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=LEARNING_RATE,
                     warmup=WARMUP_PROPORTION,
                     t_total=num_train_optimization_steps)

global_step = 0
nb_tr_steps = 0
tr_loss = 0
```

### Set logging to follow training evalutation
```python
####Log training performance and evaluation metrics####
logger.info("***** Running training *****")
logger.info("  Num examples = %d", train_examples_len)
logger.info("  Batch size = %d", TRAIN_BATCH_SIZE)
```
### Begin model training 
```python
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

if OUTPUT_MODE == "classification":
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
elif OUTPUT_MODE == "regression":
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)

train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=TRAIN_BATCH_SIZE)
model.train()
for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm_notebook(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        logits = model(input_ids, segment_ids, input_mask, labels=None)

        if OUTPUT_MODE == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif OUTPUT_MODE == "regression":
            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / GRADIENT_ACCUMULATION_STEPS
## Backpropagation 
        loss.backward()
        print("\r%f" % loss, end='')

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

# If we save using the predefined names, we can load using `from_pretrained`
output_model_file = os.path.join(OUTPUT_DIR, WEIGHTS_NAME)
output_config_file = os.path.join(OUTPUT_DIR, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(OUTPUT_DIR)
###Trained model will be in Cache Folder
```

## Model Evaluating with Test set (Twitter_Corpus_test.csv)
* Model evaluation using the twitter paraphrase corpus test set which is already labeled follows similar method above, using all functions from above. 
* Adjust the files as shown in BertEval.py 
* Make sure BERT_MODEL is the name of the trained model
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
### Load pre-trained model (see code above or BertEval.py)

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

* Without a test set the evaluation is the same as above, except there are no performance statistics to calculate. 
* Instead the probability of classification is used for each sentence pair. 
* The probabilities (positives_raw) are then used as a distance metric for further downstream analysis such as cluster or network analyses. 

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
1) Nodes = actual tweets (given by an ID number)
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
