#!/usr/bin/env python
# coding: utf-8

# In[40]:


import warnings
import os
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag
import warnings
import pandas
from string import punctuation, digits
import numpy as np
from lambeq import BobcatParser, Rewriter
from lambeq import remove_cups
from lambeq import AtomicType, IQPAnsatz
from itertools import chain

from pytket.extensions.qiskit import AerBackend
from lambeq import TketModel
from lambeq import BinaryCrossEntropyLoss
from lambeq import QuantumTrainer, SPSAOptimizer

warnings.filterwarnings("ignore")
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from lambeq import Dataset


# In[41]:


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in tagged])
    return lemmatized_text

def single_content2data(sen):
    text=lemmatize_text(sen)
    stop_words = set(stopwords.words('english'))
    sentences_0 = text
    sentences_1 = re.sub('<[^<]+?>', '', sentences_0)
    table = str.maketrans('', '', punctuation + digits)
    sentences_2 = sentences_1.translate(table)
    word_tokens = word_tokenize(sentences_2)
    filtered_sentence=[]
    for w in word_tokens:
        if not w.lower() in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1-t])
            sen=line[1:].strip()
            sen=lemmatize_text(sen)
            sentences.append(sen)
    return labels, sentences

def diagram_norm(diagram):
    rewriter = Rewriter(['prepositional_phrase', 'determiner'])
    rewritten_diagram = rewriter(diagram)
    normalised_diagram = rewritten_diagram.normal_form()
    curry_functor = Rewriter(['curry'])
    curried_diagram = curry_functor(normalised_diagram)
    return curried_diagram.normal_form()


# In[42]:


train_labels, train_data = read_data('train.txt')
dev_labels, dev_data = read_data('dev.txt')
test_labels, test_data = read_data('test.txt')


# In[43]:


parser = BobcatParser(verbose='progress')

raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_dev_diagrams = parser.sentences2diagrams(dev_data)
raw_test_diagrams = parser.sentences2diagrams(test_data)
# raw_test_diagrams = parser.sentences2diagrams(test_data,tokenised=True)


# In[44]:


train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]


# In[45]:




# In[46]:


N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE
NP=AtomicType.NOUN_PHRASE
C=AtomicType.CONJUNCTION
PU=AtomicType.PUNCTUATION

ansatz = IQPAnsatz({N: 1, S: 1, P: 1,NP:1,C:1,PU:1},
                   n_layers=1, n_single_qubit_params=2)
# n_layers : int
#             The number of layers used by the ansatz.
#         n_single_qubit_params : int, default: 3
#             The number of single qubit rotations used by the ansatz.

train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits =  [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]


# In[47]:


# In[48]:


SEED = 2

all_circuits = train_circuits+dev_circuits+test_circuits

backend = AerBackend()
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}
model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
bce = BinaryCrossEntropyLoss()
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2


# In[55]:


EPOCHS = 10000

trainer = QuantumTrainer(
    model,
    loss_function=bce,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    verbose = 'text',
    seed=0
)


# In[56]:


BATCH_SIZE = 20

train_dataset = Dataset(train_circuits,train_labels,batch_size=BATCH_SIZE)

val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)

trainer.fit(train_dataset, val_dataset, logging_step=12)


# In[ ]:




