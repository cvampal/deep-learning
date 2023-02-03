#author: Bikram Sahu {bikramsahu@iiserb.ac.in}
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk, re, pprint
import sys,os
import numpy as np
from nltk.metrics import accuracy


# In[2]:


# To convert the data in desired format

def convert_data(dir_sen,dir_lab):
    var = []
    with open(dir_sen,'r') as s:
        with open(dir_lab) as l:
            for line,label in zip(s.readlines(),l.readlines()):
                data = [(a,b) for a,b in zip(line.strip().split(' '),label.strip().split(' '))]
                var.append(data)
    return var


# In[3]:


# Training data
dir_sen = 'data/train/sentences.txt'
dir_lab = 'data/train/labels2.txt'
training_data = convert_data(dir_sen, dir_lab)


# In[4]:


# Test data
dir_sen = 'data/test/sentences.txt'
dir_lab = 'data/test/labels2.txt'
test_data = convert_data(dir_sen, dir_lab)

test_sent = open(dir_sen, 'r')
test_sentences = []
for line in test_sent:
    linearray = line.split()
    for word in linearray:
        test_sentences.append(word)


# In[5]:


#Interpolated Tagger

t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(training_data, backoff=t0)
t2 = nltk.BigramTagger(training_data, backoff=t1)

# Accuracy
t2.evaluate(test_data)


# In[6]:


def sent_to_words(sent_data):
    word_data = []
    for i in sent_data:
        for j in i:
            word_data.append(j)
    return word_data


# In[7]:


# Actual Tags of the test dataset

training_words = sent_to_words(training_data)
tag_fd = nltk.FreqDist(tag for (word, tag) in training_words)
pos_tags = tag_fd.keys()
pos_tag_counts = tag_fd.values()
pos_tags_Actual = []
for i in sorted (tag_fd) : 
    pos_tags_Actual.append(i)
    print ((i, tag_fd[i]), end =" ")


# In[8]:


# Tagged using Interpolated Bigram tagger
test_words = sent_to_words(test_data)
test_tagged = t2.tag(test_sentences)
tag_fd = nltk.FreqDist(tag for (word, tag) in test_tagged)
pos_tags = tag_fd.keys()
pos_tag_counts = tag_fd.values()
for i in sorted (tag_fd) : 
    print ((i, tag_fd[i]), end =" ")


# In[9]:


# Confusion Matrix
reference = []
test = []
for i in range(len(test_words)):
    (word, Actualtag) = test_words[i]
    reference.append(Actualtag)
    (word, postagged) = test_tagged[i]
    test.append(postagged)
    
    
fileout = open('cmout.txt', 'w')
fileout.write(nltk.ConfusionMatrix(reference, test).pretty_format(sort_by_count=True))
fileout.close()

