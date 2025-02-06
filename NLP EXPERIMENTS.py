#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk 
from nltk.tokenize import word_tokenize 
from nltk.tag import pos_tag 
from nltk.chunk import RegexpParser 
 
# Input sentence 
sentence = "The quick brown fox jumps over the lazy dog." 
 
# Tokenize the sentence 
tokens = word_tokenize(sentence) 
 
# Perform POS tagging 
pos_tags = pos_tag(tokens) 
 
# Define chunking patterns using regular expressions 
chunking_grammar = r""" 
    NP: {<DT|JJ|NN.*>+} 
    VP: {<VB.*><NP|PP>*} 
""" 
 
# Create a chunk parser 
chunk_parser = RegexpParser(chunking_grammar) 
 
# Parse the POS-tagged text to extract chunks 
chunks = chunk_parser.parse(pos_tags) 
 
# Print the chunks 
chunks.pretty_print()


# In[2]:


import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import brown

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

# Load a corpus (Brown Corpus as an example)
corpus_text = " ".join(brown.words()[:100])  # Taking first 100 words from the Brown corpus

# Tokenize the corpus
tokens = word_tokenize(corpus_text)

# Perform POS tagging
pos_tags = pos_tag(tokens)

# Filter words by specific POS (e.g., Nouns)
nouns = [word for word, tag in pos_tags if tag.startswith('NN')]

# Print results
print("POS Tagged Corpus:", pos_tags)
print("\nExtracted Nouns:", nouns)


# In[ ]:




