# Stanza

## Table of Contents

1. [Introduction](#Introduction)
2. [Installation](#Installation)
2. [Sentimental Analysis using Stanza](Stanza.md)

## Introduction

Stanza is a package similar to Nltk. It offers a suite of tools that can efficiently and accurately process natural language, from the raw text to advanced tasks like syntactic analysis and entity recognition. Stanza also brings the latest NLP models to multiple languages, making it a powerful tool for anyone who needs to work with multilingual text data.

Stanza is a Python natural language analysis package. It contains tools, which can be used in a pipeline, to convert a string containing human language text into lists of sentences and words, to generate base forms of those words, their parts of speech and morphological features, to give a syntactic structure dependency parse, and to recognize named entities. The toolkit is designed to be parallel among more than 70 languages, using the Universal Dependencies formalism.

Stanza is built with highly accurate neural network components that also enable efficient training and evaluation with your own annotated data. 
In addition, Stanza includes a Python interface to the CoreNLP Java package and inherits additional functionality from there, such as constituency parsing, coreference resolution, and linguistic pattern matching.


## Features

- Native Python implementation requiring minimal efforts to set up;
- Full neural network pipeline for robust text analytics, including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition;
- Pretrained neural models supporting [70 (human) languages](https://stanfordnlp.github.io/stanza/models.html#human-languages-supported-by-stanza);
A stable, officially maintained Python interface to CoreNLP.


## Installation

`
pip install stanza
`

Let's check if everything is working fine

```python
import stanza
stanza.download('en') # download English model
nlp = stanza.Pipeline('en') # initialize English neural pipeline
doc = nlp("University Of Arizona is a great place to learn natural language processing") # run annotation over a sentence
print(doc.entities)
```
```json
[{
  "text": "University Of Arizona",
  "type": "ORG",
  "start_char": 0,
  "end_char": 21
}]
```
for more installation instructions follow the [official link](https://stanfordnlp.github.io/stanza/installation_usage.html)
