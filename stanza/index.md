# Stanza

## Table of Contents

1. [Introduction](#Introduction)
2. [Features](#Features)
3. [Why do we need Stanza?](#WhyStanza) 
4. [Installation](#Installation)
5. [Stanza Basic Components](#Stanza_Basic_Components)
6. [Sentiment Analysis using Stanza](#SentimentAnalysis)
7. [Local Development Guide](#LocalDevelopmentGuide)

<a name="Introduction"></a>
## 1. Introduction

Stanza is a package similar to Nltk. It offers a suite of tools that can efficiently and accurately process natural language, from raw text to advanced tasks like syntactic analysis and entity recognition. Stanza also brings the latest NLP models to multiple languages, making it a powerful tool for anyone who needs to work with multilingual text data.

Stanza is a Python natural language analysis package. It contains tools, which can be used in a pipeline, to convert a string containing human language text into lists of sentences and words, to generate base forms of those words, their parts of speech, and morphological features, to give a syntactic structure dependency parse, and to recognize named entities. The toolkit is designed to be parallel among more than 70 languages, using the Universal Dependencies formalism.

Stanza is built with highly accurate neural network components that also enable efficient training and evaluation with your own annotated data. 
In addition, Stanza includes a Python interface to the CoreNLP Java package and inherits additional functionality from there, such as constituency parsing, coreference resolution, and linguistic pattern matching.

<a name="Features"></a>
## 2. Features

- Native Python implementation requiring minimal effort to set up;
- Full neural network pipeline for robust text analytics, including tokenization, multi-word token (MWT) expansion, lemmatization, part-of-speech (POS) and morphological features tagging, dependency parsing, and named entity recognition;
- Pretrained neural models supporting [70 (human) languages](https://stanfordnlp.github.io/stanza/models.html#human-languages-supported-by-stanza);
A stable, officially maintained Python interface to CoreNLP.


<a name="WhyStanza"></a>
## 3. Why do we need Stanza?

Choosing the best NLP library depends on your specific needs and requirements. If you need to work with languages other than English, NLTK may not be the best choice due to its limited language support. In contrast, both Stanza and Spacy offer pre-trained models for a variety of languages. If you're looking for faster processing, both Spacy and Stanza support GPU, which can speed up processing times. When it comes to community support, NLTK and Spacy have larger and more established communities, which can be beneficial when seeking help or resources.

Stanza stands out for its [biomedical models](https://stanfordnlp.github.io/stanza/biomed_model_performance.html), which offer high accuracy for NLP tasks in the biomedical field. Sci-Spacy also provides biomedical models, but their performance is slightly lower compared to Stanza. Although Stanza is a newer library, it utilizes state-of-the-art models trained on large datasets, which can lead to greater accuracy in many NLP tasks. In contrast, both Spacy and NLTK are more established and mature libraries. Finally, Spacy and Stanza provide several additional features out-of-the-box, such as named entity recognition, while NLTK may require custom models to implement these features.

<a name="Installation"></a>
## 4. Installation

``` bash
pip install stanza
```

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


<a name="Stanza_Basic_Components"></a>

## 5. Stanza Basic Components

Please check the components [here](BasicComponents.md)

<a name="SentimentAnalysis"></a>

## 6. Sentiment Analysis using Stanza

We will use Stanza to do the Sentiment analysis. 

Please check the analysis [here](Stanza.md)

Here's the colab [link](https://colab.research.google.com/drive/1SVBPyfczj_KtpsbAJmHcJX4ulcMrk4O3?usp=sharing) as well

<a name="LocalDevelopmentGuide"></a>
## 7. Local Development Guide
[Guide](../local_development.md)