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


## Sentimental Analysis using Stanza

### Dataset

We will use [Amazon US review dataset](https://huggingface.co/datasets/amazon_us_reviews) of 
hugging face.

As this dataset is huge, we will be using data of a subcategory i.e. Personal_Care_Appliances_v1_00

It has about 85981 reviews. Let's take a look at one sample record 

```json
{
    'marketplace': 'US',
    'customer_id': '32114233',
    'review_id': 'R1QX6706ZWJ1P5',
    'product_id': 'B00OYRW4UE',
    'product_parent': '223980852',
    'product_title': 'Elite Sportz Exercise Sliders are Double Sided and Work Smoothly on Any Surface. Wide Variety of Low Impact Exercise’s You Can Do. Full Body Workout, Compact for Travel or Home Ab Workout',
    'product_category': 'Personal_Care_Appliances',
    'star_rating': 5,
    'helpful_votes': 0,
    'total_votes': 0,
    'vine': 0,
    'verified_purchase': 1,
    'review_headline': 'Good quality. Shipped',
    'review_body': 'Exactly as described. Good quality. Shipped fast',
    'review_date': '2015-08-31'
 }
```

let's get the required data and label. We will use the review_body and star_rating to classify the sentiment.






### Stanza's Sentiment Processor

