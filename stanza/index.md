# Stanza

## Table of Contents

1. [Introduction](#Introduction)
2. [Why do we need Stanza?](#WhyStanza) 
3. [Installation](#Installation)
4. [Stanza Basic Components](#Stanza_Basic_Components)
5. [Interesting Stanza Things](#interesting_stanza_things)
6. [Sentiment Analysis using Stanza](#SentimentAnalysis)
7. [Learn more about Stanza here](#LearnMore)
8. [Local Development Guide](#LocalDevelopmentGuide)

<a name="Introduction"></a>
## 1. Introduction

Please check the introduction [here](introduction.md)

<a name="WhyStanza"></a>
## 2. Why do we need Stanza?

Choosing the best NLP library depends on your specific needs and requirements. If you need to work with languages other than English, NLTK may not be the best choice due to its limited language support. In contrast, both Stanza and Spacy offer pre-trained models for a variety of languages. If you're looking for faster processing, both Spacy and Stanza support GPU, which can speed up processing times. When it comes to community support, NLTK and Spacy have larger and more established communities, which can be beneficial when seeking help or resources.

Stanza stands out for its [biomedical models](https://stanfordnlp.github.io/stanza/biomed_model_performance.html), which offer high accuracy for NLP tasks in the biomedical field. Sci-Spacy also provides biomedical models, but their performance is slightly lower compared to Stanza. Also, Spacy tokenizer does not handle multi-word token expansion for other languages. Although Stanza is a newer library, it utilizes state-of-the-art models trained on large datasets, which can lead to greater accuracy in many NLP tasks. In contrast, both Spacy and NLTK are more established and mature libraries. Finally, Spacy and Stanza provide several additional features out-of-the-box, such as named entity recognition, while NLTK may require custom models to implement these features. Also, Stanza includes a Python interface to the Stanford CoreNLP Java package that is a preferred choice for many NLP applications because of comprehensive features, accuracy, performance, language support, and integration options.

<a name="Installation"></a>
## 3. Installation

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

## 4. Stanza Basic Components

Please check the components [here](BasicComponents.md)


<a name="interesting_stanza_things"></a>
## 5. Interesting Stanza Things 

Please check the interesting Stanza things [here](interesting_stanza.md)

<a name="SentimentAnalysis"></a>
## 6. Sentiment Analysis using Stanza

We will use Stanza to do the Sentiment analysis. 

Please check the analysis [here](Stanza.md)

Here's the colab [link](https://colab.research.google.com/drive/1SVBPyfczj_KtpsbAJmHcJX4ulcMrk4O3?usp=sharing) as well

<a name="LearnMore"></a>
### 7. Learn more about Stanza here from other existing tutorials:
- https://stanfordnlp.github.io/stanza/tutorials.html
- https://colab.research.google.com/github/stanfordnlp/stanza/blob/master/demo/Stanza_Beginners_Guide.ipynb
- https://analyticsindiamag.com/how-to-use-stanza-by-stanford-nlp-group-with-python-code/
- https://pemagrg.medium.com/nlp-using-stanza-3775c7e00f2a

To understand the process of modifying this tutorial using Docker, refer to the below local development guide:

<a name="LocalDevelopmentGuide"></a>
## 8. Local Development Guide
Please check the guide [here](../local_development.md)