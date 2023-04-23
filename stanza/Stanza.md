```python
%%capture
!pip install stanza
```


```python
use_gpu = False
shrink_dataset = True
```




```python
import stanza
stanza.download('en') # download English model
nlp = stanza.Pipeline('en', use_gpu=use_gpu) # initialize English neural pipeline

```


    Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.5.0.json:   0%|   …


    2023-04-23 12:59:16 INFO: Downloading default packages for language: en (English) ...
    2023-04-23 12:59:18 INFO: File exists: /Users/pal/stanza_resources/en/default.zip
    2023-04-23 12:59:21 INFO: Finished downloading models and saved to /Users/pal/stanza_resources.
    2023-04-23 12:59:21 INFO: Checking for updates to resources.json in case models have been updated.  Note: this behavior can be turned off with download_method=None or download_method=DownloadMethod.REUSE_RESOURCES



    Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.5.0.json:   0%|   …


    2023-04-23 12:59:22 INFO: Loading these models for language: en (English):
    ============================
    | Processor    | Package   |
    ----------------------------
    | tokenize     | combined  |
    | pos          | combined  |
    | lemma        | combined  |
    | constituency | wsj       |
    | depparse     | combined  |
    | sentiment    | sstplus   |
    | ner          | ontonotes |
    ============================
    
    2023-04-23 12:59:22 INFO: Using device: cpu
    2023-04-23 12:59:22 INFO: Loading: tokenize
    2023-04-23 12:59:22 INFO: Loading: pos
    2023-04-23 12:59:22 INFO: Loading: lemma
    2023-04-23 12:59:22 INFO: Loading: constituency
    2023-04-23 12:59:23 INFO: Loading: depparse
    2023-04-23 12:59:23 INFO: Loading: sentiment
    2023-04-23 12:59:23 INFO: Loading: ner
    2023-04-23 12:59:23 INFO: Done loading processors!



```python
doc = nlp("University Of Arizona is a great place to learn natural language processing") # run annotation over a sentence
print(doc.entities)
```

    [{
      "text": "University Of Arizona",
      "type": "ORG",
      "start_char": 0,
      "end_char": 21
    }]



```python
!pip install datasets
```

    Requirement already satisfied: datasets in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (2.11.0)
    Requirement already satisfied: responses<0.19 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (0.18.0)
    Requirement already satisfied: pandas in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (1.3.5)
    Requirement already satisfied: huggingface-hub<1.0.0,>=0.11.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (0.13.4)
    Requirement already satisfied: numpy>=1.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (1.23.4)
    Requirement already satisfied: pyarrow>=8.0.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (10.0.1)
    Requirement already satisfied: aiohttp in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (3.8.1)
    Requirement already satisfied: packaging in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (23.0)
    Requirement already satisfied: requests>=2.19.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (2.28.2)
    Requirement already satisfied: xxhash in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (3.2.0)
    Requirement already satisfied: fsspec[http]>=2021.11.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (2023.3.0)
    Requirement already satisfied: pyyaml>=5.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (5.4.1)
    Requirement already satisfied: tqdm>=4.62.1 in /Users/pal/Library/Python/3.10/lib/python/site-packages (from datasets) (4.64.1)
    Requirement already satisfied: multiprocess in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (0.70.14)
    Requirement already satisfied: dill<0.3.7,>=0.3.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from datasets) (0.3.6)
    Requirement already satisfied: charset-normalizer<3.0,>=2.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (2.1.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (1.8.2)
    Requirement already satisfied: attrs>=17.3.0 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (22.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (1.3.3)
    Requirement already satisfied: async-timeout<5.0,>=4.0.0a3 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (4.0.2)
    Requirement already satisfied: multidict<7.0,>=4.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (6.0.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from aiohttp->datasets) (1.3.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from huggingface-hub<1.0.0,>=0.11.0->datasets) (4.5.0)
    Requirement already satisfied: filelock in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from huggingface-hub<1.0.0,>=0.11.0->datasets) (3.12.0)
    Requirement already satisfied: certifi>=2017.4.17 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (2022.12.7)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (1.26.15)
    Requirement already satisfied: idna<4,>=2.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from requests>=2.19.0->datasets) (3.4)
    Requirement already satisfied: python-dateutil>=2.7.3 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from pandas->datasets) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from pandas->datasets) (2022.7.1)
    Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages (from python-dateutil>=2.7.3->pandas->datasets) (1.16.0)
    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.0.1[0m[39;49m -> [0m[32;49m23.1[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpip install --upgrade pip[0m



```python
from datasets import load_dataset
```


```python

```


```python
dataset = load_dataset('amazon_us_reviews','Personal_Care_Appliances_v1_00') 
```

    Found cached dataset amazon_us_reviews (/Users/pal/.cache/huggingface/datasets/amazon_us_reviews/Personal_Care_Appliances_v1_00/0.1.0/17b2481be59723469538adeb8fd0a68b0ba363bbbdd71090e72c325ee6c7e563)



      0%|          | 0/1 [00:00<?, ?it/s]



```python
dataset['train'][0]
```




    {'marketplace': 'US',
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
     'review_date': '2015-08-31'}




```python
dataset['train'].shape
```




    (85981, 15)




```python
dataset['train'][0]
```




    {'marketplace': 'US',
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
     'review_date': '2015-08-31'}




```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
```

    [nltk_data] Error loading stopwords: <urlopen error [SSL:
    [nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    [nltk_data]     unable to get local issuer certificate (_ssl.c:997)>





    False




```python
nlp = stanza.Pipeline(lang='en', processors='tokenize,lemma',use_gpu=use_gpu)
```

    2023-04-23 12:59:29 INFO: Checking for updates to resources.json in case models have been updated.  Note: this behavior can be turned off with download_method=None or download_method=DownloadMethod.REUSE_RESOURCES



    Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.5.0.json:   0%|   …


    2023-04-23 12:59:29 INFO: Loading these models for language: en (English):
    ========================
    | Processor | Package  |
    ------------------------
    | tokenize  | combined |
    | lemma     | combined |
    ========================
    
    2023-04-23 12:59:29 INFO: Using device: cpu
    2023-04-23 12:59:29 INFO: Loading: tokenize
    2023-04-23 12:59:29 INFO: Loading: lemma
    2023-04-23 12:59:29 INFO: Done loading processors!



```python
size = dataset['train'].shape[0]
if shrink_dataset:
  size  = 100
```


```python
from tqdm.auto import tqdm
```


```python
def pre_process_review_texts(texts):
  processed_texts = []
  for text in tqdm(texts):
    doc = nlp(text.lower())
    lemmatized_tokens= []
    for sentence in doc.sentences:
        for word in sentence.words:
          if word.lemma and word.lemma not in stopwords.words('english'):
            lemmatized_tokens.append(word.lemma)
    processed_text = ' '.join(lemmatized_tokens)
    processed_texts.append(processed_text)
  return processed_texts

processed_texts = pre_process_review_texts(dataset['train']['review_body'][0:size])
```


      0%|          | 0/100 [00:00<?, ?it/s]



```python
def pre_process_ratings(ratings):
  sentiments  = []
  for rating in ratings:
    if rating <=2:
      sentiments.append(0)
    elif rating >=4:
      sentiments.append(2)
    else:
      sentiments.append(1)
  return sentiments
true_labels = pre_process_ratings(dataset['train']['star_rating'][0:size])
```


```python
from collections import defaultdict
nlp_sentiment = stanza.Pipeline(lang='en', processors='tokenize,sentiment',use_gpu=use_gpu)

```

    2023-04-23 12:59:31 INFO: Checking for updates to resources.json in case models have been updated.  Note: this behavior can be turned off with download_method=None or download_method=DownloadMethod.REUSE_RESOURCES



    Downloading https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.5.0.json:   0%|   …


    2023-04-23 12:59:31 INFO: Loading these models for language: en (English):
    ========================
    | Processor | Package  |
    ------------------------
    | tokenize  | combined |
    | sentiment | sstplus  |
    ========================
    
    2023-04-23 12:59:31 INFO: Using device: cpu
    2023-04-23 12:59:31 INFO: Loading: tokenize
    2023-04-23 12:59:31 INFO: Loading: sentiment
    2023-04-23 12:59:32 INFO: Done loading processors!



```python
def get_sentiment(text):
  sentiments = defaultdict(int)
  doc = nlp_sentiment(text)
  total =0
  for sentence in doc.sentences:
    sentiments[sentence.sentiment]+=1
    total+=1
  all_values = dict([ (sentiment,sentiments[sentiment]/total) for sentiment in sentiments])
  sorted_values = sorted(all_values.items(), key=lambda x:x[1],reverse=True)
  return sorted_values[0][0]
```


```python
def get_sentiment_using_stanza(texts,labels):
  predicted_sentiments =[]
  true_sentiments = []
  for idx,text in tqdm(enumerate(texts),total=len(texts)):
    try:
      sentiment = get_sentiment(text)
      predicted_sentiments.append(sentiment)
      true_sentiments.append(labels[idx])
    except IndexError:
       pass
  return predicted_sentiments,true_sentiments
predicted_sentiments_stanza,true_sentiments = get_sentiment_using_stanza(processed_texts,true_labels)
```


      0%|          | 0/100 [00:00<?, ?it/s]



```python
from sklearn.metrics import classification_report
print(classification_report(true_sentiments, predicted_sentiments_stanza))
```

                  precision    recall  f1-score   support
    
               0       0.60      0.55      0.57        11
               1       0.08      0.75      0.15         4
               2       0.96      0.60      0.74        85
    
        accuracy                           0.60       100
       macro avg       0.55      0.63      0.49       100
    weighted avg       0.89      0.60      0.70       100
    



```python
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
```


```python
nltk.download('punkt')
nltk.download('wordnet')
```

    [nltk_data] Error loading punkt: <urlopen error [SSL:
    [nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    [nltk_data]     unable to get local issuer certificate (_ssl.c:997)>
    [nltk_data] Error loading wordnet: <urlopen error [SSL:
    [nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    [nltk_data]     unable to get local issuer certificate (_ssl.c:997)>





    False




```python
def preprocess_review_texts_using_nltk(texts):
  processed_texts = []
  for text in tqdm(texts):
      tokens = word_tokenize(text.lower())
      filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
      lemmatizer = WordNetLemmatizer()
      lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
      processed_text = ' '.join(lemmatized_tokens)
      processed_texts.append(processed_text)
  return processed_texts
processed_texts = preprocess_review_texts_using_nltk(dataset['train']['review_body'][0:size])
```


      0%|          | 0/100 [00:00<?, ?it/s]



```python
nltk.download('vader_lexicon')
```

    [nltk_data] Error loading vader_lexicon: <urlopen error [SSL:
    [nltk_data]     CERTIFICATE_VERIFY_FAILED] certificate verify failed:
    [nltk_data]     unable to get local issuer certificate (_ssl.c:997)>





    False




```python
analyzer = SentimentIntensityAnalyzer()

def get_sentiment_using_nltk(texts,labels):
  predicted_sentiments =[]
  true_sentiments = []
  for idx,text in tqdm(enumerate(texts),total=len(texts)):
    try:
      scores = analyzer.polarity_scores(text)
      sentiment_score = scores['compound']
      predicted_sentiments.append(round(sentiment_score+1))
      true_sentiments.append(labels[idx])
    except IndexError:
       pass
  return predicted_sentiments,true_sentiments
predicted_sentiments_nltk,true_sentiments = get_sentiment_using_nltk(processed_texts,true_labels)
```


      0%|          | 0/100 [00:00<?, ?it/s]



```python
print(classification_report(true_sentiments, predicted_sentiments_nltk))
```

                  precision    recall  f1-score   support
    
               0       0.50      0.18      0.27        11
               1       0.05      0.50      0.09         4
               2       0.96      0.65      0.77        85
    
        accuracy                           0.59       100
       macro avg       0.51      0.44      0.38       100
    weighted avg       0.88      0.59      0.69       100
    

