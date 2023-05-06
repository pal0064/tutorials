# Interesting Stanza Things

- We can run Stanza online using the [web ui](http://stanza.run/)
- Stanza supports bulk processing of documents. 
    ``` python
    import stanza
    nlp = stanza.Pipeline(lang="en") 
    documents = ["Stanza looks cool.", "Stanza is powerful."] 
    out_docs = nlp.bulk_process(documents) 
    ```
- Stanaza supports the addition of a new [language](https://stanfordnlp.github.io/stanza/new_language.html)
-  Take a quick look at the performance of Stanza’s pre-trained models on all supported languages. 
- Stanza can use Spacy for fast tokenization and sentence segmentation. You need to install Spacy before executing the below command.
    ``` python
    import stanza
    nlp = stanza.Pipeline(lang="en",processors={'tokenize': 'spacy'}) 
    documents = ["Stanza looks cool.", "Stanza is powerful."] 
    out_docs = nlp.bulk_process(documents) 
    ```

- Stanza provides text cleaning tools as well for the language identification processor. The text cleaning will remove shortened urls, hashtags, user handles, and emojis. 
``` python
    nlp = Pipeline(lang="multilingual", processors="langid", langid_clean_text=True)
```
- Stanza supports multilingual text as well. It can detect the language of the text, and run the appropriate language-specific Stanza pipeline on the text.
``` python
from stanza.pipeline.multilingual import MultilingualPipeline
nlp = MultilingualPipeline()
docs = ["Hello world!", "C'est une phrase française.", "This is an English sentence."]
docs = nlp(docs)
```

Let's move to the next [step](Stanza.md) i.e. Sentiment Analysis using Stanza
