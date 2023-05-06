# Stanza Basic Components

## Document

A Document object holds the annotation of an entire document and is automatically generated when a string is annotated by the Pipeline(Will discuss this shortly).  It contains a collection of Sentences and entities.

Some important properties of a Document:

- text: The raw text of the document.
- sentences: The list of sentences in this document.
- entities: The list of entities in this document

A Document can return all of the words in this Document in order or return all of the tokens in this Document in order.


## Sentence

A Sentence object represents a sentence and contains a list of Tokens in the sentence, a list of all its Words, as well as a list of entities in the sentence.

Some important properties of Sentence:

- text: The raw text of the sentence.
- tokens: The list of tokens in this sentence.
- words: The list of words in this sentence.
- entities: The list of entities in this sentence
- sentiment: The sentiment value for this sentence. We will talk about this shortly.

## Token

A Token object holds a token and a list of its underlying syntactic Words. A token can be a multi-word token as well.
Some important properties of Token:
- text: The text of this token. Example: ‘The’.
- words: The list of syntactic words underlying this token.
- ner: The NER tag of this token, in BIOES format. Example: ‘B-ORG’.

## Word

A Word object holds a syntactic word and all of its word-level annotations. Words are used in all downstream syntactic analyses such as tagging, lemmatization, and parsing. 

Some important properties of Word:
- text: The text of this word. Example: ‘The’.
- lemma: The lemma of this word.
- upos: The universal part-of-speech of this word. Example: ‘NOUN’.

## Pipeline:

At a high level, to start annotating text, you need to first initialize a Pipeline, which pre-loads and chains up a series of Processors. A Pipeline takes in raw text or a Document object that contains partial annotations, runs the specified processors in succession, and returns an annotated Document. 

It takes multiple parameters. Some important ones are:
1. lang
    -   Language code (e.g., "en") or language name (e.g., "English")
2. processors
    - Each processor performs a specific NLP task (e.g., tokenization, dependency parsing, or named entity recognition).
    - This can either be specified as a comma-separated list of processor names to use (e.g., 'tokenize,pos'), or a Python dictionary with Processor names as keys and packages as corresponding values (e.g., {'tokenize': 'ewt', 'pos': 'ewt'}).
    - Some processor options, We can define our custom processors or existing processor like Spacy as well:
        - tokenize
            - Tokenizes the text and performs sentence segmentation. 
            - We will use this in our tutorial
        - mwt
             - Expands multi-word tokens (MWT) predicted by the TokenizeProcessor.
             - For more info check [this](https://stanfordnlp.github.io/stanza/mwt.html)

        - pos
            - Labels tokens with their universal POS (UPOS) tags, treebank-specific POS (XPOS) tags, and universal morphological features (UFeats).
             - For more info check [this](https://universaldependencies.org/u/pos/)
        - lemma
            - Generates the word lemmas for all words in the Document. 
            - We will use this in our tutorial 
        - ner
             - Recognize named entities for all token spans in the corpus.
             - For more info check [this](https://stanfordnlp.github.io/stanza/ner.html)
        - sentiment
            - Assign per-sentence sentiment scores. 
            - We will use this in our tutorial  
3. use_gpu
    - Attempt to use a GPU if available
4. {processor}_model_path
     - Path to load an alternate model.
5. {processor}_pretrain_path
    - For processors which use word vectors, a path to load an alternate set of word vectors.


``` python
import stanza
nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True)
```


Let's check some interesting things about Stanza [here](interesting_stanza.md)

