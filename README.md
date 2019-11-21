# Word-Embedding
Train Word2Vec and FastText word embedding model

How to run the code:
``` 
python3 word_embedding.py -m [model_type] -sg [0 or 1]
```

```
python3 word_embedding_load.py -m [model_type] -sg [0 or 1] -w [words_to_search]
```

Model Type must be either "word2vec" or "fasttext"
SG must be either 0 (for CBOW) or 1 (for Skip-Gram)
The code will load model in .bin format
If the model is in .txt format, type model_type as the file name


Example:
``` 
python3 word_embedding.py -m word2vec -sg 0
```

```
python3 word_embedding_load.py -m word2vec -sg 0 -w decrease success
```



