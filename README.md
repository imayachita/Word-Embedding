# Word-Embedding
Train Word2Vec and FastText word embedding model

How to run the code:


To train word embedding from machine-readable documents in .pdf or .txt format
``` 
python3 word_embedding.py -m [model_type] -sg [0 or 1] -s [stopwords_file] -p [path_to_data_folder] -epoch [number_of_epochs]
```

To load word embedding model and search associated words
```
python3 word_embedding_load.py -m [model_type] -sg [0 or 1] -w [words_to_search]
```


If you only want to convert pdf to csv:
``` 
python3 convert_pdf_text.py -i [directory] -o [output_file_name]

``` 


Model Type must be either "word2vec" or "fasttext"
SG must be either 0 (for CBOW) or 1 (for Skip-Gram)
The code will load model in .bin format
If the model is in .txt format, type model_type as the file name


Example:
``` 
python3 convert_pdf_text.py -i ./data -o text.csv

``` 

``` 
python3 word_embedding.py -m word2vec -sg 0 -s stopwords.txt -p ./data -epoch 1000
```

```
python3 word_embedding_load.py -m word2vec -sg 0 -w decrease success
```



