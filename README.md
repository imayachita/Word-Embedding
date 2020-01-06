# Word-Embedding
Train Word2Vec and FastText word embedding model

# Setup
``` pip install -r requirements.txt```


# Running the code:
To train word embedding from machine-readable documents in .pdf or .txt format
``` 
python3 word_embedding.py -m [model_type] -sg [0 or 1] -s [stopwords_file] -p [path_to_data_folder] -epoch [number_of_epochs]
```

To load word embedding model and search associated words
```
python3 word_embedding_load.py -m [model_type] -sg [0 or 1] -p [model_path] -w [words_to_search]
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
python3 word_embedding_load.py -m word2vec -sg 0 -w stark
```

Results:
```
Model name:  model_word2vec

Word to search:  stark 

Most similar words:  [('winterfell', 0.3728424310684204), ('father', 0.35336238145828247), ('robb', 0.3526594638824463), ('lord', 0.30873656272888184), ('king', 0.303555965423584), ('catelyn', 0.28472280502319336), ('son', 0.26981428265571594), ('realm', 0.269493043422699), ('lady', 0.2655394971370697), ('dead', 0.25631558895111084), ('eddard_stark', 0.25362730026245117), ('brother', 0.24403387308120728), ('ned', 0.24251966178417206), ('war', 0.23529788851737976), ('riverrun', 0.23312583565711975), ('vale', 0.2312782108783722), ('renly', 0.22937656939029694), ('bastard', 0.22729867696762085), ('enemies', 0.22699710726737976), ('lord_eddard', 0.22475899755954742)] 

```



