# Word-Embedding
Train Word2Vec and FastText word embedding model <br>
The models are trained on 5 Game of Thrones books (A Song of Ice and Fire). <br>
There are 4 models trained:
1. Word2Vec with CBOW (Continuous Bag of Words)
2. Word2Vec with Skip-Gram
3. FastText with CBOW
4. FastText with Skip-Gram

Because .bin models are very huge in size, only .txt models are uploaded.


# Setup
The code is written in Python 3.6
``` pip install -r requirements.txt```


# Running the code:
To train word embedding from machine-readable documents in .pdf or .txt format
``` 
python word_embedding.py -m [model_type] -sg [0 or 1] -s [stopwords_file] -p [path_to_data_folder] -epoch [number_of_epochs]
```

To load word embedding model and search associated words:

```
python word_embedding_load.py -m [model_type] -sg [0 or 1] -w [words_to_search] -p [model_path] -topn [number_of_top_similar_words_to_show]
```

If you only want to convert pdf to csv:
``` 
python convert_pdf_text.py -i [directory] -o [output_file_name]

``` 


Model Type must be either "word2vec" or "fasttext". <br>
SG equals to 0 refers to for CBOW and 1 refers to Skip-Gram. <br>
The code will load model in .bin format. <br>
If the model is in .txt format, type model_type as the file name. <br>
Default ```topn``` is 10, based on Gensim documentation. <br>
We can search for multiple words and phrases. <br>
- To search for multiple words, separate different words with space <br>
- To search for phrases, separate multiple words within a phrase with underscore


Example:
``` 
python convert_pdf_text.py -i ./data -o text.csv
``` 

``` 
python word_embedding.py -m word2vec -sg 0 -s stopwords.txt -p ./data -epoch 1000
```

```
python word_embedding_load.py -m word2vec -sg 0 -w stark jon_snow -p ./GOT_model
```

Results:
```
Model name:  model_word2vec

Word to search:  stark 

Most similar words:  [('winterfell', 0.3728424310684204), ('father', 0.35336238145828247), ('robb', 0.3526594638824463), ('lord', 0.30873656272888184), ('king', 0.303555965423584), ('catelyn', 0.28472280502319336), ('son', 0.26981428265571594), ('realm', 0.269493043422699), ('lady', 0.2655394971370697), ('dead', 0.25631558895111084)] 


Word to search:  jon_snow 

Most similar words:  [('jon', 0.44456174969673157), ('bran', 0.2948411703109741), ('castle_black', 0.2853333055973053), ('man', 0.2648966908454895), ('wall', 0.25946474075317383), ('mance', 0.25083184242248535), ('winterfell', 0.24593092501163483), ('night_watch', 0.24114950001239777), ('crow', 0.23541709780693054), ('alfyn_crowkiller', 0.2346397042274475)] 
```

Another example with FastText model:
```
python word_embedding_load.py -m fasttext -sg 0 -w stark jon_snow -p ./GOT_model
```

Result:
```
Model name:  model_fasttext

Word to search:  stark 

Most similar words:  [('starks', 0.5628231763839722), ('starks_winterfell', 0.5375465750694275), ('lord_stark', 0.5259763598442078), ('lord_eddard_stark', 0.50931715965271), ('ward_eddard_stark', 0.48959609866142273), ('stark_winterfell', 0.48720109462738037), ('lady_stark', 0.48462986946105957), ('son_eddard_stark', 0.47510993480682373), ('house_stark', 0.4697829484939575), ('karstark', 0.46301573514938354)] 


Word to search:  jon_snow 

Most similar words:  [('jon_snow_ygritte', 0.6690560579299927), ('jon_snow_reflected', 0.561360776424408), ('jon', 0.5090968608856201), ('fallen_snow', 0.42110997438430786), ('night_watch', 0.39625218510627747), ('night_watch_takes', 0.3886195421218872), ('lord_snow', 0.3877685070037842), ('lord_commander_night_watch', 0.37715011835098267), ('benjen_stark', 0.37189415097236633), ('wildlings', 0.35399898886680603)] 
```

Another example with Skip-Gram:
```
python word_embedding_load.py -m word2vec -sg 1 -w stark jon_snow -p ./GOT_model
```

Result:
```
Model name:  model_word2vec_sg

Word to search:  stark 

Most similar words:  [('ward_lady_catelyn', 0.3863148093223572), ('jammos', 0.3757057785987854), ('iord', 0.36879587173461914), ('white_field', 0.3511703610420227), ('father_ward', 0.34293508529663086), ('ser_whalen', 0.3387035131454468), ('ser_forley', 0.33681389689445496), ('pardoned', 0.3281732499599457), ('lady_lysa_arryn', 0.32632267475128174), ('sack_king_landing', 0.32433879375457764)] 


Word to search:  jon_snow 

Most similar words:  [('ned_stark_bastard', 0.32651451230049133), ('father_ward', 0.32257044315338135), ('squire_dalbridge', 0.3176880180835724), ('dark_anger', 0.3141580820083618), ('raider_leader_war_band', 0.31240659952163696), ('raised_hood', 0.30803531408309937), ('mance_rayder', 0.3076817989349365), ('ragwyle', 0.30176836252212524), ('wolves_shadowcat', 0.3009984791278839), ('jon', 0.2983658015727997)]
```


# Conclusion
Compared to Word2Vec, FastText model chunks the words into subwords, therefore the most associated words will tend to have similar subwords. Hence, FastText model will be useful if, for example, we want to find mispelled words in our corpus.

For this corpus, CBOW model seems to perform better than Skip-Gram
