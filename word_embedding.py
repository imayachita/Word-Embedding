import csv
import collections
from pathlib import Path
import re
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from matplotlib import pyplot

from gensim.test.utils import datapath
from gensim.models.word2vec import Text8Corpus,BrownCorpus
from gensim.models.phrases import Phrases, Phraser
from gensim.models.phrases import original_scorer
from gensim.utils import simple_preprocess
from gensim.models import KeyedVectors
import gensim.models.wrappers
from gensim.models.keyedvectors import FastTextKeyedVectors

from os import listdir
import os
from os.path import isfile, join
import time
from datetime import timedelta

def combine_txt(path,output_file_name):
    '''
    Combine txt files into 1 only file
    Path: the folder contains all the text files
    '''
    filenames = [f for f in os.listdir(path) if isfile(join(path, f))]

    with open(output_file_name, 'w',encoding='utf-8') as outfile:
        for fname in filenames:
            print(fname)
            with open(path+'/'+fname,encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write('\n\n')

def import_raw_data(path, fname):
    '''
    Assumes the first column in the CSV is the text data and reads into
    this script.

    Args:
        path (str): path to the data
        fname (str): filename of the data
    Returns:
        raw_data (list): list of loaded data
    '''
    raw_data = []
    with open(path + fname, 'r', encoding="utf8") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            raw_data.append(row[0])

    return (raw_data)  # to be changed

def clean_specialLetters_2(cell):
    """
    Cleaning out special characters and non-unicode characters.

    Args:
        cell (str): input string
    Returns:
        clean (str): cleaned string
    """
    removed = re.sub('[^A-Za-z0-9]+', ' ', cell)
    clean = removed.encode("ascii", errors="ignore").decode()
    return clean

def remove_numbers(cell):
    """
    Cleaning out numbers.

    Args:
        cell (str): input string
    Returns:
        clean (str): cleaned string
    """
    removed = re.sub('[0-9]+', '', cell)
    return removed

def combine_2_txt(file1,file2,out_file):
    #combine raw_data_list from deepsearch output to all text files
    #combine all text files into 1 file
    filenames = [file1,file2]

    with open(out_file, 'w',encoding='utf-8') as outfile:
        for fname in filenames:
            with open(fname,'r',encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)
                    outfile.write('\n')

def read_txt_file(raw_data_fname):
    #open the combined corpus and save it to list

    f = open(raw_data_fname,'r',encoding='utf-8')
    raw_data_list = f.read()
    f.close()
    raw_data_list = sent_tokenize(raw_data_list)
    return raw_data_list

def load_stopwords(stopwords_file):
    #load stopwords

    f = open(stopwords_file,'r',encoding='utf-8')
    stopwords = f.read().split('\n')
    f.close()
    return stopwords


def remove_stopwords(text,stopwords_file):
    #remove the stopwords
    stopwords = load_stopwords(stopwords_file)
    filtered_words = []
    for sentence in text:
        tokenized = word_tokenize(sentence)
        cleaned = [word for word in tokenized if word not in stopwords]
        cleaned = ' '.join(word for word in cleaned)
        filtered_words.append(cleaned)

    return filtered_words


def clean_data(raw_data_list,stopwords_file):
    #clean the raw data by removing special characters and numbers
    clean_data_list_to_txt=[]
    for text in raw_data_list:
        pro_text = text.casefold()
        pro_text = clean_specialLetters_2(pro_text)
        pro_text = remove_numbers(pro_text)
        clean_data_list_to_txt.append(pro_text)

    clean_data_list_to_txt = remove_stopwords(clean_data_list_to_txt,stopwords_file)
    return clean_data_list_to_txt


def is_duplicating(phrase):
    '''
    e.g. input: 'ft ft', return True. input: 'oil gas', return False.
    '''
    words = phrase.split()
    counts = {}
    for word in words:
        if word not in counts:
            counts[word] = 0
        counts[word] += 1

    if counts[word]<=1:
        return False
    else:
        return True


#replace space in phrases with underscore
def replace_space_with_underscore_in_phrase(text):
    training_data = []
    sentence_stream = [doc.split(" ") for doc in text]
    bigram = Phrases(sentence_stream, min_count=3, delimiter=b' ')
    trigram  = Phrases(bigram[sentence_stream], min_count=3, delimiter=b' ')
    tri = Phraser(trigram)
    bi = Phraser(bigram)

    for i,sentence in enumerate(text):
        words = tri[bi[sentence.split()]]
        for i,word in enumerate(words):
            if ' ' in word and not is_duplicating(word):
                phrase = word.replace(' ','_')
                words[i] = phrase
        cleaned = ' '.join(word for word in words)
        training_data.append(cleaned)
    return training_data


def tokenize(training_data):
    #word tokenize training data
    tokenized = []
    for sentence in training_data:
        sentence = word_tokenize(sentence)
        tokenized.append(sentence)
    return tokenized


def train(tokenized, model='fasttext'):
    if model == 'fasttext':
        print('Training model', model,'.....')
        # build vocabulary and train model
        model = FastText(
            tokenized,
            size=300,
            window=10,
            min_count=5,
            workers=10,
            iter=500,
            sg=0)
        return model

    elif model == 'word2vec':
        print('Training model', model, '.....')
        # build vocabulary and train model
        model = Word2Vec(
            tokenized,
            size=300,
            window=10,
            min_count=5,
            workers=10,
            iter=500,
            sg=0)
        return model

    else:
        print('Model name should be fasttext or word2vec')

def save_embedding_model(model,model_dir,model_name):
    print('Saving model', model_name, ' .....')
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    model.save(os.path.join(model_dir, '{}.bin'.format(model_name)))
    model.wv.save_word2vec_format(os.path.join(model_dir, '{}.txt'.format(model_name)))


if __name__ == "__main__":
    start_time = time.time()
    file1 = './data/Final_Data/text_df_to_NER.txt'
    file2 = './data/Final_Data/clean_syn.txt'
    out_file = 'combined.txt'
    model_dir = './model_mispelling_word2vec/'
    # model = 'fasttext'
    model = 'word2vec'
    model_name = 'model_'+model
    stopwords_file = 'stopwords.txt'
    # combine_2_txt(file1,file2,out_file)
    raw_data_list = read_txt_file(out_file)
    raw_data_list = clean_data(raw_data_list,stopwords_file)
    training_data = replace_space_with_underscore_in_phrase(raw_data_list)

    training_data = tokenize(training_data)
    print(training_data)
    # model = train(training_data,model)
    # save_embedding_model(model,model_dir,model_name)
    # print ('Elapsed Time:', str(timedelta(seconds=(time.time()-start_time))))
