from pathlib import Path
import re
import pandas as pd
import sys
import argparse
import numpy as np
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.phrases import Phrases, Phraser

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from os import listdir
import os
from os.path import isfile, join
import time
from datetime import timedelta
from convert_pdf_txt import read_pdf


def import_raw_data(fname):
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
    df = pd.read_csv(fname)
    raw_data = df['Text'].values
    # print('Number of sentences: ', len(raw_data))
    return raw_data


def combine_txt(path,output_file_name):
    '''
    Combine txt files into 1 only file
    Args:
        path: the folder contains all the text files
        output_file_name: file name of the output file
    '''
    file_list = [f for f in os.listdir(path) if isfile(join(path, f))]
    # print('file_list: ', file_list)
    filenames = list(filter(lambda f: f.endswith(('.txt','.TXT')), file_list))
    # print('filenames: ', filenames)

    with open(output_file_name, 'w',encoding='utf-8',errors='ignore') as outfile:
        print('combining text files into 1 file ....')
        for fname in tqdm(filenames):
            # print('File: ', fname)
            with open(os.path.join(path,fname),encoding='utf-8',errors='ignore') as infile:
                for line in infile:
                    outfile.write(line)
                outfile.write('\n\n')

    data = read_txt_file(output_file_name)
    # print(data)
    return data


def clean_specialLetters(text):
    """
    Clean out special characters and non-unicode characters

    Args:
        text: input string
    Returns:
        clean: cleaned string
    """
    removed = re.sub('[^A-Za-z0-9]+', ' ', text)
    clean = removed.encode("ascii", errors="ignore").decode()
    return clean


def remove_numbers(text):
    """
    Clean out numbers

    Args:
        text: input string
    Returns:
        clean: cleaned string
    """
    removed = re.sub('[0-9]+', '', text)
    return removed


def read_txt_file(fname):
    """
    Open the combined corpus and save it to list

    Args:
        data: txt file
    Returns:
        data_list: list where each element is a sentence from the text file

    """

    f = open(fname,'r',encoding='utf-8')
    data_list = f.read()
    f.close()
    data_list = sent_tokenize(data_list)
    return data_list


def remove_stopwords(text,stopwords_file):
    """
    Remove the stopwords

    Args:
        text: text to clean
        stopwords_file: txt file that contains stopwords
    Returns:
        filtered_words: text after stopwords are removed
    """
    f = open(stopwords_file,'r',encoding='utf-8')
    stopwords = f.read().split('\n')
    filtered_words = []
    for sentence in text:
        tokenized = word_tokenize(sentence)
        cleaned = [word for word in tokenized if word not in stopwords]
        cleaned = ' '.join(word for word in cleaned)
        filtered_words.append(cleaned)

    return filtered_words


def clean_data(raw_data_list,stopwords_file=None):
    """
    Clean the raw data by removing special characters and numbers

    Args:
        raw_data_list: data to clean in list format, where each line is 1 sentence
        stopwords_file: txt file that contains stopwords
    Returns:
        clean_data_list: cleaned data in list format
    """
    clean_data_list=[]
    for text in raw_data_list:
        pro_text = text.casefold()
        pro_text = clean_specialLetters(pro_text)
        pro_text = remove_numbers(pro_text)
        clean_data_list.append(pro_text)

    if stopwords_file != None:
        clean_data_list = remove_stopwords(clean_data_list,stopwords_file)

    return clean_data_list


def is_duplicating(phrase):
    """
    Check if a phrase consists of duplicating words
    e.g.
        input: 'in in' will return True
        input: 'in out' will return False

    Args:
        phrase: list of phrases
    Returns:
        boolean

    """
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


def Train_Phraser(text):
    """
    Train Phraser and replace white space in phrases with underscore
    Args:
        text: list where each element is 1 sentence
    Returns:
        training_data: list with phrases separated by underscore
    """
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
    """
    Word tokenize training data
    Args:
        training_data: list where each element is 1 sentence
    Returns:
        tokenized: list after word tokenized
    """
    tokenized = []
    for sentence in training_data:
        sentence = word_tokenize(sentence)
        tokenized.append(sentence)
    return tokenized


def train(train_data, iter=500, sg=0, model='fasttext'):
    """
    Train word embedding model
    Args:
        train_data: list after word tokenized
        iter: number of iterations
        model: word embedding algorithms. Either 'fasttext' or 'word2vec'
    Returns:
        model: word embedding model
    """

    assert model == 'fasttext' or model == 'word2vec', "model must be fasttext or word2vec"
    if model == 'fasttext':
        print('Training model', model,'.....')
        # build vocabulary and train model
        model = FastText(
            train_data,
            size=300,
            window=10,
            min_count=5,
            workers=10,
            iter=iter,
            sg=sg)
        return model

    elif model == 'word2vec':
        print('Training model', model, '.....')
        # build vocabulary and train model
        model = Word2Vec(
            train_data,
            size=300,
            window=10,
            min_count=5,
            workers=10,
            iter=iter,
            sg=sg)
        return model


def save_embedding_model(model,model_dir,model_name):
    """
    Save trained embedding model
    Args:
        model: word embedding model to save
        model_dir: word embedding model directory
        model_name: model name
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir()
    print('Saving model in ', model_dir, ' .....')
    model.save(os.path.join(model_dir, '{}.bin'.format(model_name)))
    model.wv.save_word2vec_format(os.path.join(model_dir, '{}.txt'.format(model_name)))


if __name__ == "__main__":
    start_time = time.time()

    ap = argparse.ArgumentParser()

    ap.add_argument("-m", type=str, required=True, help="model type either word2vec or fasttext")
    ap.add_argument("-sg", type=str, required=True, help="choose 0 for CBOW or 1 for Skip-gram. Not required if model type is txt file")
    ap.add_argument("-s", type=str, required=False, help="stopwords file")
    ap.add_argument("-p", type=str, required=False, help="data path")
    ap.add_argument("-epoch", type=int, required=True, help="number of epochs")


    args = ap.parse_args()

    model_type = args.m
    sg = args.sg
    stopwords_file = args.s
    path = args.p
    n_iter = args.epoch


    # path = './data/'
    out_file_txt = 'combined_text.txt'
    out_file_pdf = 'combined_pdf.txt'
    model_dir = './model/'

    assert model_type=='word2vec' or model_type=='fasttext', "model type must be word2vec or fasttext."
    assert sg=='1' or sg=='0', "sg must be 1 for skip-gram or 0 for CBOW"

    file_list=[]
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root,file))
    # print('File list: ', file_list)
    assert len(file_list)>0, "Data folder is empty."

    txt_flag = 0
    pdf_flag = 0
    for f in file_list:
        if '.txt' in f:
            txt_flag = 1
        elif '.pdf' in f:
            pdf_flag = 1

    if sg=='1':
        model_name = 'model_'+model_type+'_sg'
    elif sg=='0':
        model_name = 'model_'+model_type

    # print(txt_flag,pdf_flag)
    if len(file_list) > 1:
        if txt_flag == 1 and pdf_flag == 0:
            raw_data_list = combine_txt(path,out_file_txt)
        elif pdf_flag == 1 and txt_flag == 0:
            data_pdf = read_pdf(path,out_file_pdf)
            raw_data_list = data_pdf['Text'].values
        else:
            data_txt = combine_txt(path,out_file_txt)
            data_pdf = read_pdf(path,out_file_pdf)
            data_pdf = data_pdf['Text'].values
            raw_data_list = np.concatenate((data_txt,data_pdf),axis=0)
    else:
        assert '.txt' in file_list[0] or '.csv' in file_list[0] or '.pdf' in file_list[0], "File must be .txt or .csv or .pdf"
        print(file_list[0])
        if '.txt' in file_list[0]:
            raw_data_list = read_txt_file(file_list[0])
        elif '.csv' in file_list[0]:
            raw_data_list = import_raw_data(file_list[0])
        else:
            data_pdf = read_pdf(path,out_file_pdf)
            raw_data_list = data_pdf['Text'].values


    print('Number of Sentences: ', len(raw_data_list))

    # fname = 'facilities_text.csv'
    clean_data = clean_data(raw_data_list,stopwords_file)
    training_data = Train_Phraser(clean_data)

    training_data = tokenize(training_data)
    model = train(train_data=training_data,iter=n_iter,sg=sg,model=model_type)
    save_embedding_model(model,model_dir,model_name)

    print ('Elapsed Time:', str(timedelta(seconds=(time.time()-start_time))))
