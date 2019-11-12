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

import argparse
import sys

def load_model(model_dir, model_name,model_type='FastText'):
    if '.txt' in model_name:
        model = KeyedVectors.load_word2vec_format(os.path.join(model_dir,model_name))
        return model
    elif model_type=='FastText':
        model = FastText.load(os.path.join(model_dir,model_name)+'.bin')
        return model
    elif model_type=='Word2Vec':
        model = Word2Vec.load(os.path.join(model_dir,model_name)+'.bin')
        return model
    else:
        print('Model type must be FastText or Word2Vec')


def most_similar_words(word,model,topn=20):
    return model.most_similar(word,topn=topn)

if __name__ == "__main__":
    words=sys.argv[1:]
    model_type='FastText'
    # model_dir='/home/inneke/Documents/D_drive/Balikpapan_handil/codes/word_embedding/model_mispelling'
    # model_name='model_fasttext_mispelling'
    model_dir='/home/inneke/Documents/D_drive/Balikpapan_handil/codes/Extrofitting-master'
    # model_name='Unsup_extro_domain2_20.txt'
    # model_name='Unsup_extro_glove2_1.txt'
    model_name='Unsup_extro_glove2_5.txt'
    # model_name='SLBGlossary_extro_domain2.txt'
    # model_name = 'Trial2.txt'
    # model_name = 'Unsup_extro_glove2.txt'
    # model_name = 'Framenet_extro_glove2.txt'
    # model_name='Unsup_extro_domain2.txt'
    # model_name = 'Trial_newlabel100.txt'
    # model_name = 'Trial_oldlabel100.txt'
    # model_name='SimpleGloveExtrofit_ppdb1.txt'
    # model_name='ExtrofitafterRetrofit100.txt'
    # model_name='SLBGlossaryPpdbExtrofit1.txt'
    # model_name='PpdbExtrofit1.txt'
    # model_name='SLBGlossaryFramenetExtrofit_Dim100.txt'
    # model_name='GloveSLBGlossaryPpdbExtrofit100.txt'
    # model_dir='/home/inneke/Documents/D_drive/Balikpapan_handil/codes/retrofitting-master'
    # model_name='RetrofitafterExtrofit100.txt'
    # model_name='glove_retrofit.txt'
    # model_name='glove_retrofit_ppdb.txt'
    # model_name='slb_retrofitted.txt'
    # model_name='retrofitted_slb_ppdb.txt'
    # model_name='sample_vec.txt'
    # model_name='model_mispelling_fasttext.txt'
    # model_name='glove.txt'

    print('Model name: ', model_name)
    model=load_model(model_dir, model_name,model_type)
    for word in words:
        print('\nWord to search: ', word, '\n')
        print('Most similar words: ', most_similar_words(word,model), '\n')
