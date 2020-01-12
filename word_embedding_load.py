from gensim.models import Word2Vec
from gensim.models.fasttext import FastText

from gensim.models import KeyedVectors

import os
import argparse
import sys


def load_model(model_dir, model_name,model_type='fasttext'):
    '''
    Load word embedding model
    Args:
        model_dir: model directory
        model_name: model name
        model_type: word embedding algorithms. Either 'fasttext' or 'word2vec'
    Returns:
        model: word embedding model
    '''
    if '.txt' in model_name:
        model = KeyedVectors.load_word2vec_format(os.path.join(model_dir,model_name))
        return model
    else:
        assert model_type == 'fasttext' or model_type == 'word2vec', "model must be fasttext or word2vec"
        if model_type=='fasttext':
            model = FastText.load(os.path.join(model_dir,model_name)+'.bin')
            return model
        elif model_type=='word2vec':
            model = Word2Vec.load(os.path.join(model_dir,model_name)+'.bin')
            return model


def most_similar_words(word,model,topn=10):
    '''
    Find the associated filtered_words
    Args:
        word: list of words to search
        model: word embedding model
        topn: the top n associated words to search
    '''
    x = model.most_similar(word,topn=10)
    return x


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-m", type=str, required=True, help="model type")
    ap.add_argument("-sg", type=str, required=False, help="choose 0 for CBOW or 1 for Skip-gram. Not required if model type is txt file")
    ap.add_argument("-w", nargs='+', required=True, help="words to search")
    ap.add_argument("-p", type=str, required=True, help="model path")
    ap.add_argument("-topn", type=int, required=False, help="Top-N most similar words. Default: 20")


    args = ap.parse_args()

    model_type = args.m
    sg = args.sg
    words = args.w
    model_dir = args.p
    topn = args.topn


    if '.txt' in model_type:
        model_name = model_type
    else:
        assert model_type=='word2vec' or model_type=='fasttext', "model type must be word2vec or fasttext."
        assert sg=='1' or sg=='0', "sg must be 1 for skip-gram or 0 for CBOW"
        if sg=='1':
            model_name = 'model_'+model_type+'_sg'
        elif sg=='0':
            model_name = 'model_'+model_type

    print('Model name: ', model_name)
    model=load_model(model_dir, model_name,model_type)

    for word in words:
        print('\nWord to search: ', word, '\n')
        print('Most similar words: ', most_similar_words(word,model,topn), '\n')
