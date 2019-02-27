import pickle
from pathlib import Path
import bcolz
import numpy as np

from utils import load_data

def load_glove_data(glove_path):
    '''
    Load the word vectors that have been extracted from the glove
    files.
    '''
    vectors = bcolz.open(glove_path / '6B.50.dat')[:]
    with open(glove_path / '6B.50_words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open(glove_path / '6B.50_idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)

    glove = {word : vectors[word2idx[word]] for word in words}

    return glove, vectors.shape[-1]


def create_weight_matrix(glove, stoi, emb_dim, random_scale = 0.6):
    '''
    This function creates an embedding weight matrix by fetching the
    vector from glove if possible and randomly initializing it otherwise
    '''
    target_vocab  = list(stoi.keys())
    vocab_size    = len(target_vocab)
    weight_matrix = np.zeros((vocab_size, emb_dim), dtype = np.float32)
    words_found   = 0

    for word in target_vocab:
        if word == '<UNK>':
            vect = glove['<unk>']
            words_found += 1
        elif word in glove:
            vect = glove[word]
            words_found += 1
        else:
            vect = np.random.normal(scale = random_scale, size = (emb_dim, ))
        weight_matrix[stoi[word]] = vect

    return weight_matrix, words_found, vocab_size

if __name__ == '__main__':
    _, stoi, _, _ = load_data(
        Path('../../models'),
        Path('../../data/processed')
    )
    glove, emb_dim = load_glove_data(Path('../../glove/'))
    weight_matrix, words_found, vocab_size = create_weight_matrix(glove, stoi, emb_dim)
    print(words_found)
    print(vocab_size)
    print(100 * words_found / vocab_size)
