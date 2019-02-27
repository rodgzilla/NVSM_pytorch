import pickle
from pathlib import Path
import bcolz
import numpy as np

from utils import load_data

def extract_glove_embeddings(glove_path):
    words        = []
    words_to_idx = {}
    vectors      = bcolz.carray(
        np.zeros(1),
        rootdir = glove_path / '6B.50.dat',
        mode = 'w'
    )

    # Extracting the words and the vectors from the glove file and
    # storing them respectively in a list and a bcolz carray.
    with open(glove_path / 'glove.6B.50d.txt', 'rb') as f:
        for idx, l in enumerate(f):
            line               = l.decode().split()
            word               = line[0]
            words.append(word)
            words_to_idx[word] = idx
            vect               = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    # Reshaping the vectors array to (400001, 50) (400000 vocabulary +
    # the unknown token) and storing it on disk.
    vectors = bcolz.carray(
        vectors[1:].reshape((-1, 50)),
        rootdir = glove_path / '6B.50.dat',
        mode = 'w'
    )
    vectors.flush()

    # Saving the list of words and the correspondance between word
    # and vocabulary index on disk.
    with open(glove_path / '6B.50_words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open(glove_path / '6B.50_idx.pkl', 'wb') as f:
        pickle.dump(words_to_idx, f)

if __name__ == '__main__':
    glove_path = Path('../../glove/')
    extract_glove_embeddings(glove_path)
