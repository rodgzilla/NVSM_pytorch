import pickle
import spacy

from pathlib import Path
from tqdm import tqdm
from glob import glob
from collections import Counter

spacy_en = spacy.load('en')

def tokenize(text):
    '''
    Takes a string as input and returns a list of tokens. This function
    utilizes the tokenizer algorithm from the english spacy package.
    '''
    return [tok.text for tok in spacy_en.tokenizer(text) if not tok.is_stop]

def load_docs(filepaths):
    '''
    Opens and load the content of a list of files.
    '''
    documents = []
    for filepath in tqdm(filepaths, desc = 'Loading doc content'):
        with open(filepath) as file:
            documents.append(file.read().strip().lower())

    return documents

def tokenize_docs(documents):
    '''
    Tokenises a list of documents.
    '''
    tokenized_documents = [tokenize(doc) for doc in tqdm(documents, desc = 'Tokenizing documents')]

    return tokenized_documents

def create_vocabulary(tokenized_documents, max_voc_size = 60000):
    '''
    Creates the set of words presents in the document and the
    two dictionaries stoi (token to voc index) and itos (voc index to
    token).
    '''
    word_counter  = Counter(token for doc in tokenized_documents for token in doc)
    vocabulary    = {token for token, _ in word_counter.most_common()[:max_voc_size]}
    # vocabulary    = {token for doc in tokenized_documents for token in doc}
    stoi          = {token : i + 2 for i, token in enumerate(vocabulary)}
    stoi['<PAD>'] = 0
    stoi['<UNK>'] = 1
    itos          = {i : token for token, i in stoi.items()}

    return vocabulary, stoi, itos

def create_features(source_folder, model_folder, dest_data_folder):
    '''
    Tokenizes the documents, creates the vocabulary and both index
    token dictionaries. This functions serializes everything into
    the folder given as arguments.
    '''
    all_documents = []
    for cat_folder in source_folder.iterdir():
        if not cat_folder.is_dir():
            continue
        category            = cat_folder.name
        filepaths           = list(cat_folder.iterdir())
        documents           = load_docs(filepaths)
        tokenized_documents = tokenize_docs(documents)
        category_documents  = [
            {
                'name'     : name,
                'category' : category,
                'tokens'   : tokens
            }
            for name, tokens in zip(
                [path.name for path in filepaths],
                tokenized_documents
            )
        ]
        all_documents.extend(category_documents)

    voc, stoi, itos = create_vocabulary(doc['tokens'] for doc in all_documents)
    with open(model_folder / 'vocabulary.pkl', 'wb') as voc_file:
        pickle.dump(voc, voc_file)
    with open(model_folder / 'stoi.pkl', 'wb') as stoi_file:
        pickle.dump(stoi, stoi_file)
    with open(model_folder / 'itos.pkl', 'wb') as itos_file:
        pickle.dump(itos, itos_file)
    with open(dest_data_folder / 'tokenized_docs.pkl', 'wb') as tok_docs_file:
        pickle.dump(all_documents, tok_docs_file)

if __name__ == '__main__':
    create_features(
        Path('../../data/interim'),
        Path('../../models'),
        Path('../../data/processed')
    )
