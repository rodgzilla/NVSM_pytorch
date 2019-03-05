import pickle

from pytorch_pretrained_bert import BertTokenizer

from pathlib import Path
from tqdm import tqdm

from tokenize_documents import load_docs

def tokenize_docs(documents, tokenizer):
    '''
    Tokenises a list of documents.
    '''
    tokenized_documents = [tokenizer.tokenize(doc) for doc in tqdm(documents, desc = 'Tokenizing documents')]

    return tokenized_documents

def create_features(source_folder, dest_data_folder, tokenizer):
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
        tokenized_documents = tokenize_docs(documents, tokenizer)
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

    with open(dest_data_folder / 'tokenized_docs_bert.pkl', 'wb') as tok_docs_file:
        pickle.dump(all_documents, tok_docs_file)

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    create_features(
        Path('../../data/interim'),
        Path('../../data/processed'),
        tokenizer
    )
