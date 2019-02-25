import pdb

from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))

from nvsm_lstm import LstmNvsm
from utils import load_data, create_query_dataset
from tokenize_documents import tokenize

import torch

def load_model_and_data(model_path,
                        device,
                        model_folder = Path('../../models'),
                        data_folder = Path('../../data/processed')):
    '''
    Loads model, token to vocabulary index dictionary and list of
    document names.
    '''
    voc, stoi, itos, docs = load_data(
        model_folder,
        data_folder
    )
    doc_names             = [doc['name'] for doc in docs]
    nvsm = LstmNvsm(
        n_doc             = len(doc_names),
        n_tok             = len(stoi),
        dim_doc_emb       = 20,
        dim_tok_emb       = 30,
        neg_sampling_rate = 10,
        pad_token_id      = stoi['<PAD>'],
        n_layer           = 3,
        n_hidden          = 128,
        dropout           = 0.15
    ).to(device)
    nvsm.load_state_dict(torch.load(model_path))
    nvsm.eval()

    return stoi, doc_names, nvsm

def query_loop(nvsm, stoi, device, doc_names, n):
    '''
    Infinitely prompts the user for a query, finds the n documents of
    the dataset whose embeddings are the closest and prints their
    names.
    '''
    document_indices = torch.arange(len(doc_names)).view(1, -1)
    document_indices = document_indices.to(device)
    unk_tok_idx      = stoi['<UNK>']
    while True:
        query         = input('Your query: ')
        query_tok     = tokenize(query)
        query_tok_idx = [stoi.get(tok, unk_tok_idx) for tok in query_tok]
        query_tensor  = torch.tensor(query_tok_idx).view(1,-1).to(device)
        result        = nvsm.representation_similarity(query_tensor, document_indices)
        closest_doc_ids = result.detach().argsort(descending = True)[:n]
        print(query)
        print('\n'.join(f'\t{doc_names[idx]}' for idx in closest_doc_ids))
        print()

if __name__ == '__main__':
    device                = torch.device('cuda')
    stoi, doc_names, nvsm = load_model_and_data('../../models/nvsm_lstm.pt', device)
    query_loop(nvsm, stoi, device, doc_names, 5)
