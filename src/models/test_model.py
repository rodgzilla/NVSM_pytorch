import pdb

from pathlib import Path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))

from nvsm import NVSM
from utils import load_data, create_query_dataset
from tokenize_documents import tokenize

import torch

def load_model_and_data(model_path,
                        device,
                        model_folder = Path('../../models'),
                        data_folder = Path('../../data/processed')):
    voc, stoi, itos, docs = load_data(
        model_folder,
        data_folder
    )
    docs = docs[:20] # temp
    doc_names             = [doc['name'] for doc in docs]
    nvsm = NVSM(
        n_doc             = len(doc_names),
        n_tok             = len(stoi),
        dim_doc_emb       = 20,
        dim_tok_emb       = 30,
        neg_sampling_rate = 10,
        pad_token_id      = stoi['<PAD>']
    ).to(device)
    nvsm.load_state_dict(torch.load(model_path))
    nvsm.eval()

    return stoi, doc_names, nvsm

def query_loop(nvsm, stoi, device, doc_names):
    # pdb.set_trace()
    document_indices = torch.arange(len(doc_names)).view(1, -1)
    document_indices = document_indices.to(device)
    unk_tok_idx      = stoi['<UNK>']
    while True:
        query         = input('Your query: ')
        query_tok     = tokenize(query)
        query_tok_idx = [stoi.get(tok, unk_tok_idx) for tok in query_tok]
        query_tensor  = torch.tensor(query_tok_idx).view(1,-1).to(device)
        result        = nvsm.representation_similarity(query_tensor, document_indices)
        print(result)

if __name__ == '__main__':
    device                = torch.device('cuda')
    stoi, doc_names, nvsm = load_model_and_data('../../models/nvsm_30_20_10.pt', device)
    query_loop(nvsm, stoi, device, doc_names)
