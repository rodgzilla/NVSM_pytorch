import pdb

from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import create_dataset, create_pytorch_datasets, create_query_dataset, \
                  evaluate_queries
from train_model import train

from nvsm import NVSM, loss_function

def load_data(model_folder, data_folder):
    with open(model_folder / 'vocabulary.pkl', 'rb') as voc_file:
        voc = pickle.load(voc_file)
    with open(model_folder / 'stoi.pkl', 'rb') as stoi_file:
        stoi = pickle.load(stoi_file)
    with open(model_folder / 'itos.pkl', 'rb') as itos_file:
        itos = pickle.load(itos_file)
    with open(data_folder / 'tokenized_docs.pkl', 'rb') as tok_docs_file:
        docs = pickle.load(tok_docs_file)

    return voc, stoi, itos, docs

def main():
    voc, stoi, itos, docs = load_data(
        Path('../../models'),
        Path('../../data/processed')
    )
    docs                  = docs[:50]
    doc_names             = [doc['name'] for doc in docs]
    print('Vocabulary size', len(voc))
    n_grams, document_ids = create_dataset(
        tok_docs = [doc['tokens'] for doc in docs],
        stoi     = stoi,
        n        = 10
    )
    print('dataset size', len(n_grams))
    # pdb.set_trace()
    train_data, val_data  = create_pytorch_datasets(n_grams, document_ids)
    train_loader          = DataLoader(train_data, batch_size = 51200, shuffle = True)
    device                = torch.device('cuda')
    lamb                  = 1e-3 # regularization weight in the loss
    nvsm                  = NVSM(
        n_doc             = len(doc_names),
        n_tok             = len(stoi),
        dim_doc_emb       = 5,
        dim_tok_emb       = 10,
        neg_sampling_rate = 4,
        pad_token_id      = stoi['<PAD>']
    ).to(device)
    optimizer             = optim.Adam(nvsm.parameters(), lr = 1e-3)
    # train(nvsm, device, optimizer, 50, train_loader, loss_function, lamb, 3)
    train(nvsm, device, optimizer, 10, train_loader, loss_function, lamb, 3)
    queries_text          = [
        'violence king louis decapitated',
        'domain language translate',
        'governement robespierre',
        'perfect imperfect information',
        'ontology translation',
        'high levels of political violence',
        'state education system which promotes civic values',
        'political struggles',
        'Almost all future revolutionary movements looked back to the Revolution as their predecessor',
        'Habermas argued that the dominant cultural model in 17th century France was a "representational" culture',
        'mathematical model winning strategy',
        'solutions for two-person zero-sum games',
        'cooperative coalitions bargaining',
        'eigenvalue',
        'graph, dimension and components',
        'inner product vertex'
    ]
    batch_size            = 32
    evaluation_results    = evaluate_queries(
        nvsm,
        queries_text,
        doc_names,
        stoi,
        batch_size,
        device
    )
    for query, doc_idx in zip(queries_text, evaluation_results):
        print(f'{query:35} -> {doc_names[doc_idx]}')

if __name__ == '__main__':
    main()
