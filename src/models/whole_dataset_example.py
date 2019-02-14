import pdb

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import tokenize, load_docs, tokenize_docs, create_vocabulary, \
                  create_dataset, create_pytorch_datasets, create_query_dataset, \
                  evaluate_queries
from train_model import train

from nvsm import NVSM, loss_function

from glob import glob
from pathlib import Path

def main():
    # filepaths             = map(Path, glob('../../data/raw/**', recursive = True)[:20])
    filepaths             = map(Path, glob('../../data/raw/**', recursive = True))
    filepaths             = list(filter(lambda fn: fn.is_file(), filepaths))
    documents             = load_docs(filepaths)
    tokenized_documents   = tokenize_docs(documents)
    voc, stoi, itos       = create_vocabulary(tokenized_documents)
    n_grams, document_ids = create_dataset(tokenized_documents, stoi, 10)
    train_data, val_data  = create_pytorch_datasets(n_grams, document_ids)
    train_loader          = DataLoader(train_data, batch_size = 10000, shuffle = True)
    device                = torch.device('cuda')
    lamb                  = 1e-3 # regularization weight in the loss
    nvsm                  = NVSM(
        n_doc             = len(tokenized_documents),
        n_tok             = len(stoi),
        dim_doc_emb       = 20,
        dim_tok_emb       = 30,
        neg_sampling_rate = 4,
        pad_token_id      = stoi['<PAD>']
    ).to(device)
    optimizer             = optim.Adam(nvsm.parameters(), lr = 1e-3)
    # train(nvsm, device, optimizer, 50, train_loader, loss_function, lamb, 3)
    train(nvsm, device, optimizer, 1, train_loader, loss_function, lamb, 3)
    doc_names             = [path.name for path in filepaths]
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
