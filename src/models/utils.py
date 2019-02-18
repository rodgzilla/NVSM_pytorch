import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../features')))

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from tokenize_documents import tokenize

def create_dataset(tok_docs, stoi, n):
    '''
    Creates the dataset by extracting n-grams from the documents using a
    rolling window.
    '''
    n_grams      = []
    document_ids = []
    unk_tok_id   = stoi['<UNK>']
    for i, doc in enumerate(tok_docs):
        doc_tok_ids = [stoi.get(tok, unk_tok_id) for tok in doc]
        for n_gram in [doc_tok_ids[i : i + n] for i in range(len(doc) - n)]:
            n_grams.append(n_gram)
            document_ids.append(i)

    return n_grams, document_ids

def create_pytorch_datasets(n_grams, doc_ids, val_prop = 0.2):
    '''
    Creates and splits the pytorch dataset corresponding to the n_grams and
    doc_ids.
    '''
    n_grams_tensor = torch.tensor(n_grams)
    doc_ids_tensor = torch.tensor(doc_ids)
    full_dataset   = TensorDataset(n_grams_tensor, doc_ids_tensor)
    total_size     = len(full_dataset)
    val_size       = round(total_size * val_prop)
    train, val     = random_split(full_dataset, [total_size - val_size, val_size])

    return train, val

def create_query_dataset(queries, stoi):
    '''
    Creates a TensorDataset of a list of string queries in order to run queries
    on the model.
    '''
    pad_token         = stoi['<PAD>']
    tokenized_queries = [tokenize(query) for query in queries]
    queries_tok_idx   = [[stoi.get(tok, stoi['<UNK>']) for tok in query] for query in tokenized_queries]
    max_len           = max(len(query) for query in queries_tok_idx)
    padded_queries    = [query + [pad_token] * (max_len - len(query)) for query in queries_tok_idx]
    queries_tensor    = torch.tensor(padded_queries)
    dataset           = TensorDataset(queries_tensor)

    return dataset

def evaluate_queries(nvsm, queries_text, doc_names, stoi, batch_size, device):
    '''
    Run a list of queries on the model.
    '''
    query_dataset    = create_query_dataset(queries_text, stoi)
    test_loader      = DataLoader(query_dataset, batch_size = batch_size)
    results          = []
    document_indices = torch.stack([torch.arange(len(doc_names))] * batch_size)
    document_indices = document_indices.to(device)
    for (queries,) in test_loader:
        queries = queries.to(device)
        result  = nvsm.representation_similarity(queries, document_indices[:queries.shape[0]])
        results.extend(list(result.argmax(dim = 1).cpu().numpy()))

    return results
