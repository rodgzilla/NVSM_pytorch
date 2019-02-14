import spacy

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader

spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_docs(filepaths):
    documents = []
    for filepath in filepaths:
        with open(filepath) as file:
            documents.append(file.read().strip().lower())

    return documents

def tokenize_docs(documents):
    tokenized_documents = [tokenize(doc) for doc in documents]

    return tokenized_documents

def create_vocabulary(tokenized_documents):
    vocabulary    = {token for doc in tokenized_documents for token in doc}
    stoi          = {token : i + 2 for i, token in enumerate(vocabulary)}
    stoi['<PAD>'] = 0
    stoi['<UNK>'] = 1
    itos          = {i : token for token, i in stoi.items()}

    return vocabulary, stoi, itos

def create_dataset(tok_docs, stoi, n):
    n_grams      = []
    document_ids = []
    for i, doc in enumerate(tok_docs):
        doc_tok_ids = [stoi[tok] for tok in doc]
        for n_gram in [doc_tok_ids[i : i + n] for i in range(len(doc) - n)]:
            n_grams.append(n_gram)
            document_ids.append(i)

    return n_grams, document_ids

def create_pytorch_datasets(n_grams, doc_ids, val_prop = 0.2):
    n_grams_tensor = torch.tensor(n_grams)
    doc_ids_tensor = torch.tensor(doc_ids)
    full_dataset   = TensorDataset(n_grams_tensor, doc_ids_tensor)
    total_size     = len(full_dataset)
    val_size       = round(total_size * val_prop)
    train, val     = random_split(full_dataset, [total_size - val_size, val_size])

    return train, val

def create_query_dataset(queries, stoi):
    pad_token         = stoi['<PAD>']
    tokenized_queries = [tokenize(query) for query in queries]
    queries_tok_idx   = [[stoi.get(tok, stoi['<UNK>']) for tok in query] for query in tokenized_queries]
    max_len           = max(len(query) for query in queries_tok_idx)
    padded_queries    = [query + [pad_token] * (max_len - len(query)) for query in queries_tok_idx]
    queries_tensor    = torch.tensor(padded_queries)
    dataset           = TensorDataset(queries_tensor)

    return dataset

def evaluate_queries(nvsm, queries_text, doc_names, stoi, batch_size, device):
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
