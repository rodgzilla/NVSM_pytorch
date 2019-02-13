import torch
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim

import spacy

from nvsm import NVSM
from nvsm import loss_function

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

def train(nvsm, device, optimizer, epochs, train_loader, lamb, print_every):
    for epoch in range(epochs):
        for i, (n_grams, doc_ids) in enumerate(train_loader):
            n_grams    = n_grams.to(device)
            doc_ids    = doc_ids.to(device)
            optimizer.zero_grad()
            pred_proba = nvsm(n_grams, doc_ids)
            loss       = loss_function(nvsm, pred_proba, lamb)
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                print(f'[{epoch},{i}]: {loss}')

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

def main():
    filepaths = [
        '../../data/raw/language/Word_formation',
        '../../data/raw/language/Terminology',
        '../../data/raw/history/Jacobin',
        '../../data/raw/history/French_Revolution',
        '../../data/raw/math/Game_theory',
        '../../data/raw/math/Laplacian_matrix'
    ]
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
    train(nvsm, device, optimizer, 50, train_loader, lamb, 3)
    # train(nvsm, device, optimizer, 3, train_loader, lamb, 3)
    doc_names             = [path.split('/')[-1] for path in filepaths]
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
    # max_size = max(len(query) for query in queries_text)
    for query, doc_idx in zip(queries_text, evaluation_results):
        print(f'{query:35} -> {doc_names[doc_idx]}')

main()
