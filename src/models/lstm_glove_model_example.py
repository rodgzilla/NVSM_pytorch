from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils          import create_dataset, create_pytorch_datasets, create_query_dataset, \
                           evaluate_queries, load_data
from train_model    import train
from evaluate_model import evaluate, generate_eval

from nvsm_lstm_glove import NVSMLSTM, loss_function

def main():
    glove_path            = Path('../../glove')
    model_folder          = Path('../../models')
    data_folder           = Path('../../data/processed')
    model_path            = model_folder / 'nvsm_lstm_glove.pt'
    batch_size            = 1000
    voc, stoi, itos, docs = load_data(
        model_folder,
        data_folder
    )
    docs = docs[:20]
    doc_names             = [doc['name'] for doc in docs]
    print('Vocabulary size', len(voc))
    n_grams, document_ids = create_dataset(
        tok_docs = [doc['tokens'] for doc in docs],
        stoi     = stoi,
        n        = 10
    )
    print('N-grams number', len(n_grams))
    k_values              = [1, 3, 5, 10]
    (train_data,
     eval_data,
     eval_train_data)     = create_pytorch_datasets(n_grams, document_ids)
    print('Train dataset size', len(train_data))
    print('Eval dataset size', len(eval_data))
    print('Eval (training) dataset size', len(eval_train_data))
    train_loader          = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    eval_loader           = DataLoader(eval_data, batch_size = batch_size, shuffle = False)
    eval_train_loader     = DataLoader(eval_train_data, batch_size = batch_size, shuffle = False)
    device                = torch.device('cuda')
    lamb                  = 1e-3 # regularization weight in the loss
    nvsm                  = NVSMLSTM(
        n_doc             = len(doc_names),
        dim_doc_emb       = 20,
        neg_sampling_rate = 10,
        n_layer           = 3,
        n_hidden          = 128,
        dropout           = 0.15,
        stoi              = stoi,
        glove_path        = glove_path
    ).to(device)
    optimizer             = optim.Adam(nvsm.parameters(), lr = 1e-3)
    train(
        nvsm          = nvsm,
        device        = device,
        optimizer     = optimizer,
        epochs = 1,
        # epochs        = 120,
        train_loader  = train_loader,
        eval_loader   = eval_train_loader,
        k_values      = k_values,
        loss_function = loss_function,
        lamb          = lamb,
        print_every   = 500
    )
    torch.save(nvsm.state_dict(), model_path)
    nvsm.eval()
    recall_at_ks = evaluate(
        nvsm          = nvsm,
        device        = device,
        eval_loader   = eval_loader,
        recalls       = k_values,
        loss_function = loss_function,
    )
    print(generate_eval(k_values, recall_at_ks))
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
