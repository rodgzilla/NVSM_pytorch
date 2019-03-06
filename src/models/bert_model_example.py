import pickle
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils          import create_dataset, create_pytorch_datasets, create_query_dataset, \
                           evaluate_queries
from train_model    import train
from evaluate_model import evaluate, generate_eval

from nvsm_bert import NVSMBERT, loss_function

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

def load_data(data_folder, pretrained_model):
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    with open(data_folder / 'tokenized_docs_bert.pkl', 'rb') as tok_docs_file:
        docs = pickle.load(tok_docs_file)

    return docs, tokenizer

def create_dataset(tok_docs, tokenizer, n):
    '''
    Creates the dataset by extracting n-grams from the documents using a
    rolling window.
    '''
    n_grams      = []
    document_ids = []
    unk_tok_id   = tokenizer.vocab['[UNK]']
    cls_tok_id   = tokenizer.vocab['[CLS]']
    sep_tok_id   = tokenizer.vocab['[SEP]']
    for i, doc in enumerate(tok_docs):
        doc_tok_ids = [tokenizer.vocab[tok] for tok in doc]
        for n_gram in [doc_tok_ids[i : i + n] for i in range(len(doc) - n)]:
            if all(tok == unk_tok_id for tok in n_gram):
                continue
            n_grams.append([cls_tok_id] + n_gram + [sep_tok_id])
            document_ids.append(i)

    return n_grams, document_ids

def main():
    pretrained_model      = 'bert-base-uncased'
    glove_path            = Path('../../glove')
    model_folder          = Path('../../models')
    data_folder           = Path('../../data/processed')
    model_path            = model_folder / 'nvsm_bert.pt'
    batch_size            = 120 # for 150, 8053 / 8113MB GPU memory, to tweak
    epochs                = 3
    docs, tokenizer       = load_data(
        data_folder,
        pretrained_model
    )
    docs = docs[:20]
    doc_names             = [doc['name'] for doc in docs]
    n_grams, document_ids = create_dataset(
        tok_docs  = [doc['tokens'] for doc in docs],
        tokenizer = tokenizer,
        n         = 10
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
    lamb                  = 1e-3
    nvsm                  = NVSMBERT(
        pretrained_model  = pretrained_model,
        n_doc             = len(doc_names),
        dim_doc_emb       = 20,
        neg_sampling_rate = 10,
    ).to(device)
    # BERT custom optimizer
    param_optimizer = list(nvsm.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        params = optimizer_grouped_parameters,
        lr = 5e-5,
        warmup = 0.1,
        t_total = len(train_loader) * epochs
    )
    # train(
    #     nvsm          = nvsm,
    #     device        = device,
    #     optimizer     = optimizer,
    #     epochs        = epochs,
    #     train_loader  = train_loader,
    #     eval_loader   = eval_train_loader,
    #     k_values      = k_values,
    #     loss_function = loss_function,
    #     lamb          = lamb,
    #     print_every   = 500
    # )
    torch.save(nvsm.state_dict(), model_path)
    nvsm.eval()
    # recall_at_ks = evaluate(
    #     nvsm          = nvsm,
    #     device        = device,
    #     eval_loader   = eval_loader,
    #     recalls       = k_values,
    #     loss_function = loss_function,
    # )
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
