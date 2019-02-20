from tqdm import tqdm

import torch
from evaluate_model import evaluate, generate_eval

def train(nvsm, device, optimizer, epochs, train_loader, eval_loader,
          k_values, loss_function, lamb, print_every):
    for epoch in tqdm(range(epochs),
                      desc = 'Epochs',
                      ncols = 70):
        tqdm_train_loader = tqdm(
            enumerate(train_loader),
            desc  = 'Batch',
            total = len(train_loader),
            ncols = 70,
            leave = True
        )
        for i, (n_grams, doc_ids) in tqdm_train_loader:
            n_grams    = n_grams.to(device)
            doc_ids    = doc_ids.to(device)
            optimizer.zero_grad()
            pred_proba = nvsm(n_grams, doc_ids)
            loss       = loss_function(nvsm, pred_proba, lamb)
            loss.backward()
            optimizer.step()
            if i % print_every == 0:
                nvsm.eval()
                recall_at_ks = evaluate(
                    nvsm          = nvsm,
                    device        = device,
                    eval_loader   = eval_loader,
                    recalls       = k_values,
                    loss_function = loss_function,
                )
                nvsm.train()
                model_eval = generate_eval(k_values, recall_at_ks)
                print(f'  [{epoch}, {i:5d}]: {loss:5.4f} || {model_eval}')
