import torch
from evaluate_model import evaluate, print_eval

def train(nvsm, device, optimizer, epochs, train_loader, eval_loader,
          k_values, loss_function, lamb, print_every):
    for epoch in range(epochs):
        for i, (n_grams, doc_ids) in enumerate(train_loader):
            n_grams    = n_grams.to(device)
            doc_ids    = doc_ids.to(device)
            optimizer.zero_grad()
            pred_proba = nvsm(n_grams, doc_ids)
            loss       = loss_function(nvsm, pred_proba, lamb)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(nvsm.parameters(), 0.5)
            # for p in nvsm.parameters():
            #     p.data.add_(-1e-3, p.grad.data)
            optimizer.step()
            if i % print_every == 0:
                print(f'[{epoch}, {i}]: {loss:6.4f}')
                recall_at_ks = evaluate(
                    nvsm          = nvsm,
                    device        = device,
                    eval_loader   = eval_loader,
                    recalls       = k_values,
                    loss_function = loss_function,
                )
                print_eval(k_values, recall_at_ks)
