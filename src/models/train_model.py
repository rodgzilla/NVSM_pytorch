def train(nvsm, device, optimizer, epochs, train_loader, loss_function, lamb, print_every):
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
