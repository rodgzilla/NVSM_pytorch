import pdb

from sklearn.neighbors import NearestNeighbors
import numpy as np

def _extract_numpy_doc_embs(nvsm):
    return nvsm.doc_emb.weight.detach().cpu().numpy()

def evaluate(nvsm, device, eval_loader, recalls, loss_function, lamb):
    doc_embs = _extract_numpy_doc_embs(nvsm)
    nn_docs  = NearestNeighbors(n_neighbors = max(recalls), metric = 'cosine')
    nn_docs.fit(doc_embs)
    total_query  = 0
    doc_hit_at_k = [0] * len(recalls)
    for i, (n_grams, doc_ids) in enumerate(eval_loader):
        total_query += n_grams.shape[0]
        n_grams      = n_grams.to(device)
        n_gram_embs  = nvsm.stand_projection(n_grams)
        n_gram_embs  = n_gram_embs.detach().cpu().numpy()
        neighbors    = nn_docs.kneighbors(n_gram_embs, return_distance = False)
        doc_ids      = doc_ids.numpy().reshape(-1, 1)
        hits         = neighbors == doc_ids
        for i, k in enumerate(recalls):
            doc_found_in_k_neigh  = np.any(hits[:, :k], axis = 1)
            doc_hit_at_k[i]      += doc_found_in_k_neigh.sum()

    return [hit_number / total_query for hit_number in doc_hit_at_k]
