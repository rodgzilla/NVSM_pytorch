import torch
from torch.utils.data import TensorDataset, DataLoader

def create_query_dataset(queries_text, tokenizer):
    queries_tokens    = []
    cls_tok_id        = tokenizer.vocab['[CLS]']
    sep_tok_id        = tokenizer.vocab['[SEP]']
    create_padding    = lambda l : [0] * (max_len - len(l))
    tokenized_queries = [tokenizer.tokenize(query) for query in queries_text]
    tokenized_queries = [query[:510] for query in tokenized_queries]
    tok_idx_queries   = [
        [cls_tok_id] + [tokenizer.vocab[tok] for tok in query_toks] + [sep_tok_id]
        for query_toks in tokenized_queries
    ]
    max_len = min(512, max(len(query) for query in tok_idx_queries))

    tok_idx_queries   = [
        query + create_padding(query)
        for query in tok_idx_queries
    ]
    attention_masks   = [
        [1] * len(query) + create_padding(query)
        for query in tok_idx_queries
    ]
    query_tensor      = torch.tensor(tok_idx_queries)
    mask_tensor       = torch.tensor(attention_masks)
    dataset           = TensorDataset(query_tensor, mask_tensor)

    return dataset

def evaluate_queries_bert(nvsm, queries_text, doc_names, tokenizer,
                          batch_size, device):
    query_dataset    = create_query_dataset(queries_text, tokenizer)
    test_loader      = DataLoader(query_dataset, batch_size = batch_size)
    results          = []
    document_indices = torch.stack([torch.arange(len(doc_names))] * batch_size)
    document_indices = document_indices.to(device)
    for (queries, mask) in test_loader:
        queries = queries.to(device)
        mask    = mask.to(device)
        result  = nvsm.representation_similarity(
            queries,
            document_indices[:queries.shape[0]],
            attention_mask = mask
        )
        results.extend(list(result.argmax(dim = 1).cpu().numpy()))

    return results
