import torch
from nvsm_lstm import QueryEncoder

encoder = QueryEncoder(
    n_layer     = 2,
    n_token     = 32,
    dim_tok_emb = 22,
    n_hidden    = 16,
    dim_doc_emb = 11,
    dropout     = 0.1
)

X = torch.randint(32, (10, 4))
hidden = encoder.init_hidden(4)
result = encoder(X, hidden)
