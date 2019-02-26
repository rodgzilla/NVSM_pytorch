import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsm import NVSM

class QueryEncoder(nn.Module):
    def __init__(self, n_layer, n_token, dim_tok_emb, n_hidden, dim_doc_emb, dropout):
        super(QueryEncoder, self).__init__()
        self.drop          = nn.Dropout(dropout)
        self.emb_layer     = nn.Embedding(n_token, dim_tok_emb)
        self.lstm          = nn.LSTM(dim_tok_emb, n_hidden, n_layer)
        self.hidden_to_doc = nn.Linear(n_hidden, dim_doc_emb)
        self.n_layer       = n_layer
        self.n_hidden      = n_hidden

    def init_hidden(self, batch_size):
        return (
            self.hidden_to_doc.weight.new_zeros(
                self.n_layer,
                batch_size,
                self.n_hidden
            ),
            self.hidden_to_doc.weight.new_zeros(
                self.n_layer,
                batch_size,
                self.n_hidden
            )
        )

    def forward(self, query, hidden):
        emb              = self.emb_layer(query)
        emb              = self.drop(emb)
        emb              = torch.transpose(emb, 0, 1)
        out_lstm, hidden = self.lstm(emb, hidden)
        out_lstm         = out_lstm[-1] # only consider the encoding of the final token
        out_lstm         = self.drop(out_lstm)
        query_emb        = self.hidden_to_doc(out_lstm)

        return query_emb

class NVSMLSTM(NVSM):
    def __init__(self, n_doc, n_tok, dim_doc_emb, dim_tok_emb, neg_sampling_rate,
                 n_layer, n_hidden, dropout):
        super(NVSMLSTM, self).__init__(n_doc, dim_doc_emb, neg_sampling_rate)
        self.query_encoder     = QueryEncoder(
            n_layer     = n_layer,
            n_token     = n_tok,
            dim_tok_emb = dim_tok_emb,
            n_hidden    = n_hidden,
            dim_doc_emb = dim_doc_emb,
            dropout     = dropout
        )
        self.batchnorm         = nn.BatchNorm1d(dim_doc_emb)

    def query_embedding(self, query):
        hidden = self.query_encoder.init_hidden(query.shape[0])

        return self.query_encoder(query, hidden)

def loss_function(nvsm, pred, lamb):
    '''
    Computes the loss function which is just a mean of
    the negative log likehood.
    '''
    return -pred.mean()
