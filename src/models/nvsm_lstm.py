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

    def query_to_tensor(self, query):
        hidden = self.query_encoder.init_hidden(query.shape[0])

        return self.query_encoder(query, hidden)

    def normalize_query_tensor(self, query_tensor):
        '''
        Divides each query tensor by its L2 norm. This method corresponds to
        the function 'norm' in the article.
        '''
        norm = torch.norm(query_tensor, dim = 1) # we might have to detach this value
                                                 # from the computation graph.
        return query_tensor / norm.unsqueeze(-1)

    def query_to_doc_space(self, query):
        '''
        Projects a query vector into the document vector space. This method corresponds
        to the function 'f' in the article.
        '''
        return self.tok_to_doc(query)

    def non_stand_projection(self, n_gram):
        '''
        Computes the non-standard projection of a n-gram into the document vector
        space. This method corresponds to the function 'T^~' in the article.
        '''
        n_gram_tensor      = self.query_to_tensor(n_gram)
        norm_n_gram_tensor = self.normalize_query_tensor(n_gram_tensor)
        projection         = self.query_to_doc_space(norm_n_gram_tensor)

        return projection

    def non_stand_projection(self, n_gram):
        '''
        Computes the non-standard projection of a n-gram into the document vector
        space. This method corresponds to the function 'T^~' in the article.
        '''
        n_gram_tensor      = self.query_to_tensor(n_gram)

        return n_gram_tensor

    def stand_projection(self, batch):
        '''
        Computes the standard projection of a n-gram into document vector space with
        a hardtanh activation. This method corresponds to the function 'T' in the
        article.
        '''
        non_stand_proj = self.non_stand_projection(batch)
        bn             = self.batchnorm(non_stand_proj)
        activation     = F.hardtanh(bn)

        return activation

    query_embedding = query_to_tensor


def loss_function(nvsm, pred, lamb):
    '''
    Computes the loss function which is just a mean of
    the negative log likehood.
    '''
    return -pred.mean()
