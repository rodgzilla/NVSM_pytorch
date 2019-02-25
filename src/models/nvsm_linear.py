import torch
import torch.nn as nn
import torch.nn.functional as F

from nvsm import NVSM

class NVSMLinear(NVSM):
    def __init__(self, n_doc, n_tok, dim_doc_emb, dim_tok_emb, neg_sampling_rate,
                 pad_token_id):
        super(NVSMLinear, self).__init__(n_doc, dim_doc_emb, neg_sampling_rate)
        self.tok_emb           = nn.Embedding(n_tok, embedding_dim = dim_tok_emb)
        self.tok_to_doc        = nn.Linear(dim_tok_emb, dim_doc_emb)
        self.batchnorm         = nn.BatchNorm1d(dim_doc_emb)
        self.pad_token_id      = pad_token_id

    def query_to_tensor(self, query):
        '''
        Computes the average of the word embeddings of the query. This method
        corresponds to the function 'g' in the article.
        '''
        # Create a mask to ignore padding embeddings
        query_mask    = (query != self.pad_token_id).float()
        # Compute the number of tokens in each query to properly compute the
        # average
        tok_by_input  = query_mask.sum(dim = 1)
        query_tok_emb = self.tok_emb(query)
        query_tok_emb = query_tok_emb * query_mask.unsqueeze(-1)
        # Compute the average of the embeddings
        query_emb     = query_tok_emb.sum(dim = 1) / tok_by_input.unsqueeze(-1)

        return query_emb

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

    query_embedding = stand_projection

def loss_function(nvsm, pred, lamb):
    '''
    Computes the loss according to the formula (8) in the article.
    '''
    output_term = pred.mean()
    sum_square  = lambda m: (m.weight * m.weight).sum()
    reg_term    = sum_square(nvsm.tok_emb) + \
                  sum_square(nvsm.doc_emb) + \
                  sum_square(nvsm.tok_to_doc)
    loss        = -output_term + (lamb / (2 * pred.shape[0])) * reg_term

    return loss
