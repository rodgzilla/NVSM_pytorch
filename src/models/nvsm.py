from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

class NVSM(ABC, nn.Module):
    def __init__(self, n_doc, dim_doc_emb, neg_sampling_rate):
        super(NVSM, self).__init__()
        self.n_doc = n_doc
        self.doc_emb           = nn.Embedding(n_doc, embedding_dim = dim_doc_emb)
        self.neg_sampling_rate = neg_sampling_rate

    @abstractmethod
    def query_embedding(self, query, **kwargs):
        pass

    def score(self, query, document):
        '''
        Computes the cosine similarity between a query and a document embedding.
        This method corresponds to the function 'score' in the article.
        '''
        # batch dot product using batch matrix multiplication
        num   = torch.bmm(query.unsqueeze(1), document.unsqueeze(-1))
        denum = torch.norm(query, dim = 1) * torch.norm(document, dim = 1)

        return num / denum

    def representation_similarity(self, query, document, **query_kwargs):
        '''
        Computes the similarity between a query and a document. This method corresponds
        to the function 'P' in the article.
        '''
        query_proj   = self.query_embedding(query, **query_kwargs)
        document_emb = self.doc_emb(document)
        # If we have a single document to match against each query, we have
        # to reshape the tensor to compute a simple dot product.
        # Otherwise, we compute a simple matrix multiplication to match the
        # query against each document.
        if len(document_emb.shape) == 2:
            document_emb = document_emb.unsqueeze(1)
        if len(query_proj.shape) == 2:
            query_proj = query_proj.unsqueeze(-1)
        dot_product   = torch.bmm(document_emb, query_proj)
        similarity    = torch.sigmoid(dot_product)

        return similarity.squeeze()

    def forward(self, query, document):
        '''
        Approximates the probability of document given query by uniformly sampling
        constrastive examples. This method corresponds to the 'P^~' function in the
        article.
        '''
        # Positive term, this should be maximized as it indicates how similar the
        # correct document is to the query
        pos_repr = self.representation_similarity(query, document)

        # Sampling uniformly 'self.neg_sampling_rate' documents to compute the
        # negative term. We first randomly draw the indices of the documents and
        # then we compute the similarity with the query.
        device          = document.device
        z               = self.neg_sampling_rate # corresponds to the z variable in
                                                 # the article
        n_docs          = self.doc_emb.num_embeddings
        neg_sample_size = (query.size(0), z)
        neg_sample      = torch.randint(low = 0, high = n_docs, size = neg_sample_size)
        neg_sample      = neg_sample.to(device)
        neg_repr        = self.representation_similarity(query, neg_sample)

        # Probability computation
        positive_term = torch.log(pos_repr)
        # -inf used to come from this line as neg_repr has 0 values. Adding
        # a an epsilon prevent torch.log to be applied on 0 input.
        negative_term = torch.log(1 - neg_repr + 1e-40).sum(dim = 1)
        # negative_term = torch.log(1 - neg_repr).sum(dim = 1)
        proba         = ((z + 1) / (2 * z)) * (z * positive_term + negative_term)

        return proba
