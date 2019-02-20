import torch
import torch.nn as nn
import torch.nn.functional as F

class NVSM(nn.Module):
    def __init__(self, n_doc, n_tok, dim_doc_emb, dim_tok_emb, neg_sampling_rate,
                 pad_token_id):
        super(NVSM, self).__init__()
        self.doc_emb           = nn.Embedding(n_doc, embedding_dim = dim_doc_emb)
        self.tok_emb           = nn.Embedding(n_tok, embedding_dim = dim_tok_emb)
        self.tok_to_doc        = nn.Linear(dim_tok_emb, dim_doc_emb) # to replace by a LSTM
        self.bias              = nn.Parameter(torch.Tensor(dim_doc_emb))
        self.batchnorm         = nn.BatchNorm1d(dim_doc_emb)
        self.neg_sampling_rate = neg_sampling_rate
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

    def score(self, query, document):
        '''
        Computes the cosine similarity between a query and a document embedding.
        This method corresponds to the function 'score' in the article.
        '''
        # batch dot product using batch matrix multiplication
        num   = torch.bmm(query.unsqueeze(1), document.unsqueeze(-1))
        denum = torch.norm(query, dim = 1) * torch.norm(document, dim = 1)

        return num / denum

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

    def representation_similarity(self, query, document):
        '''
        Computes the similarity between a query and a document. This method corresponds
        to the function 'P' in the article.
        '''
        document_emb  = self.doc_emb(document)
        query_proj    = self.stand_projection(query)
        # If we have a single document to match against each query, we have
        # to reshape the tensor to compute a simple dot product.
        # Otherwise, we compute a simple matrix multiplication to match the
        # query against each document.
        if len(document_emb.shape) == 2:
            document_emb = document_emb.unsqueeze(1)
        if len(query_proj.shape) == 2:
            query_proj = query_proj.unsqueeze(-1)
        dot_product   = torch.bmm(document_emb, query_proj)
#        dot_product   = torch.bmm(document_emb, query_proj.unsqueeze(-1))
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
    loss        = -output_term

    return loss
