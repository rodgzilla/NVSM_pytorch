import pdb

import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel

from nvsm import NVSM

class QueryEncoder(nn.Module):
    def __init__(self, pretrained_model, dim_doc_emb):
        super(QueryEncoder, self).__init__()
        self.bert          = BertModel.from_pretrained(pretrained_model)
        hidden_size        = self.bert.config.hidden_size
        self.hidden_to_doc = nn.Linear(hidden_size, dim_doc_emb)

    def forward(self, query):
        segments_ids             = torch.zeros_like(query)
        enc_layer, pooled_output = self.bert(
            query,
            segments_ids,
            output_all_encoded_layers = False
        )
        query_emb                = self.hidden_to_doc(pooled_output)
        # pdb.set_trace()

        return query_emb

class NVSMBERT(NVSM):
    def __init__(self, pretrained_model, n_doc, dim_doc_emb, neg_sampling_rate):
        super(NVSMBERT, self).__init__(n_doc, dim_doc_emb, neg_sampling_rate)
        self.query_encoder     = QueryEncoder(
            pretrained_model = pretrained_model,
            dim_doc_emb      = dim_doc_emb
        )
        self.batchnorm         = nn.BatchNorm1d(dim_doc_emb)

    def query_embedding(self, query):
        return self.query_encoder(query)

def loss_function(nvsm, pred, lamb):
    '''
    Computes the loss function which is just a mean of
    the negative log likehood.
    '''
    return -pred.mean()
