import torch
import torch.nn as nn

class JointEmbedding(nn.Module):
    def __init__(self, vocab_size, size):
        super(JointEmbedding, self).__init__()
        
        self.size = size
        
        self.token_emb = nn.Embedding(vocab_size, size)
        self.segment_emb = nn.Embedding()
        
        self.norm = nn.LayerNorm(size)
        
    def forward(self, input_tensor):
        sentence_size = input_tensor.size(-1)
        

class AttentionHead(nn.Module):
    def __init__(self, dim_inp, dim_out):
        super(AttentionHead, self).__init__()
        
        self.dim_inp = dim_inp
        self.dim_out = dim_out
        
        self.q = nn.Linear(dim_inp, dim_out)
        self.k = nn.Linear(dim_inp, dim_out)
        self.v = nn.Linear(dim_inp, dim_out)
        
    
    def forward(self, input_tensor: torch.Tensor, attention_mask: torch.Tensor=None):
        query, key, value = self.q(input_tensor), self.k(input_tensor), self.v(input_tensor)
        
        scale = query.size(1) ** 0.5
        scores = torch.bmm(query, key.transpose(1, 2)) / scale
        
        scores = scores.masked_fill_(attention_mask, -1e9)
        attn = nn.softmax(scores, dim=-1)
        context = torch.bmm(attn, value)
        
        return context
        