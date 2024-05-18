# add all  your Encoder and Decoder code here

import torch
import torch.nn as nn
import torch.nn.functional as F
class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        seq_len, batch_size, embed_dim = query.size()
        num_heads = self.num_heads

        # Linear projections
        Q = self.query(query)  # (seq_len, batch_size, embed_dim)
        K = self.key(key)      # (seq_len, batch_size, embed_dim)
        V = self.value(value)  # (seq_len, batch_size, embed_dim)

        # Transpose for multi-head attention: (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)
        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # Split into multiple heads and reshape to (batch_size * num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, self.head_dim)
        K = K.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, self.head_dim)
        V = V.view(batch_size, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, self.head_dim)

        # Scaled dot-product attention
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_dim ** 0.5)  # (batch_size * num_heads, seq_len, seq_len)

        if attn_mask is not None:
            # Ensure attn_mask has the same shape as attn_scores
            attn_mask = attn_mask.unsqueeze(0).expand(batch_size * num_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)     
        attn_probs = self.attention_dropout(attn_probs)     # ERROR if called after softmax, won't sum to 1

        attn_output = torch.bmm(attn_probs, V)  # (batch_size * num_heads, seq_len, head_dim)

        # Reshape back to (batch_size, seq_len, embed_dim)
        attn_output = attn_output.view(batch_size, num_heads, seq_len, self.head_dim).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Transpose back to original shape: (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        attn_output = attn_output.transpose(0, 1)

        # Reshape attn_probs to (num_heads, batch_size, seq_len, seq_len) and then to (batch_size, num_heads, seq_len, seq_len)
        attn_probs = attn_probs.view(batch_size, num_heads, seq_len, seq_len)
        attn_map = attn_probs.mean(dim=1)  # Average over heads

        # Final linear projection
        output = self.out(attn_output)

        return output, attn_map


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        #self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0) # substitute with with my own MultiheadAttention
        self.self_attn = CustomMultiheadAttention(embed_dim, num_heads,dropout=0) # substitute with with my own MultiheadAttention
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # Self-attention
        attn_output, attn_map = self.self_attn(x, x, x, attn_mask=mask) # for nn.MultiheadAttention
        #attn_output, attn_map = self.self_attn(x, attn_mask=mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        
        return x,attn_map

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, embed_dim, num_heads, ff_hidden_dim, num_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_hidden_dim, dropout) 
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        seq_len, batch_size = x.shape
        positions = torch.arange(0, seq_len).unsqueeze(1).expand(seq_len, batch_size).to(x.device)
        # batch_size, seq_len = x.shape
        # positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len).to(x.device)
        
        x = self.token_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        attn_maps = []
        for layer in self.layers:
            x, attn_map = layer(x, mask)
            attn_maps.append(attn_map)
        
        logits = self.fc_out(x)
        #return F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        return logits, attn_maps


# 2D mask: (seq_len, seq_len) 
def create_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

# 3D mask: (batch_size * num_heads, seq_len, seq_len)
# def create_mask(batch_size, num_heads, seq_len):
#     mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
#     mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
#     mask = mask.expand(batch_size, num_heads, seq_len, seq_len).reshape(batch_size * num_heads, seq_len, seq_len)
#     return mask
