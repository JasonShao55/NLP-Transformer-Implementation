# add all  your Encoder and Decoder code here

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(CustomMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(1)

        # Linear projections
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Reshape into (num_heads, batch_size, seq_length, head_dim)
        Q = Q.view(-1, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, batch_size, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, batch_size, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


# n_embd = 64  # Embedding dimension
# block_size = 32  # Maximum context length for predictions
# dropout=0.1

# class Head(nn.Module):
#     """ one head of self-attention """

#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size, bias=False)
#         self.query = nn.Linear(n_embd, head_size, bias=False)
#         self.value = nn.Linear(n_embd, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x,mask=None):
#         # input of size (batch, time-step, channels)
#         # output of size (batch, time-step, head size)
#         B,T,C = x.shape
#         #print("forward x size:",x.size())
#         k = self.key(x)   # (B,T,hs)
#         q = self.query(x) # (B,T,hs)
#         # compute attention scores ("affinities")
#         wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
#         #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
#         # Apply attention mask if provided
#         if mask is not None:
#             #mask = mask[:, :T, :T]  # Ensure mask is the same size as wei
#             #print("wei size:",wei.size())
#             wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
#         wei = F.softmax(wei, dim=-1) # (B, T, T)
#         wei = self.dropout(wei)
#         # perform the weighted aggregation of the values
#         v = self.value(x) # (B,T,hs)
#         out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
#         return out,wei
# class MultiHeadAttention(nn.Module):
#     """ multiple heads of self-attention in parallel """

#     def __init__(self, n_embd, num_heads, dropout):
#         super().__init__()
#         head_size = n_embd // num_heads
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(head_size * num_heads, n_embd)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, x, attn_mask=None):
#         head_outputs = []
#         attn_maps = []
#         for h in self.heads:
#             out, attn_map = h(x, attn_mask)
#             head_outputs.append(out)
#             attn_maps.append(attn_map)
#         out = torch.cat(head_outputs, dim=-1)
#         out = self.dropout(self.proj(out))
#         return out, attn_maps


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=0) # substitute with with my own MultiheadAttention
        #self.self_attn = CustomMultiheadAttention(embed_dim, num_heads,dropout=dropout) # substitute with with my own MultiheadAttention
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

# def create_mask(batch_size, seq_len):
#     #print("create_mask:",batch_size,seq_len)
#     mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()
#     return mask
def create_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask


# def create_mask(batch_size, seq_len):
#     mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
#     return mask.unsqueeze(0)  # Add a batch dimension