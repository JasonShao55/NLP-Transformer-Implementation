# add all  your Encoder and Decoder code here

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Replacement of nn.MultiheadAttention
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


class FeedForward(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, forward_expansion * embed_size)
        self.fc2 = nn.Linear(forward_expansion * embed_size, embed_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.attention = CustomMultiheadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attention,atten_map= self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out,atten_map

class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, forward_expansion, dropout)
             for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        attn_maps = []

        for layer in self.layers:
            out, attn_map = layer(out, mask)
            attn_maps.append(attn_map)
        
        return out,attn_maps

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
    
    def forward(self, src, src_mask):
        enc_src,attn_maps = self.encoder(src, src_mask)
        return enc_src,attn_maps

class TransformerClassifier(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, num_classes):
        super(TransformerClassifier, self).__init__()
        self.transformer = Transformer(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.fc = nn.Linear(embed_size, num_classes)       
    
    def forward(self, x, mask=None):
        enc_out,attn_maps  = self.transformer(x, mask)

        # original classifier
        enc_out = enc_out.mean(dim=1)
        out = self.fc(enc_out) # output to num_classes reults, don't add anything else
        return out,attn_maps
        
        # Use the embedding of the CLS token for classification
        # cls_embedding = enc_out[:, 0, :]  # CLS token is the first token
        # out = self.fc(cls_embedding)
        # return out, attn_maps
    



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AliBiAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(AliBiAttention, self).__init__()
        self.attn = CustomMultiheadAttention(d_model, nhead)
        self.bias = nn.Parameter(torch.tril(torch.ones(d_model, d_model)).unsqueeze(0))

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.attn(query, key, value)
        attn_output += self.bias
        return attn_output, attn_output_weights

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = AliBiAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.Embedding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
