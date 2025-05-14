import torch
import torch.nn as nn
import torch.nn.functional as F
import math

DROPOUT = 0.1
MAX_SEQ_LEN = 40

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.wq(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.fc(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.dropout2 = nn.Dropout(DROPOUT)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.dropout = nn.Dropout(DROPOUT)
        self.max_seq_len = max_seq_len
        self.norm = nn.LayerNorm(d_model)

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, src_tokens, return_hidden=False):
        x = self.embedding(src_tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.fc_out(x)
        if return_hidden:
            return logits, x  # logits, hidden states
        return logits

class Critic(nn.Module):
    def __init__(self, d_model):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, hidden_states):
        x = F.relu(self.fc1(hidden_states))
        value = self.fc2(x)
        return value.squeeze(-1)

class RewardModel(nn.Module):
    def __init__(self, d_model):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)

    def forward(self, hidden_states, mask=None):
        # hidden_states: [batch, seq_len, d_model]
        # mask: [batch, seq_len] (1 for valid, 0 for pad)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            hidden_states = hidden_states * mask
            summed = hidden_states.sum(1)
            count = mask.sum(1).clamp(min=1e-6)
            pooled = summed / count
        else:
            pooled = hidden_states.mean(1)
        x = F.relu(self.fc1(pooled))
        reward = self.fc2(x)
        return reward.squeeze(-1)
