import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size):
        super(GroupQueryAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.group_size = group_size
        self.scale = self.head_dim**-0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim // group_size)
        self.value_proj = nn.Linear(embed_dim, embed_dim // group_size)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        query = self.query_proj(query)  # (batch_size, seq_len, embed_dim)
        key = self.key_proj(key)        # (batch_size, seq_len, embed_dim // group_size)
        value = self.value_proj(value)  # (batch_size, seq_len, embed_dim // group_size)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.group_size, self.head_dim).transpose(1, 2)      # (batch_size, group_size, seq_len, head_dim)
        value = value.view(batch_size, seq_len, self.group_size, self.head_dim).transpose(1, 2)  # (batch_size, group_size, seq_len, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # (batch_size, seq_len, embed_dim)

        # Project output
        output = self.out_proj(attn_output)
        return output

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, group_size, ff_dim):
        super(DecoderBlock, self).__init__()
        self.self_attn = GroupQueryAttention(embed_dim, num_heads, group_size)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)
        x = x + self.norm1(attn_output)
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.norm2(ffn_output)
        return x

class DecoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, group_size, ff_dim, num_layers):
        super(DecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, group_size, ff_dim) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        logits = self.fc_out(x)
        return logits

