import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# Define LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, original_dim, rank):
        super(LoRALayer, self).__init__()
        self.rank = rank
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.randn(original_dim, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, original_dim))

    def forward(self, x):
        # Apply low-rank adaptation: x + (x @ A @ B)
        return x + x @ self.lora_A @ self.lora_B

# Modify BERT's self-attention mechanism with LoRA
class LoRABertSelfAttention(nn.Module):
    def __init__(self, config, rank):
        super(LoRABertSelfAttention, self).__init__()
        self.original_self_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
        )
        # Add LoRA layers for query, key, and value projections
        self.lora_query = LoRALayer(config.hidden_size, rank)
        self.lora_key = LoRALayer(config.hidden_size, rank)
        self.lora_value = LoRALayer(config.hidden_size, rank)

    def forward(self, query, key, value, attention_mask=None):
        # Apply LoRA to query, key, and value
        query = self.lora_query(query)
        key = self.lora_key(key)
        value = self.lora_value(value)
        # Pass through original self-attention
        output, _ = self.original_self_attention(query, key, value, attn_mask=attention_mask)
        return output

# Wrap BERT with LoRA
class LoRABertModel(nn.Module):
    def __init__(self, config, rank):
        super(LoRABertModel, self).__init__()
        self.bert = BertModel(config)
        # Replace self-attention layers with LoRA self-attention
        for layer in self.bert.encoder.layer:
            layer.attention.self = LoRABertSelfAttention(config, rank)

    def forward(self, input_ids, attention_mask=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask)