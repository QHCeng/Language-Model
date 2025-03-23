import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random

# Hyperparameters
EMBEDDING_DIM = 100
WINDOW_SIZE = 2
NEGATIVE_SAMPLES = 5
LEARNING_RATE = 0.01
EPOCHS = 10
BATCH_SIZE = 32


def generate_training_data(sentences, window_size):
    data = []
    for sentence in sentences:
        sentence_length = len(sentence)
        for i, center_word in enumerate(sentence):
            for j in range(max(0, i - window_size), min(sentence_length, i + window_size + 1)):
                if i != j:
                    context_word = sentence[j]
                    data.append((vocab[center_word], vocab[context_word]))
    return data

training_data = generate_training_data(sentences, WINDOW_SIZE)

# Negative sampling
def negative_sampling(vocab_size, target_word, num_samples):
    negative_samples = []
    while len(negative_samples) < num_samples:
        sample = random.randint(0, vocab_size - 1)
        if sample != target_word:
            negative_samples.append(sample)
    return negative_samples

# Skip-gram model
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNegSampling, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target_word, context_word, negative_samples):
        # Positive sample
        target_embed = self.target_embeddings(target_word)  # (batch_size, embedding_dim)
        context_embed = self.context_embeddings(context_word)  # (batch_size, embedding_dim)
        positive_score = torch.sum(target_embed * context_embed, dim=1)  # (batch_size,)

        # Negative samples
        negative_embed = self.context_embeddings(negative_samples)  # (batch_size, num_samples, embedding_dim)
        negative_score = torch.bmm(negative_embed, target_embed.unsqueeze(2)).squeeze(2)  # (batch_size, num_samples)

        # Concatenate positive and negative scores
        scores = torch.cat([positive_score.unsqueeze(1), negative_score], dim=1)  # (batch_size, 1 + num_samples)
        return scores

class SkipGramDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        target_word, context_word = self.data[idx]
        negative_samples = negative_sampling(vocab_size, target_word, NEGATIVE_SAMPLES)
        return torch.tensor(target_word), torch.tensor(context_word), torch.tensor(negative_samples)

dataset = SkipGramDataset(training_data)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = SkipGramNegSampling(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for target_word, context_word, negative_samples in dataloader:
        optimizer.zero_grad()

        # Forward pass
        scores = model(target_word, context_word, negative_samples)  # (batch_size, 1 + num_samples)

        # Labels: 1 for positive sample, 0 for negative samples
        labels = torch.zeros(scores.shape[0], dtype=torch.long)  # (batch_size,)
        loss = criterion(scores, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader)}")

