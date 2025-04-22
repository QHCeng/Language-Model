from transformers import AutoModel, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"], 
    lora_dropout=0.1,
    task_type="FEATURE_EXTRACTION"
)
model = get_peft_model(model, lora_config)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(5):
    for words1, words2 in dataloader:
        optimizer.zero_grad()
        
        vecs1 = torch.stack([get_word_vector(model, tokenizer, word) for word in words1])
        vecs2 = torch.stack([get_word_vector(model, tokenizer, word) for word in words2])

        loss = cosine_loss(vecs1, vecs2, 1.0)
        loss.backward()
        optimizer.step()
