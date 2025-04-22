from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["articles.txt"],
    vocab_size=30_000,  
    min_frequency=2,
    special_tokens=["<|endoftext|>", "<|pad|>", "[UNK]"]
)
tokenizer.save_model("custom_tokenizer")
tokenizer = GPT2TokenizerFast.from_pretrained("custom_tokenizer")
tokenizer.pad_token = "<|pad|>"
train_loader = DataLoader(dataset["train"], batch_size=8, shuffle=True)
training_args = TrainingArguments(
    output_dir="./pretrain",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    logging_steps=100,
    learning_rate=5e-5,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)