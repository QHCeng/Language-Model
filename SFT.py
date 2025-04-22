from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
dataset = load_dataset("json", data_files="covid_data.json")["train"]
val_dataset = load_dataset("json", data_files="val_data.json")["train"]
instruction_template = "### Human:"
response_template = "### Assistant:"

def format_text(example):
    text = f"{instruction_template} {example['instruction']}\n{response_template} {example['response']}"
    return {"text": text, "response": example["response"]} 

dataset = dataset.map(format_text)


model_path = "./pretrained_model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
new_tokens = [instruction_template, response_template]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
    output_dir="./covid_sft",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=1e-5,
    logging_steps=50,
    evaluation_strategy="steps",  
    save_steps=1000,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset
)

trainer.train()