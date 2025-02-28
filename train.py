from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load dataset
with open("data.json", "r") as f:
    data = json.load(f)

# Convert dataset into proper format
dataset = Dataset.from_dict({
    "input_text": [f"Question: {d['question']} Answer:" for d in data],
    "target_text": [d["answer"] for d in data]
})

# Load T5 model and tokenizer
model_name = "t5-small"  # You can also try "t5-base" for better results
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(examples["target_text"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# Apply tokenization
dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./trained_model",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=500
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# Save trained model and tokenizer
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")

print("✅ Training Complete! Model saved to ./trained_model")
