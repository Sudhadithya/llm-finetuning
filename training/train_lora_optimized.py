import torch
import wandb
import json
import os

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    AutoConfig
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

print("=" * 80)
print("LLM Fine-Tuning Pipeline (RTX 4060 Optimized - FIXED)")
print("=" * 80)

# Verify CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Initialize experiment tracking
wandb.init(
    project="llm-finetuning-platform",
    config={
        "model": "microsoft/phi-2",
        "epochs": 2,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "learning_rate": 2e-4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "max_seq_length": 512,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    }
)

print("\n[1/6] Loading Dataset...")

# Load dataset
with open("data/processed/train.json") as f:
    data = json.load(f)

print(f"Loaded {len(data)} samples")

dataset = Dataset.from_list(data)

# Split for validation
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# Load tokenizer
print("\n[2/6] Loading Tokenizer...")

model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(model_name)
config.pad_token_id = tokenizer.pad_token_id

# Tokenization function
def tokenize(example):
    text = example["prompt"] + example["response"]
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors=None
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

print("Tokenizing dataset...")
train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize, remove_columns=val_dataset.column_names)

# Load model with 8-bit quantization
print("\n[3/6] Loading Model with 8-bit Quantization...")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)

print("\n[4/6] Applying LoRA Adapters...")

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments optimized for RTX 4060
print("\n[5/6] Setting Training Arguments...")

training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=5,
    eval_steps=50,
    save_steps=100,
    eval_strategy="steps",
    report_to="wandb",
    fp16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    warmup_steps=10,
    lr_scheduler_type="linear",
    seed=42,
    dataloader_num_workers=0,  # CRITICAL: Windows compatibility
    dataloader_pin_memory=False,  # CRITICAL: Fix for slow data loading
    dataloader_persistent_workers=False,  # CRITICAL: Disable persistent workers
)

print("\n[6/6] Training...")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer with fixed data loading
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Train
print("\n" + "=" * 80)
print("Starting Training...")
print("=" * 80 + "\n")

trainer.train()

# Save model
print("\n" + "=" * 80)
print("Saving Fine-Tuned Model...")
print("=" * 80 + "\n")

model.save_pretrained("models/phi2-lora")
tokenizer.save_pretrained("models/phi2-lora")

print("✅ Training Complete!")
print("Model saved to: models/phi2-lora/")

wandb.finish()