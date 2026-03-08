import torch
import json
import os
import wandb

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# -----------------------------
# GPU optimization
# -----------------------------

torch.backends.cuda.matmul.allow_tf32 = True

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# -----------------------------
# Initialize W&B
# -----------------------------

wandb.init(
    project="llm-finetuning-platform",
    config={
        "model": "microsoft/phi-2",
        "dataset": "dolly-15k",
        "epochs": 3,
        "batch_size": 2,
        "learning_rate": 2e-4,
        "lora_rank": 16
    }
)

# -----------------------------
# Load processed dataset
# -----------------------------

print("\n[1/5] Loading Dataset...")

with open("data/processed/train.json") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

print("Dataset size:", len(dataset))

# -----------------------------
# Load tokenizer
# -----------------------------

print("\n[2/5] Loading Tokenizer...")

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_name)

# Fix missing pad token
config.pad_token_id = tokenizer.eos_token_id


# -----------------------------
# Tokenization
# -----------------------------

def tokenize(example):

    text = example["prompt"] + example["response"]

    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512
    )

dataset = dataset.map(tokenize)

# -----------------------------
# QLoRA configuration
# -----------------------------

print("\n[3/5] Loading Model with QLoRA...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=bnb_config,
    device_map="auto"
)

# prepare model for k-bit training

model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA configuration
# -----------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

model.gradient_checkpointing_enable()

# -----------------------------
# Training arguments
# -----------------------------

print("\n[4/5] Setting Training Configuration...")

training_args = TrainingArguments(

    output_dir="outputs",

    per_device_train_batch_size=2,

    gradient_accumulation_steps=8,

    num_train_epochs=3,

    learning_rate=2e-4,

    logging_steps=10,

    save_steps=500,

    save_total_limit=3,

    report_to="wandb",

    fp16=True,

    optim="paged_adamw_8bit",

    logging_dir="logs"
)

# -----------------------------
# Trainer
# -----------------------------

print("\n[5/5] Starting Training Pipeline...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# -----------------------------
# Resume from checkpoint if exists
# -----------------------------

checkpoint = None

if os.path.isdir("outputs"):
    checkpoints = [
        os.path.join("outputs", d)
        for d in os.listdir("outputs")
        if "checkpoint" in d
    ]

    if checkpoints:
        checkpoint = max(checkpoints, key=os.path.getctime)
        print("Resuming from checkpoint:", checkpoint)

trainer.train(resume_from_checkpoint=checkpoint)

# -----------------------------
# Save model
# -----------------------------

print("\nSaving fine-tuned model...")

trainer.save_model("models/phi2-lora")

tokenizer.save_pretrained("models/phi2-lora")

print("\nTraining complete!")