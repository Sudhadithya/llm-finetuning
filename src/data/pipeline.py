import os
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer
from src.core.config import ExperimentConfig

class DataPipeline:
    def __init__(self, config: ExperimentConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        
    def load_and_tokenize(self) -> DatasetDict:
        dataset_path = self.config.dataset_path
        train_file = os.path.join(dataset_path, "train.jsonl")
        val_file = os.path.join(dataset_path, "val.jsonl")
        
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            raise FileNotFoundError(f"Missing train.jsonl or val.jsonl in {dataset_path}")
            
        dataset = load_dataset('json', data_files={'train': train_file, 'validation': val_file})
        
        max_length = self.config.training.max_length
        
        def tokenize(example):
            text = example["prompt"] + "\n" + example["response"] + self.tokenizer.eos_token
            return self.tokenizer(
                text, 
                truncation=True, 
                padding="max_length", 
                max_length=max_length
            )
            
        tokenized_dataset = dataset.map(tokenize, remove_columns=["prompt", "response"])
        return tokenized_dataset
