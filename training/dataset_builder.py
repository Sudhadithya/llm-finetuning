"""
Dataset Builder for LLM Fine-Tuning
Downloads and processes the Databricks Dolly dataset into instruction format
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

def create_directories():
    """Create necessary directories"""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    print("✓ Created data directories")

def load_dolly_dataset():
    """Load Databricks Dolly dataset from HuggingFace"""
    print("\n[1/4] Loading Databricks Dolly dataset...")
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k")
        print(f"✓ Loaded dataset with {len(dataset['train'])} samples")
        return dataset
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        raise

def process_dataset(dataset):
    """Process dataset into instruction format"""
    print("\n[2/4] Processing dataset into instruction format...")
    
    processed = []
    
    for idx, row in enumerate(dataset["train"]):
        instruction = row["instruction"]
        context = row["context"]
        response = row["response"]
        
        # Create prompt
        if context:
            prompt = f"""Instruction: {instruction}
Context: {context}

Answer:"""
        else:
            prompt = f"""Instruction: {instruction}

Answer:"""
        
        processed.append({
            "prompt": prompt,
            "response": response,
            "category": row.get("category", "general")
        })
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset['train'])} samples")
    
    print(f"✓ Processed {len(processed)} samples")
    return processed

def save_dataset(processed_data, output_path="data/processed/train.json"):
    """Save processed dataset to JSON"""
    print(f"\n[3/4] Saving dataset to {output_path}...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Saved {len(processed_data)} samples ({file_size_mb:.2f} MB)")

def analyze_dataset(processed_data):
    """Analyze dataset statistics"""
    print("\n[4/4] Dataset Analysis:")
    print(f"  Total samples: {len(processed_data)}")
    
    # Sample analysis
    sample = processed_data[0]
    prompt_tokens = len(sample["prompt"].split())
    response_tokens = len(sample["response"].split())
    
    print(f"  Sample prompt tokens: ~{prompt_tokens}")
    print(f"  Sample response tokens: ~{response_tokens}")
    print(f"  Total sequence length: ~{prompt_tokens + response_tokens}")
    
    # Category distribution
    categories = {}
    for item in processed_data:
        cat = item.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n  Category Distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(processed_data)) * 100
        print(f"    {cat}: {count} ({pct:.1f}%)")

def main():
    print("=" * 80)
    print("LLM Dataset Builder")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Load dataset
    dataset = load_dolly_dataset()
    
    # Process dataset
    processed = process_dataset(dataset)
    
    # Save dataset
    save_dataset(processed)
    
    # Analyze
    analyze_dataset(processed)
    
    print("\n" + "=" * 80)
    print("✅ Dataset preparation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
