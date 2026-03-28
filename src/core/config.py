import yaml
from pydantic import BaseModel, Field
from typing import List, Optional

class TrainingConfig(BaseModel):
    epochs: int = Field(default=3, gt=0)
    batch_size: int = Field(default=4, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    max_length: int = Field(default=512, gt=0)

class LoRAConfig(BaseModel):
    applied: bool = Field(default=False)
    r: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0)

class EvaluationConfig(BaseModel):
    metrics: List[str] = Field(default_factory=lambda: ["bleu", "rouge", "bertscore"])
    generate_samples: bool = Field(default=True)

class ExperimentConfig(BaseModel):
    experiment_name: str = Field(..., description="Unique identifier for the run")
    model: str = Field(..., description="HuggingFace model string")
    dataset_version: str = Field(..., description="Domain tracking string")
    dataset_path: str = Field(..., description="Path to data directory")
    
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "ExperimentConfig":
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
