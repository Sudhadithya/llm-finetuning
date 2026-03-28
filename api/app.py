import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add src to path to import inference engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.config import ExperimentConfig
from src.models.factory import ModelFactory

app = FastAPI(title="LLM Fine-Tuning API", description="Inference endpoints for domain-adapted LLMs.")

# Generate fallback config for standalone API testing
demo_config = ExperimentConfig(
    experiment_name="api_default",
    model="microsoft/phi-2",
    dataset_version="api_mock",
    dataset_path="datasets/customer_support/"
)

print("Provisioning inference factory...")
model, tokenizer = ModelFactory.create(demo_config)
model.eval()

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 150

class GenerationResponse(BaseModel):
    result: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(req: GenerationRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Text generation requests require a valid prompt.")
        
    # Mocked generation for speed in architecture template
    response_text = "Click 'Forgot Password' on the login page and follow instructions to reset your access."
    
    return GenerationResponse(result=response_text)

@app.get("/health")
async def health_check():
    return {"status": "ok", "model": demo_config.model}
