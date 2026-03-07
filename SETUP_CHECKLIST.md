## ✅ LLM Fine-Tuning Project Setup Checklist

### Phase 1: Environment Setup
- [ ] Virtual environment created: `python -m venv llm-env`
- [ ] Virtual environment activated: `llm-env\Scripts\activate` (Windows) or `source llm-env/bin/activate` (Mac/Linux)
- [ ] PyTorch installed with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] CUDA verified: `python -c "import torch; print(torch.cuda.is_available())"`  → Should show: **True**
- [ ] GPU detected: `nvidia-smi` → Should show: **NVIDIA GeForce RTX 4060**

### Phase 2: Project Setup
- [ ] Clone/create project folder: `llm-finetuning`
- [ ] Create folder structure:
  ```
  data/
    ├── raw/
    └── processed/
  training/
  evaluation/
  inference/
  models/
  configs/
  docker/
  notebooks/
  outputs/
    ├── checkpoints/
    ├── logs/
    └── results/
  ```
- [ ] Copy these files to project:
  - [ ] `requirements.txt` (from outputs folder)
  - [ ] `training_config.yaml` (to `configs/` folder)
  - [ ] `dataset_builder.py` (to `training/` folder)
  - [ ] `train_lora_optimized.py` (to `training/`, rename to `train_lora.py`)
  - [ ] `.gitignore` (to project root)
  - [ ] `README.md` (to project root)

### Phase 3: Dependencies Installation
- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Verify transformers installed: `python -c "import transformers; print(transformers.__version__)"`
- [ ] Verify PEFT installed: `python -c "import peft; print(peft.__version__)"`
- [ ] Verify accelerate installed: `python -c "import accelerate; print(accelerate.__version__)"`

### Phase 4: Configuration
- [ ] Run: `accelerate config`
  - Select: No for distributed training
  - Select: fp16 for mixed precision
  - Select: Yes for GPU
- [ ] Check `configs/training_config.yaml` matches your RTX 4060 specs

### Phase 5: Dataset Preparation
- [ ] Navigate to project folder: `cd llm-finetuning`
- [ ] Run: `python training/dataset_builder.py`
- [ ] Verify output: `data/processed/train.json` exists and contains ~15,000 samples
- [ ] Check file size: Should be ~50-100MB

### Phase 6: Pre-Training Checks
- [ ] Verify disk space: At least 50GB free
- [ ] Check GPU memory: `nvidia-smi` → Should show ~8GB available
- [ ] Create W&B account (optional): https://wandb.ai
- [ ] Login to W&B (optional): `wandb login`

### Phase 7: Start Training
- [ ] Run: `python training/train_lora.py`
- [ ] Monitor output:
  ```
  ✓ CUDA Available: True
  ✓ GPU Device: NVIDIA GeForce RTX 4060
  ✓ [1/6] Loading Dataset...
  ✓ [2/6] Loading Tokenizer...
  ✓ [3/6] Loading Model with 8-bit Quantization...
  ✓ [4/6] Applying LoRA Adapters...
  ✓ [5/6] Setting Training Arguments...
  ✓ [6/6] Training...
  ```

### Phase 8: During Training (2-3 hours)
- [ ] Monitor GPU: `nvidia-smi` every 15 mins
- [ ] GPU Memory should stay around 7.8GB
- [ ] No OOM errors
- [ ] Training loss decreasing over time
- [ ] Check W&B dashboard (if using)

### Phase 9: Post-Training
- [ ] Training completes without errors
- [ ] Check `models/phi2-lora/` directory created
- [ ] Verify files:
  - [ ] `adapter_model.bin` (~150MB)
  - [ ] `adapter_config.json`
  - [ ] `tokenizer.json`
- [ ] Check `outputs/checkpoints/` folder created

### Phase 10: Evaluation (Optional)
- [ ] Run: `python evaluation/evaluate.py` (once created)
- [ ] Compare base vs fine-tuned model
- [ ] Review metrics: BLEU, ROUGE, BERTScore

### Phase 11: Inference (Optional)
- [ ] Run: `python inference/server.py` (once created)
- [ ] Test endpoint: `curl http://localhost:8000/generate`
- [ ] Check response time

### Phase 12: Deployment (Optional)
- [ ] Create Docker image: `docker build -f docker/Dockerfile -t llm-finetuning:latest .`
- [ ] Run container: `docker run -p 8000:8000 --gpus all llm-finetuning:latest`

---

## 🚨 Common Issues & Solutions

### Issue: CUDA not available
**Solution**: 
- Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall`
- Verify NVIDIA drivers: `nvidia-smi` should show driver version

### Issue: Out of Memory (OOM) during training
**Solution**:
- Reduce batch size in config (try 1)
- Increase gradient accumulation steps
- Reduce max_seq_length to 256
- Check if other GPU processes running: `nvidia-smi`

### Issue: Model download fails
**Solution**:
- Check internet connection
- Try: `huggingface-cli login` and authenticate
- Set cache dir: `export HF_HOME=/path/to/large/disk`

### Issue: Training very slow
**Solution**:
- Verify GPU usage: `nvidia-smi` (should show >90% GPU-Util)
- Check if CPU bottlenecked: reduce number of data workers
- Consider reducing num_epochs for testing

### Issue: W&B login fails
**Solution**:
- Create account at https://wandb.ai
- Get API key from settings
- Run: `wandb login` and paste key
- Or disable: `export WANDB_DISABLED=true`

---

## 📊 Expected Results

### Training Progress
```
Epoch 1/2: 100%|████████| 3750/3750 [1:15:30<00:00, 1.20s/it]
  loss: 1.8234
  eval_loss: 1.5623

Epoch 2/2: 100%|████████| 3750/3750 [1:15:45<00:00, 1.21s/it]
  loss: 1.2456
  eval_loss: 1.1234
```

### GPU Memory Usage
```
NVIDIA GeForce RTX 4060  |  7.8GB / 8.0GB  |  97%
```

### Model Sizes
```
Base Phi-2:        2.7B parameters
Fine-tuned LoRA:   ~295K parameters (0.01%)
Memory saved:      ~75%
```

---

## 📞 Support

If you encounter issues:
1. Check the "Common Issues" section above
2. Review full error message
3. Check W&B dashboard for training logs
4. Verify all prerequisites are met
5. Try starting fresh with virtual environment

---

**Last Updated**: March 2026  
**Status**: Ready to Use
