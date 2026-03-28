# LLMOps Decision Engine

Fine-tune, evaluate, and **decide** which LLM configurations are worth deploying. 
This framework tracks cost, performance, and failure cases to generate actionable recommendations for real-world ML systems.

## The Problem

Most LLM fine-tuning workflows answer:
- Can we train a model?

But completely fail to answer:
- Which configuration should we actually deploy?
- Is the execution improvement worth the operational cost?
- Why does the model fail in real-world scenarios?

## The Solution

This system provides a decision-support platform for LLM experimentation that:
- Tracks cost, semantic performance, and failure boundaries inherently.
- Compares experiments symmetrically.
- Generates actionable deployment recommendations.

## What Makes This Different?

Unlike typical LLM notebook tutorials, this system is designed for deployment analysis:
- Tracks **cost, absolute performance, and output failure limits together**.
- Enables **reproducible experiment comparison** through central JSON telemetry.
- Generates **actionable deployment decisions**, answering the defining organizational question: *"Should we ship this model?"*

---

## Quickstart

**Pipeline Flow:** `Train` -> `Evaluate` -> `Compare` -> `Insights` -> `Decision Report`

```bash
# 1. Train
python -m src.cli train --config configs/exp_001.yaml

# 2. Evaluate
python -m src.cli evaluate --run-id exp_001

# 3. Compare multiple runs
python -m src.cli compare --runs exp_001 exp_002

# 4. Generate engineering insights
python -m src.cli insights --run-id exp_002

# 5. Output localized decision report
python -m src.cli report --run-id exp_002
```

---

## System Design

The system leverages strict Pydantic schemas, decoupled data pipelines, and a factory generation pattern to ensure absolute alignment between training loops and evaluation inference architectures.

```
llm-experiments-lab/
│
├── api/                      # FastAPI deployment endpoint
├── configs/                  # YAML experiment configurations
├── datasets/                 
│   └── customer_support/     # Domain-specific datasets 
├── docker/                   # Deployment containers
├── experiments/
│   └── logs/                 # JSON runtime, failure, and cost telemetry
├── src/
│   ├── core/                 # Pydantic configs and JSON logging
│   ├── data/                 # Tokenization and prompt alignment
│   ├── models/               # PEFT/LoRA and AutoModel initialization
│   ├── training/             # HF Trainer execution sequences
│   ├── evaluation/           # ROUGE/BLEU computation and failure analysis
│   ├── insights/             # Log analysis and metric thresholds
│   ├── report.py             # Decision report generation
│   └── cli.py                # Unified routing
└── Makefile                  
```

---

## System Demonstration

To prove the framework's capability as an active decision-support system, an evaluation was conducted comparing a heavy base model (`exp_001`) against an optimized PEFT/LoRA configuration (`exp_002`).

### 1. Metrics and Cost Comparison

**Result Summary:** LoRA achieves comparable semantic performance at an ~85% lower cost.

| Metric | `exp_001` (Base) | `exp_002` (LoRA, Rank 8) | Delta |
|--------|------------------|--------------------------|-------|
| Eval Loss | 1.65 | 1.10 | (-0.55 ↓) |
| ROUGE-1 | 0.40 | 0.75 | (+0.35 ↑) |
| BLEU | 0.28 | 0.52 | (+0.24 ↑) |
| BERTScore F1 | 0.78 | 0.91 | (+16.6% ↑) |
| Time | 60 mins | 8.6 mins | (85% ↓) |
| Cost (T4) | ~ $0.35 | ~ $0.05 | ($0.30 ↓) |

**Interpretation:**
- **BLEU/ROUGE**: Lexical evaluation (measuring exact keyword overlap).
- **BERTScore**: Semantic evaluation (measuring contextual intent capture).
*Insight: Although BLEU improvements are numerically moderate (0.52), the BERTScore F1 shows an exceptional gain to 0.91, indicating the model thoroughly understands the objective meaning despite deviating from exact ground-truth phrasing.*

### 2. Failure Case Diagnostics

Instead of relying solely on ROUGE averages, the framework exposes generative limitations:

**`exp_001` (Base Model) Failures:**
- *Query:* "My subscription is too expensive (>50 tokens)"
- *Issue:* Length constraint caused generation hallucination.

**`exp_002` (LoRA Model) Failures:**
- *Query:* "I want to cancel my enterprise API plan"
- *Issue:* Domain gap: Dataset lacks B2B examples, causing consumer-level answers.

### 3. Automated Decision Report Output

The capstone of the repository natively extracts telemetry into actionable technical directives. Executing `python -m src.cli report --run-id exp_002` yields:

```text
LLM Experiment Report — exp_002
============================================================
Dataset: customer_support_v1
Model: microsoft/phi-2

--- Performance & Cost ---
ROUGE: 0.75 | BLEU: 0.52
Training Time: 520.0 seconds
GPU: T4-Cloud
Estimated Cost: $0.0506

--- Semantic Performance ---
BERTScore F1: 0.91

--- Decision ---
 * DEPLOY: LoRA configuration
 * JUSTIFICATION: Achieves high semantic accuracy (BERTScore F1: 0.91) at minimal operational cost ($0.0506).

--- Limitations ---
 * Weak on long or enterprise-level queries.
 * Dataset lacks B2B coverage.

--- Next Steps ---
 * Expand dataset with enterprise support examples.
============================================================
```
