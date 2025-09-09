# Gemma-3 Deep Architecture Study

Deep exploration of Gemma-3 model architecture with focus on understanding internal mechanics, LoRA techniques, and optimization strategies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download Gemma-3 4B model (requires Hugging Face account)
python scripts/download_model.py --model google/gemma-3-4b

# Run basic inference test
python scripts/test_inference.py
```

## Project Structure

```
gemma/
├── CLAUDE.md           # Detailed project documentation & research
├── experiments/        # Core experiments
│   ├── lora/          # LoRA implementations
│   │   ├── rank1_dynamic_merger.py  # Novel rank-1 merging
│   │   └── relora_trainer.py        # ReLoRA implementation
│   ├── capacity/      # Capacity analysis
│   └── attention/     # Attention mechanism studies
├── analyze/           # Analysis tools
│   └── capacity_meter.py  # Memorization vs generalization
├── scripts/           # Utility scripts
├── models/           # Model checkpoints
├── data/             # Datasets
└── visualizations/   # Generated plots
```

## Key Features

### 1. Dynamic Rank-1 LoRA Merging
Novel technique where rank-1 LoRA layers are initialized and merged at each training step, leveraging the inherent rank-1 nature of gradient descent updates.

### 2. ReLoRA Implementation
High-rank training through sequential low-rank updates with jagged learning rate scheduling and periodic resets.

### 3. Capacity Analysis
Based on "How much do language models memorize?" - measures model capacity (~3.6 bits/parameter) and tracks memorization vs generalization.

### 4. Mac Metal Optimization
Optimized for Apple Silicon using PyTorch MPS backend and MLX framework.

## Research Papers

1. **How much do language models memorize?** (arxiv:2505.24832)
2. **Set Block Decoding** (arxiv:2509.04185)  
3. **ReLoRA: High-Rank Training Through Low-Rank Updates** (arxiv:2307.05695)

## Hardware Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- 32GB+ unified memory for 4B model
- 64GB+ for 12B model

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed documentation including:
- Architecture deep dive
- Implementation details
- Experiment designs
- Research questions
- Progress tracking

## License

Research project - see individual model licenses for usage restrictions.