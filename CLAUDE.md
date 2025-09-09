# Gemma-3 Deep Architecture Study

## Project Goals

This project aims to achieve a deep understanding of the Gemma-3 architecture with focus on:
1. Understanding internal model structure and layer interactions
2. Visualizing how fine-tuning modifies different layers and model areas  
3. Building and blending LoRA adapters with advanced techniques
4. Implementing state-of-the-art optimization strategies
5. Mac Metal architecture optimization for local development

We'll start with Gemma-3 4B for manageable experiments, then expand to 12B.

## Key Research Papers to Implement

### 1. How much do language models memorize? (arxiv:2505.24832)
**Key Findings:**
- GPT-style models have capacity of ~3.6 bits per parameter
- Models memorize until capacity fills, then "grokking" begins and generalization takes over
- Double descent occurs when dataset size exceeds model capacity
- Above certain dataset size, membership inference fails

**Implementation Ideas:**
- Measure Gemma-3's effective capacity through bitstring experiments
- Track memorization vs generalization during fine-tuning
- Visualize capacity saturation points across layers

### 2. Set Block Decoding (arxiv:2509.04185)  
**Key Innovation:**
- Accelerates inference by integrating standard next token prediction and masked token prediction
- Samples multiple (not necessarily consecutive) future tokens in parallel
- Uses advanced solvers from discrete diffusion literature

**Implementation Ideas:**
- Implement SBD for Gemma-3 inference
- Benchmark speedup vs standard autoregressive decoding
- Visualize parallel token generation patterns

### 3. ReLoRA: High-Rank Training Through Low-Rank Updates (arxiv:2307.05695)
**Core Technique:**
- Sequential low-rank updates that aggregate to high-rank
- Parameter merging and resets at intervals
- Jagged learning rate schedule with 2x LR during ReLoRA stage
- 5.5GB RAM savings, 9-40% speed improvement

**Implementation Ideas:**
- Implement ReLoRA training loop for Gemma-3
- Visualize rank evolution during training
- Compare with standard LoRA and full fine-tuning

## Advanced LoRA Techniques

### Dynamic Rank-1 Merging (from Twitter/X discussions)
**Concept:** 
- Initialize new rank-1 LoRA at each training step
- Merge immediately into base model
- Gradient descent updates are inherently rank-1 for batch size 1
- Frequent subspace changes mimic full-rank updates

**Implementation Plan:**
- Build rank-1 LoRA merge trainer
- Track effective rank evolution via SVD analysis
- Visualize subspace trajectory during training
- Compare convergence with standard LoRA approaches

## Gemma-3 Architecture Deep Dive

### Core Architecture Features
- **Attention Mechanism**: 5:1 ratio of local to global attention layers
  - Local: sliding window of 1024 tokens
  - Global: handles long-range dependencies
  - KV-cache memory reduced from 60% to <15%
- **Context Length**: 128K tokens (16x increase from Gemma-2)
- **RoPE**: Base frequency 1M for global, 10K for local layers
- **Model Sizes**: 270M, 1B, 4B, 12B, 27B parameters

### Implementation Stack
- **Framework**: PyTorch with Metal Performance Shaders (MPS) backend
- **Mac Optimization**: 
  - MLX framework for Apple Silicon optimization
  - Unified memory architecture (direct GPU memory access)
  - ~75% of system RAM usable for GPU
- **Quantization**: Support for 4-bit, 8-bit quantization
- **Mixed Precision**: bf16/fp16 training

## Experiments & Visualizations

### 1. Layer Analysis Tools
```python
# Planned modules:
- layer_analyzer.py: Track activation patterns, gradient flow
- attention_visualizer.py: Visualize local vs global attention
- capacity_meter.py: Measure memorization vs generalization
```

### 2. LoRA Experiments
```python
# Planned experiments:
- rank1_merger.py: Dynamic rank-1 LoRA merging
- relora_trainer.py: ReLoRA implementation
- lora_blender.py: Multi-LoRA composition techniques
```

### 3. Optimization Benchmarks
```python
# Benchmarking suite:
- inference_accelerator.py: SBD implementation
- metal_optimizer.py: Mac-specific optimizations
- memory_profiler.py: Track RAM/VRAM usage
```

## Development Environment

### Setup Instructions
```bash
# 1. Clone and setup environment
cd /Users/andrewyakovlev/Dev/fidelic/gemma
./setup.sh  # Creates venv and installs dependencies

# 2. Activate environment
source venv/bin/activate
# Or use: source activate.sh for quick activation

# 3. Download model
python scripts/download_model.py --model gemma-3-4b

# 4. Run experiments
python run_experiments.py  # Interactive menu
```

### Quick Commands (Makefile)
```bash
make setup       # Complete environment setup
make install     # Install dependencies
make download    # Download Gemma-3 4B model
make test        # Run inference tests
make benchmark   # Run Metal/MLX benchmarks
make experiments # Launch experiment menu
make clean       # Clean cache files
make check       # Check environment
```

### Dependencies
```bash
# Core
torch>=2.0  # MPS backend support
transformers  # Gemma model support
mlx  # Apple Silicon optimization

# Visualization
matplotlib
plotly
tensorboard

# Analysis
numpy
scipy  # SVD analysis
pandas
```

### Hardware Requirements
- Mac with Apple Silicon (M1/M2/M3/M4)
- Minimum 32GB unified memory for 4B model
- 64GB+ recommended for 12B model

## Commands & Scripts

### Model Loading
```bash
# Download Gemma-3 4B
python scripts/download_model.py --model gemma-3-4b

# Test inference
python scripts/test_inference.py --model gemma-3-4b --backend mps
```

### Training
```bash
# Standard LoRA fine-tuning
python train_lora.py --model gemma-3-4b --rank 16

# ReLoRA training
python train_relora.py --model gemma-3-4b --reset-interval 1000

# Rank-1 dynamic merging
python train_rank1_merge.py --model gemma-3-4b --merge-freq 1
```

### Analysis
```bash
# Visualize attention patterns
python analyze/attention_patterns.py --checkpoint latest

# Measure capacity
python analyze/capacity_analysis.py --model gemma-3-4b

# Profile memory usage
python analyze/memory_profile.py --backend mps
```

## Research Questions to Explore

1. **Capacity & Memorization**
   - What is Gemma-3's effective capacity in bits/parameter?
   - How does capacity vary across layers?
   - When does grokking occur during fine-tuning?

2. **LoRA Dynamics**
   - How does rank-1 merging compare to higher-rank LoRA?
   - Can we visualize the subspace trajectory?
   - What's the optimal merge frequency?

3. **Attention Mechanism**
   - How do local vs global layers specialize during fine-tuning?
   - Can we reduce the 5:1 ratio without performance loss?
   - How does attention pattern change with context length?

4. **Mac Metal Optimization**
   - What's the optimal batch size for unified memory?
   - How does MLX compare to PyTorch MPS for Gemma?
   - Can we leverage Neural Engine for specific operations?

## Progress Tracking

### Phase 1: Foundation (Current)
- [x] Research Gemma-3 architecture
- [x] Document key papers and techniques
- [ ] Set up development environment
- [ ] Download and test Gemma-3 4B model

### Phase 2: Core Implementation
- [ ] Basic LoRA trainer
- [ ] ReLoRA implementation
- [ ] Rank-1 dynamic merging
- [ ] Attention visualizer

### Phase 3: Advanced Experiments
- [ ] Capacity analysis
- [ ] SBD inference accelerator
- [ ] Multi-LoRA blending
- [ ] Metal optimization benchmarks

### Phase 4: Scaling
- [ ] Migrate to Gemma-3 12B
- [ ] Distributed training setup
- [ ] Production optimization

## Related Projects
- `/dev/fidelic/the-forge`: Contains latest optimization docs
- Official Gemma PyTorch: https://github.com/google/gemma_pytorch
- MLX Examples: https://github.com/ml-explore/mlx-examples

## Notes & Observations
*This section will be updated with experimental findings and insights*

---
Last Updated: 2025-09-09