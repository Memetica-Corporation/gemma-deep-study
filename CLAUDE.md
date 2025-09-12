# Gemma Deep Architecture Study on M3 Ultra

## Hardware Specs - Mac Studio M3 Ultra
- **Chip**: Apple M3 Ultra (32 cores: 24 performance + 8 efficiency)
- **Memory**: 256GB Unified Memory
- **GPU**: 80-core Apple GPU with Metal 3 support
- **Neural Engine**: 32-core for ML acceleration
- **Memory Bandwidth**: 800 GB/s
- **Max Model Size**: Can handle models up to ~150B parameters in memory

## Project Goals

This project aims to achieve a deep understanding of the Gemma architecture with focus on:
1. Understanding internal model structure and layer interactions
2. Visualizing how fine-tuning modifies different layers and model areas  
3. Building and blending LoRA adapters with advanced techniques
4. Implementing state-of-the-art optimization strategies
5. Mac Metal architecture optimization leveraging all 80 GPU cores

We're focusing on Gemma-3-12B-IT (released May 2025) as our primary model, with full capacity to run inference and fine-tuning locally.

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

### Core Architecture Features (Gemma-3)
- **Attention Mechanism**: Sliding window attention with grouped-query attention (GQA)
  - Local: sliding window of 4096 tokens
  - Global: interleaved for long-range dependencies
  - KV-cache optimization through GQA
- **Context Length**: 8K tokens (expandable with RoPE scaling)
- **RoPE**: Base frequency 10000, supports interpolation
- **Model Sizes**: 2B, 7B, 12B, 27B parameters
- **Current Model**: Gemma-3-12B-IT (instruction-tuned variant)

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
cd /Users/memetica-studio/dev/gemma-deep
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model (Gemma-3-12B-IT)
python download_model.py

# 4. Test setup
python test_model.py  # Verify MPS and model loading

# 5. Run experiments
python run_experiments.py  # Interactive menu
```

### Quick Commands (Makefile)
```bash
make setup       # Complete environment setup
make install     # Install dependencies
make download    # Download Gemma-3-12B model
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

### Hardware Requirements & Capabilities
- **Minimum**: Mac with Apple Silicon (M1/M2/M3/M4)
- **Our Setup**: M3 Ultra with 256GB RAM
- **Model Capacity**:
  - 2B model: ~8GB RAM (runs on any M-series)
  - 7B model: ~28GB RAM (requires M1 Max or higher)
  - 12B model: ~48GB RAM (requires M2 Max or higher)
  - 27B model: ~108GB RAM (requires M2/M3 Ultra)
  - Our system can handle multiple 12B models or one 27B model with room to spare

## Commands & Scripts

### Model Loading
```bash
# Download Gemma-3-12B-IT (our primary model)
python download_model.py

# Test inference with MPS acceleration
python test_model.py --backend mps

# Run with MLX for optimized Apple Silicon performance
python test_model.py --backend mlx
```

### Training
```bash
# Standard LoRA fine-tuning (12B model)
python train_lora.py --model gemma-3-12b-it --rank 16 --batch-size 8

# ReLoRA training with M3 Ultra optimization
python train_relora.py --model gemma-3-12b-it --reset-interval 1000 --device mps

# Rank-1 dynamic merging
python train_rank1_merge.py --model gemma-3-12b-it --merge-freq 1

# Multi-GPU training across all 80 cores
python train_distributed.py --model gemma-3-12b-it --gpus 80
```

### Analysis
```bash
# Visualize attention patterns
python analyze/attention_patterns.py --model gemma-3-12b-it --checkpoint latest

# Measure capacity (12B = ~43.2B bits capacity)
python analyze/capacity_analysis.py --model gemma-3-12b-it

# Profile memory usage on M3 Ultra
python analyze/memory_profile.py --backend mps --ram 256

# Benchmark inference speed
python benchmark/inference_speed.py --model gemma-3-12b-it --device mps
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
- [x] Research Gemma architecture
- [x] Document key papers and techniques
- [x] Set up development environment on M3 Ultra
- [x] Install all ML libraries (PyTorch, MLX, transformers)
- [ ] Download and test Gemma-3-12B-IT model

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

### Phase 4: Scaling & Production
- [x] Direct deployment of 12B model (optimal for our use case)
- [ ] Multi-model ensemble with 256GB RAM
- [ ] Distributed inference across 80 GPU cores
- [ ] Production optimization with Metal 3

## Related Projects
- `/dev/fidelic/the-forge`: Contains latest optimization docs
- Official Gemma PyTorch: https://github.com/google/gemma_pytorch
- MLX Examples: https://github.com/ml-explore/mlx-examples

## Notes & Observations
*This section will be updated with experimental findings and insights*

## Current Status
- **Environment**: Python 3.12.10 with venv activated
- **Libraries**: Latest versions of PyTorch, MLX, transformers installed
- **Models**: Gemma-3-12B-IT successfully downloaded and converted
- **Hardware**: M3 Ultra fully configured for ML workloads

## 🚀 Performance Breakthrough (2025-09-12)

### Benchmark Results on M3 Ultra

| Configuration | Memory | Speed | Notes |
|--------------|--------|-------|-------|
| **MLX 4-bit quantized** | 6.8GB | **75 tok/s** | ⭐ Best overall - 2.7x faster than FP16! |
| MLX FP16 | 23.7GB | 28 tok/s | Highest quality |
| PyTorch MPS FP16 | ~24GB | 15 tok/s | Sampling issues, needs config |

### Key Findings

1. **Quantization Paradox**: 4-bit model is FASTER than FP16
   - Cache efficiency: 6.8GB fits entirely in GPU cache
   - 80 GPU cores process smaller chunks more efficiently
   - 800GB/s memory bandwidth eliminates traditional bottlenecks

2. **MLX vs PyTorch**: MLX is 1.88x faster for FP16, 5x faster for quantized
   - Direct Metal API access
   - Better unified memory utilization
   - Optimized for Apple Silicon architecture

3. **Natural Generation Behavior**:
   - Haiku prompt: 18 tokens (correctly concise)
   - Explanations: 500+ tokens (verbose by default)
   - Speed consistent regardless of output length

### Working Model Paths

```bash
# 4-bit quantized (FASTEST - 75 tok/s, 6.8GB RAM)
mlx_lm.generate --model ./models_mlx/gemma-3-12b-it-q4-working --prompt "Your prompt"

# FP16 (Best quality - 28 tok/s, 23.7GB RAM)
mlx_lm.generate --model ./models_mlx/gemma-3-12b-it --prompt "Your prompt"
```

### Quantization Fix

Initial attempts failed because of incorrect conversion flags. Working method:
```python
from mlx_lm import convert
convert(
    hf_path='google/gemma-3-12b-it',
    mlx_path='./models_mlx/gemma-3-12b-it-q4-working',
    quantize=True,  # Critical flag
    q_bits=4,
    q_group_size=64
)
```

---
Last Updated: 2025-09-12