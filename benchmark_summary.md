# 🚀 Gemma-3-12B Benchmark Results on M3 Ultra

## Hardware
- **Chip**: Apple M3 Ultra (32 cores: 24 performance + 8 efficiency)  
- **GPU**: 80-core Apple GPU with Metal 3
- **Memory**: 256GB Unified Memory
- **Memory Bandwidth**: 800 GB/s

## Model Configurations Tested

### 1. PyTorch MPS FP16
- **Status**: ✅ Loads successfully
- **Memory**: ~24GB when loaded
- **Speed**: ~15 tok/s average (when working)
- **Issues**: Sampling errors with temperature > 0

### 2. PyTorch MPS 8-bit Quantized  
- **Status**: ❌ Requires bitsandbytes (not available for MPS)
- **Alternative**: Could use torch.quantization but limited MPS support

### 3. MLX FP16
- **Status**: ✅ Works perfectly
- **Memory**: 23.7GB consistent
- **Speed**: 27-29 tokens/second
- **File size**: 22GB on disk

### 4. MLX 4-bit Quantized
- **Status**: ⚠️ Conversion works but loading has issues
- **Memory**: Expected ~8-10GB
- **Speed**: Expected similar to FP16
- **File size**: 6.2GB on disk (72% smaller!)

## Performance Results

### Token Generation Speed (tokens/second)
| Model | Short (Haiku) | Medium (Explanation) | Long (Recipe) | Average |
|-------|---------------|---------------------|---------------|---------|
| PyTorch MPS | 10.53 | 16.96 | 17.21 | 14.90 |
| MLX FP16 | 29.00 | 27.62 | 27.66 | 28.09 |
| **Speedup** | **2.75x** | **1.63x** | **1.61x** | **1.88x** |

### Natural Generation Behavior
- **Haiku prompt**: Generated 18 tokens naturally (correctly short)
- **Explanation prompts**: Wanted to generate 500+ tokens (hit limit)
- **Memory stable**: 23.7GB throughout all generation lengths

## Key Findings

1. **MLX is 1.88x faster** than PyTorch MPS on average
2. **Consistent performance**: MLX maintains ~28 tok/s regardless of output length
3. **Memory efficiency**: Both use similar memory (~24GB) for FP16
4. **Quantization benefits**: 4-bit reduces model size by 72% but needs fixes
5. **M3 Ultra advantage**: 800GB/s memory bandwidth eliminates memory bottlenecks

## Recommendations

### For Production Inference
✅ **Use MLX FP16** - Fast, stable, excellent performance

### For Memory-Constrained Scenarios  
🔧 Fix and use MLX 4-bit quantized (6.2GB vs 22GB)

### For Training/Research
✅ Use PyTorch with proper generation config

### For Maximum Speed
- MLX with batch processing
- Consider Flash Attention when available
- Use Metal Performance Shaders directly for custom ops

## Next Steps

1. Fix 4-bit quantized model loading issue
2. Implement proper PyTorch sampling config
3. Test larger batch sizes
4. Benchmark training/fine-tuning performance
5. Implement ReLoRA and dynamic rank-1 merging

---
*Benchmark conducted on 2025-09-12*
