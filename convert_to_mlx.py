#!/usr/bin/env python3
"""
Convert Gemma-2-27B model to MLX format for optimized Apple Silicon performance
"""

import os
import sys
import time
from pathlib import Path
import mlx_lm
from mlx_lm import convert

def convert_model():
    """Convert HuggingFace model to MLX format"""
    
    model_id = "google/gemma-3-12b-it"
    mlx_model_path = Path("./models_mlx/gemma-3-12b-it")
    
    print("🍎 MLX Model Converter for Gemma-3-12B-IT")
    print("="*60)
    print(f"📥 Source: {model_id}")
    print(f"📤 Target: {mlx_model_path}")
    print(f"💻 Hardware: M3 Ultra with 256GB RAM")
    print("="*60)
    
    # Create MLX models directory
    mlx_model_path.parent.mkdir(exist_ok=True)
    
    try:
        print("\n⚙️ Starting conversion to MLX format...")
        print("📝 This will create an optimized version for Apple Silicon")
        print("⏱️ Expected time: 5-10 minutes\n")
        
        start_time = time.time()
        
        # Convert model
        convert.convert(
            hf_path=model_id,
            mlx_path=str(mlx_model_path),
            quantize=None,  # No quantization for first test
            dtype="float16"  # Use fp16 for balance of speed/quality
        )
        
        conversion_time = time.time() - start_time
        
        print(f"\n✅ Conversion complete in {conversion_time:.2f} seconds")
        print(f"📁 MLX model saved to: {mlx_model_path}")
        
        # Calculate model size
        total_size = sum(
            f.stat().st_size for f in mlx_model_path.rglob("*") if f.is_file()
        )
        print(f"💾 MLX model size: {total_size / (1024**3):.2f} GB")
        
        print("\n🎯 Next steps:")
        print("  1. Run benchmark_inference.py to compare speeds")
        print("  2. Try 4-bit quantization for 4x memory savings")
        print("  3. Test with different prompt lengths")
        
        return mlx_model_path
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("💡 Make sure the HuggingFace model is fully downloaded first")
        sys.exit(1)

def convert_with_quantization():
    """Convert model with 4-bit quantization for memory efficiency"""
    
    model_id = "google/gemma-3-12b-it"
    mlx_model_path_q4 = Path("./models_mlx/gemma-3-12b-it-q4")
    
    print("\n🔧 Converting with 4-bit quantization...")
    print("📊 This reduces model from ~24GB to ~6GB")
    
    try:
        start_time = time.time()
        
        convert.convert(
            hf_path=model_id,
            mlx_path=str(mlx_model_path_q4),
            quantize="q4",  # 4-bit quantization
            dtype="float16"
        )
        
        conversion_time = time.time() - start_time
        
        print(f"✅ 4-bit conversion complete in {conversion_time:.2f} seconds")
        print(f"📁 Quantized model saved to: {mlx_model_path_q4}")
        
        # Calculate size reduction
        total_size = sum(
            f.stat().st_size for f in mlx_model_path_q4.rglob("*") if f.is_file()
        )
        print(f"💾 Quantized model size: {total_size / (1024**3):.2f} GB")
        print(f"🎯 Memory savings: ~75%")
        
    except Exception as e:
        print(f"❌ Quantization failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Gemma to MLX format")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also create 4-bit quantized version"
    )
    
    args = parser.parse_args()
    
    # Convert to standard MLX format
    mlx_path = convert_model()
    
    # Optionally create quantized version
    if args.quantize:
        convert_with_quantization()
    
    print("\n🚀 Ready for benchmarking!")