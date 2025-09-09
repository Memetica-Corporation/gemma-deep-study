#!/usr/bin/env python3
"""
Download and setup Gemma-3 models for local experimentation
Supports both PyTorch and MLX formats
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig
)
from huggingface_hub import snapshot_download
import json


def check_metal_availability():
    """Check if Metal Performance Shaders are available"""
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("✅ Metal Performance Shaders (MPS) available")
            return True
    print("⚠️  MPS not available, will use CPU")
    return False


def download_gemma_pytorch(model_name: str, output_dir: str, quantize: str = None):
    """Download Gemma model in PyTorch format"""
    print(f"📥 Downloading {model_name} in PyTorch format...")
    
    # Configure quantization if requested
    quantization_config = None
    if quantize == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("📊 Using 4-bit quantization (NF4)")
    elif quantize == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("📊 Using 8-bit quantization")
    
    try:
        # Download tokenizer
        print("📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(output_dir)
        
        # Download model
        print("🧠 Downloading model weights...")
        device_map = "auto" if check_metal_availability() else "cpu"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            dtype=torch.bfloat16 if not quantize else "auto",
            low_cpu_mem_usage=True
        )
        
        # Save model
        model.save_pretrained(output_dir)
        
        # Save config
        config_info = {
            "model_name": model_name,
            "format": "pytorch",
            "quantization": quantize,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "param_count": sum(p.numel() for p in model.parameters())
        }
        
        with open(os.path.join(output_dir, "download_config.json"), "w") as f:
            json.dump(config_info, f, indent=2)
            
        print(f"✅ Model downloaded successfully to {output_dir}")
        print(f"📊 Parameters: {config_info['param_count']:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def download_gemma_mlx(model_name: str, output_dir: str):
    """Download Gemma model for MLX framework"""
    print(f"📥 Downloading {model_name} for MLX...")
    
    try:
        # MLX models are typically converted versions
        mlx_model_name = model_name.replace("google/", "mlx-community/")
        
        # Download the MLX version if available
        snapshot_download(
            repo_id=mlx_model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"✅ MLX model downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"⚠️  MLX version not found, converting from PyTorch...")
        # Fall back to PyTorch and provide conversion instructions
        if download_gemma_pytorch(model_name, output_dir + "_pytorch"):
            print("\n📝 To convert to MLX format:")
            print("pip install mlx")
            print(f"python -m mlx_lm.convert --hf-path {output_dir}_pytorch --mlx-path {output_dir}")
            return True
        return False


def setup_gemma_models():
    """Setup all Gemma model variants"""
    models = {
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "gemma-3-4b-pt": "google/gemma-3-4b-pt",
        "gemma-3-12b-it": "google/gemma-3-12b-it",
        "gemma-3-12b-pt": "google/gemma-3-12b-pt",
        "gemma-3-27b-it": "google/gemma-3-27b-it",
        "gemma-3-27b-pt": "google/gemma-3-27b-pt",
        "gemma-3-1b-it": "google/gemma-3-1b-it",
        "gemma-3-1b-pt": "google/gemma-3-1b-pt"
    }
    
    return models


def verify_installation(model_dir: str):
    """Verify model installation and test loading"""
    print(f"\n🔍 Verifying installation in {model_dir}...")
    
    try:
        # Check for model files
        model_files = list(Path(model_dir).glob("*.safetensors")) + \
                     list(Path(model_dir).glob("*.bin"))
        
        if not model_files:
            print("⚠️  No model weights found")
            return False
            
        # Try loading tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("✅ Tokenizer loaded successfully")
        
        # Check config
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                print(f"✅ Model config found: {config.get('model_type', 'unknown')}")
                
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Gemma-3 models")
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-4b-it",
        choices=["gemma-3-4b-it", "gemma-3-4b-pt", "gemma-3-12b-it", "gemma-3-12b-pt", "gemma-3-27b-it", "gemma-3-27b-pt", "gemma-3-1b-it", "gemma-3-1b-pt"],
        help="Model variant to download (it=instruction-tuned, pt=pretrained)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "mlx", "both"],
        help="Model format"
    )
    parser.add_argument(
        "--quantize",
        type=str,
        choices=["4bit", "8bit"],
        help="Quantization (PyTorch only)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify installation after download"
    )
    
    args = parser.parse_args()
    
    # Get model mapping
    models = setup_gemma_models()
    model_name = models[args.model]
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"""
╔══════════════════════════════════════╗
║     Gemma-3 Model Downloader         ║
╚══════════════════════════════════════╝
    
Model: {args.model}
Format: {args.format}
Quantization: {args.quantize or 'None'}
Output: {output_dir}
    """)
    
    # Check system
    check_metal_availability()
    
    # Download based on format
    success = False
    if args.format in ["pytorch", "both"]:
        pytorch_dir = output_dir if args.format == "pytorch" else output_dir + "_pytorch"
        success = download_gemma_pytorch(model_name, pytorch_dir, args.quantize)
        
    if args.format in ["mlx", "both"]:
        mlx_dir = output_dir if args.format == "mlx" else output_dir + "_mlx"
        success = download_gemma_mlx(model_name, mlx_dir) or success
        
    # Verify if requested
    if success and args.verify:
        verify_dir = output_dir if args.format != "both" else output_dir + "_pytorch"
        verify_installation(verify_dir)
        
    if success:
        print("\n✨ Setup complete! Next steps:")
        print("1. Run: python scripts/test_inference.py")
        print("2. Start experiments: python experiments/lora/rank1_dynamic_merger.py")
    else:
        print("\n❌ Download failed. Please check your connection and credentials.")
        sys.exit(1)


if __name__ == "__main__":
    main()