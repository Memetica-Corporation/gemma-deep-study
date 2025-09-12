#!/usr/bin/env python3
"""
Download Gemma-3-12B-IT model from HuggingFace with optimizations for M3 Ultra
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_gemma_12b():
    """Download Gemma-3-12B-IT model from HuggingFace"""
    
    model_id = "google/gemma-3-12b-it"  # Gemma-3-12B-IT (released May 2025)
    cache_dir = Path("./models")
    cache_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Starting download of {model_id}")
    print(f"📁 Cache directory: {cache_dir}")
    print(f"💻 Device: Apple M3 Ultra with 256GB RAM")
    print(f"🔧 Using MPS (Metal Performance Shaders) backend")
    
    try:
        # Download model files
        print("\n📥 Downloading model files...")
        model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        print(f"✅ Model downloaded to: {model_path}")
        
        # Download tokenizer
        print("\n📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir
        )
        print("✅ Tokenizer downloaded successfully")
        
        # Verify MPS availability
        if torch.backends.mps.is_available():
            print("\n✅ Metal Performance Shaders (MPS) is available")
            print(f"🎯 MPS device ready for inference")
        else:
            print("\n⚠️ MPS not available, will use CPU")
        
        print("\n🎉 Download complete!")
        print(f"📊 Model size: ~12B parameters")
        print("💡 Next steps: Run test_model.py to verify the setup")
        
        return model_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\n💡 Check your internet connection and try again")
        sys.exit(1)

if __name__ == "__main__":
    download_gemma_12b()