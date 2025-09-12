#!/usr/bin/env python3
"""
Benchmark inference speed: PyTorch MPS vs MLX on M3 Ultra
"""

import torch
import time
import numpy as np
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import mlx.core as mx
import mlx_lm
from typing import Dict, List
import json

class InferenceBenchmark:
    def __init__(self):
        self.model_id = "google/gemma-3-12b-it"
        self.results = []
        self.test_prompts = [
            "Write a short poem about artificial intelligence:",
            "Explain the theory of relativity in simple terms:",
            "What are the key differences between Python and Rust?",
            "Describe the process of photosynthesis:",
            "How does a transformer neural network work?"
        ]
    
    def benchmark_pytorch_mps(self) -> Dict:
        """Benchmark PyTorch with MPS backend"""
        print("\n" + "="*60)
        print("🔥 PyTorch MPS Benchmark")
        print("="*60)
        
        results = {
            "backend": "PyTorch MPS",
            "device": "mps",
            "load_time": 0,
            "inference_times": [],
            "tokens_per_second": [],
            "memory_usage": 0
        }
        
        try:
            # Record initial memory
            initial_memory = psutil.Process().memory_info().rss / (1024**3)
            
            # Load model
            print("📥 Loading model...")
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir="./models",
                local_files_only=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir="./models",
                local_files_only=True,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            results["load_time"] = time.time() - start_time
            print(f"✅ Model loaded in {results['load_time']:.2f}s")
            
            # Record memory after loading
            loaded_memory = psutil.Process().memory_info().rss / (1024**3)
            results["memory_usage"] = loaded_memory - initial_memory
            
            # Warmup
            print("🔥 Warming up...")
            warmup_input = tokenizer("Hello", return_tensors="pt")
            warmup_input = {k: v.to("mps") for k, v in warmup_input.items()}
            with torch.no_grad():
                _ = model.generate(**warmup_input, max_new_tokens=10)
            
            # Benchmark inference
            print("⚡ Running inference benchmarks...")
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"  Test {i}/5: ", end="", flush=True)
                
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to("mps") for k, v in inputs.items()}
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True
                    )
                inference_time = time.time() - start_time
                
                results["inference_times"].append(inference_time)
                tokens_per_sec = 100 / inference_time
                results["tokens_per_second"].append(tokens_per_sec)
                print(f"{tokens_per_sec:.2f} tok/s")
            
            # Calculate averages
            results["avg_inference_time"] = np.mean(results["inference_times"])
            results["avg_tokens_per_second"] = np.mean(results["tokens_per_second"])
            
            print(f"\n📊 Average: {results['avg_tokens_per_second']:.2f} tokens/second")
            print(f"💾 Memory usage: {results['memory_usage']:.2f} GB")
            
            # Cleanup
            del model
            torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results["error"] = str(e)
        
        return results
    
    def benchmark_mlx(self) -> Dict:
        """Benchmark MLX backend"""
        print("\n" + "="*60)
        print("🍎 MLX Benchmark")
        print("="*60)
        
        results = {
            "backend": "MLX",
            "device": "mlx",
            "load_time": 0,
            "inference_times": [],
            "tokens_per_second": [],
            "memory_usage": 0
        }
        
        try:
            # Record initial memory
            initial_memory = psutil.Process().memory_info().rss / (1024**3)
            
            # Load model
            print("📥 Loading MLX model...")
            start_time = time.time()
            
            model, tokenizer = mlx_lm.load(
                "./models_mlx/gemma-3-12b-it",
                tokenizer_config={"trust_remote_code": True}
            )
            
            results["load_time"] = time.time() - start_time
            print(f"✅ Model loaded in {results['load_time']:.2f}s")
            
            # Record memory after loading
            loaded_memory = psutil.Process().memory_info().rss / (1024**3)
            results["memory_usage"] = loaded_memory - initial_memory
            
            # Warmup
            print("🔥 Warming up...")
            _ = mlx_lm.generate(
                model, tokenizer,
                prompt="Hello",
                max_tokens=10
            )
            
            # Benchmark inference
            print("⚡ Running inference benchmarks...")
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"  Test {i}/5: ", end="", flush=True)
                
                start_time = time.time()
                _ = mlx_lm.generate(
                    model, tokenizer,
                    prompt=prompt,
                    max_tokens=100,
                    temp=0.7
                )
                inference_time = time.time() - start_time
                
                results["inference_times"].append(inference_time)
                tokens_per_sec = 100 / inference_time
                results["tokens_per_second"].append(tokens_per_sec)
                print(f"{tokens_per_sec:.2f} tok/s")
            
            # Calculate averages
            results["avg_inference_time"] = np.mean(results["inference_times"])
            results["avg_tokens_per_second"] = np.mean(results["tokens_per_second"])
            
            print(f"\n📊 Average: {results['avg_tokens_per_second']:.2f} tokens/second")
            print(f"💾 Memory usage: {results['memory_usage']:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results["error"] = str(e)
        
        return results
    
    def compare_results(self, pytorch_results: Dict, mlx_results: Dict):
        """Compare and display benchmark results"""
        print("\n" + "="*70)
        print("📊 BENCHMARK COMPARISON - Gemma-3-12B on M3 Ultra")
        print("="*70)
        
        # Create comparison table
        print(f"\n{'Metric':<25} {'PyTorch MPS':<20} {'MLX':<20} {'MLX Speedup':<15}")
        print("-" * 70)
        
        # Load time
        if "error" not in pytorch_results and "error" not in mlx_results:
            speedup = pytorch_results["load_time"] / mlx_results["load_time"]
            print(f"{'Model Load Time':<25} {pytorch_results['load_time']:<20.2f}s {mlx_results['load_time']:<20.2f}s {speedup:<15.2f}x")
            
            # Inference speed
            speedup = mlx_results["avg_tokens_per_second"] / pytorch_results["avg_tokens_per_second"]
            print(f"{'Tokens/Second':<25} {pytorch_results['avg_tokens_per_second']:<20.2f} {mlx_results['avg_tokens_per_second']:<20.2f} {speedup:<15.2f}x")
            
            # Memory usage
            mem_reduction = (1 - mlx_results["memory_usage"] / pytorch_results["memory_usage"]) * 100
            print(f"{'Memory Usage (GB)':<25} {pytorch_results['memory_usage']:<20.2f} {mlx_results['memory_usage']:<20.2f} {mem_reduction:<15.1f}% less")
            
            # Average latency
            print(f"{'Avg Latency (100 tok)':<25} {pytorch_results['avg_inference_time']:<20.2f}s {mlx_results['avg_inference_time']:<20.2f}s")
        
        # Save results to file
        with open("benchmark_results.json", "w") as f:
            json.dump({
                "pytorch_mps": pytorch_results,
                "mlx": mlx_results,
                "hardware": "M3 Ultra 256GB",
                "model": "gemma-3-12b-it"
            }, f, indent=2)
        
        print("\n✅ Results saved to benchmark_results.json")
        
        # Performance insights
        print("\n" + "="*70)
        print("💡 PERFORMANCE INSIGHTS")
        print("="*70)
        print("""
        Based on M3 Ultra architecture:
        • MLX leverages unified memory more efficiently
        • Direct Metal API access reduces overhead
        • Optimized matrix operations for Apple Silicon
        • Better cache utilization with 80 GPU cores
        
        Recommendations:
        • Use MLX for production inference
        • PyTorch MPS for training and experimentation
        • Consider 4-bit quantization for 4x memory savings
        • Batch size 1-4 optimal for this hardware
        """)

def main():
    print("🚀 Gemma-3-12B Inference Benchmark Suite")
    print("💻 Hardware: M3 Ultra with 256GB RAM")
    print("🎯 Goal: Compare PyTorch MPS vs MLX performance\n")
    
    benchmark = InferenceBenchmark()
    
    # Check if models exist
    if not Path("./models").exists():
        print("❌ PyTorch model not found. Run download_model.py first")
        return
    
    # Run PyTorch benchmark
    pytorch_results = benchmark.benchmark_pytorch_mps()
    
    # Check if MLX model exists
    if Path("./models_mlx/gemma-3-12b-it").exists():
        # Run MLX benchmark
        mlx_results = benchmark.benchmark_mlx()
        
        # Compare results
        benchmark.compare_results(pytorch_results, mlx_results)
    else:
        print("\n⚠️ MLX model not found")
        print("💡 Run: python convert_to_mlx.py")
    
    print("\n🎉 Benchmark complete!")

if __name__ == "__main__":
    main()