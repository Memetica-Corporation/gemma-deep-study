#!/usr/bin/env python3
"""
Precise comparison of MLX fp16 vs 4-bit quantized models
Same prompts, same token counts, detailed metrics
"""

import os
import time
import subprocess
import json
import psutil

def run_mlx_test(model_path, prompt, max_tokens=100):
    """Run MLX inference and capture detailed metrics"""
    cmd = [
        "mlx_lm.generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0"  # Deterministic output
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    # Parse output
    output_lines = result.stdout.strip().split('\n')
    
    # Extract metrics from output
    metrics = {}
    for line in output_lines:
        if "Prompt:" in line:
            parts = line.split(',')
            for part in parts:
                if "tokens-per-sec" in part:
                    metrics['prompt_speed'] = float(part.split()[0])
                elif "tokens" in part:
                    metrics['prompt_tokens'] = int(part.split()[0])
        elif "Generation:" in line:
            parts = line.split(',')
            for part in parts:
                if "tokens-per-sec" in part:
                    metrics['generation_speed'] = float(part.split()[0])
                elif "tokens" in part and "tokens-per-sec" not in part:
                    metrics['generation_tokens'] = int(part.split()[0])
        elif "Peak memory:" in line:
            metrics['peak_memory_gb'] = float(line.split()[-2])
    
    metrics['total_time'] = end_time - start_time
    
    # Get the generated text
    start_marker = "=========="
    if start_marker in result.stdout:
        text_parts = result.stdout.split(start_marker)
        if len(text_parts) >= 3:
            metrics['generated_text'] = text_parts[1].strip()
    
    return metrics

def compare_models():
    """Compare fp16 vs 4-bit quantized models"""
    
    print("🔬 Gemma-3-12B Model Comparison: FP16 vs 4-bit Quantized")
    print("="*70)
    
    # Test prompts - variety of types
    test_cases = [
        {
            "prompt": "Explain quantum computing in simple terms:",
            "tokens": 100
        },
        {
            "prompt": "Write a Python function to calculate fibonacci numbers:",
            "tokens": 150
        },
        {
            "prompt": "What are the key differences between supervised and unsupervised learning?",
            "tokens": 100
        }
    ]
    
    models = {
        "fp16": "./models_mlx/gemma-3-12b-it",
        "q4": "./models_mlx/gemma-3-12b-it-q4"
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n📊 Testing {model_name.upper()} Model")
        print("-"*40)
        
        results[model_name] = []
        
        for i, test in enumerate(test_cases, 1):
            print(f"\nTest {i}/3: {test['prompt'][:50]}...")
            
            metrics = run_mlx_test(model_path, test['prompt'], test['tokens'])
            metrics['prompt'] = test['prompt']
            results[model_name].append(metrics)
            
            print(f"  ⚡ Generation: {metrics.get('generation_speed', 0):.2f} tok/s")
            print(f"  💾 Memory: {metrics.get('peak_memory_gb', 0):.2f} GB")
            print(f"  ⏱️ Total time: {metrics.get('total_time', 0):.2f}s")
    
    # Detailed comparison
    print("\n" + "="*70)
    print("📊 DETAILED COMPARISON")
    print("="*70)
    
    # Average speeds
    fp16_speeds = [r.get('generation_speed', 0) for r in results['fp16']]
    q4_speeds = [r.get('generation_speed', 0) for r in results['q4']]
    
    fp16_memory = [r.get('peak_memory_gb', 0) for r in results['fp16']]
    q4_memory = [r.get('peak_memory_gb', 0) for r in results['q4']]
    
    print(f"\n🚀 Generation Speed (tokens/second):")
    print(f"  FP16 Average: {sum(fp16_speeds)/len(fp16_speeds):.2f}")
    print(f"  Q4 Average:   {sum(q4_speeds)/len(q4_speeds):.2f}")
    print(f"  Difference:   {abs(sum(fp16_speeds)/len(fp16_speeds) - sum(q4_speeds)/len(q4_speeds)):.2f}")
    
    print(f"\n💾 Peak Memory Usage (GB):")
    print(f"  FP16 Average: {sum(fp16_memory)/len(fp16_memory):.2f}")
    print(f"  Q4 Average:   {sum(q4_memory)/len(q4_memory):.2f}")
    print(f"  Savings:      {sum(fp16_memory)/len(fp16_memory) - sum(q4_memory)/len(q4_memory):.2f} GB")
    
    # Output quality comparison
    print(f"\n📝 Output Quality Check (First prompt):")
    if results['fp16'] and results['q4']:
        fp16_text = results['fp16'][0].get('generated_text', '')[:200]
        q4_text = results['q4'][0].get('generated_text', '')[:200]
        
        print(f"\nFP16 Output:\n{fp16_text}...")
        print(f"\nQ4 Output:\n{q4_text}...")
        
        # Check if outputs are similar
        if fp16_text[:100] == q4_text[:100]:
            print("\n✅ Outputs are identical (first 100 chars)")
        else:
            print("\n⚠️ Outputs differ slightly")
    
    # Save detailed results
    with open("model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n💾 Detailed results saved to model_comparison.json")
    
    return results

def investigate_quantization():
    """Investigate why quantization isn't reducing memory"""
    print("\n" + "="*70)
    print("🔍 QUANTIZATION INVESTIGATION")
    print("="*70)
    
    # Check model file sizes
    print("\n📁 Model File Sizes:")
    
    models_to_check = [
        ("./models_mlx/gemma-3-12b-it", "FP16"),
        ("./models_mlx/gemma-3-12b-it-q4", "4-bit")
    ]
    
    for model_path, name in models_to_check:
        if os.path.exists(model_path):
            total_size = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    if file.endswith(('.safetensors', '.npz', '.json')):
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        total_size += size
            
            print(f"  {name}: {total_size / (1024**3):.2f} GB on disk")
    
    print("\n💡 Possible reasons for similar memory usage:")
    print("  1. MLX may be dequantizing weights during inference")
    print("  2. KV-cache and activations remain in fp16")
    print("  3. Model architecture overhead dominates at 12B scale")
    print("  4. Memory measurement includes system overhead")
    
    print("\n🔧 Recommendations:")
    print("  • Use 4-bit for slightly better speed (cache efficiency)")
    print("  • Use fp16 for best quality")
    print("  • Try 2-bit quantization for aggressive memory savings")
    print("  • Consider model pruning for further optimization")

def main():
    print("🚀 Starting Comprehensive Model Comparison")
    print("💻 M3 Ultra with 256GB RAM\n")
    
    # Check system resources
    mem = psutil.virtual_memory()
    print(f"💾 Available RAM: {mem.available / (1024**3):.1f} GB")
    print(f"📊 RAM Usage: {mem.percent}%\n")
    
    # Run comparison
    results = compare_models()
    
    # Investigate quantization
    investigate_quantization()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()