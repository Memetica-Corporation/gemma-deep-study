#!/usr/bin/env python3
"""
Natural generation benchmark - let models generate until they stop naturally
Compare PyTorch MPS vs MLX, both fp16 versions
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import psutil
import json

# Test prompts that should elicit different length responses
TEST_PROMPTS = [
    "Write a haiku about artificial intelligence:",  # Should be short
    "Explain the water cycle:",  # Should be medium
    "Write a detailed step-by-step recipe for chocolate chip cookies:"  # Should be longer
]

def benchmark_pytorch_natural():
    """Benchmark PyTorch MPS with natural generation"""
    print("\n" + "="*60)
    print("🔥 PyTorch MPS FP16 - Natural Generation")
    print("="*60)
    
    results = []
    
    try:
        # Load model
        print("📥 Loading model...")
        start_time = time.time()
        
        model_id = "google/gemma-3-12b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="mps",
            low_cpu_mem_usage=True
        )
        
        load_time = time.time() - start_time
        print(f"✅ Loaded in {load_time:.2f}s")
        
        # Test each prompt
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\nTest {i}/3: {prompt[:50]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to("mps")
            input_len = inputs['input_ids'].shape[1]
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,  # High limit, but model will stop naturally
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            gen_time = time.time() - start_time
            
            output_len = outputs.shape[1] - input_len
            tokens_per_sec = output_len / gen_time
            
            generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            
            results.append({
                "prompt": prompt,
                "input_tokens": input_len,
                "output_tokens": output_len,
                "generation_time": gen_time,
                "tokens_per_sec": tokens_per_sec,
                "generated_text": generated_text,
                "text_length": len(generated_text)
            })
            
            print(f"  📝 Generated {output_len} tokens in {gen_time:.2f}s")
            print(f"  ⚡ {tokens_per_sec:.2f} tokens/sec")
            print(f"  📏 Text length: {len(generated_text)} characters")
            print(f"  Preview: {generated_text[:100]}...")
        
        # Cleanup
        del model
        torch.mps.empty_cache()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return results

def benchmark_mlx_natural():
    """Benchmark MLX with natural generation"""
    print("\n" + "="*60)
    print("🍎 MLX FP16 - Natural Generation")
    print("="*60)
    
    results = []
    model_path = "./models_mlx/gemma-3-12b-it"
    
    try:
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\nTest {i}/3: {prompt[:50]}...")
            
            # Run MLX generation
            cmd = [
                "mlx_lm.generate",
                "--model", model_path,
                "--prompt", prompt,
                "--max-tokens", "500",  # High limit, model will stop naturally
                "--temp", "0.7"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            gen_time = time.time() - start_time
            
            # Parse output from MLX
            output_lines = result.stdout.strip().split('\n')
            tokens_per_sec = 0
            output_tokens = 0
            peak_memory = 0
            
            for line in output_lines:
                if "Generation:" in line:
                    # Parse: "Generation: X tokens, Y.YY tokens-per-sec"
                    parts = line.split()
                    for j, part in enumerate(parts):
                        if part == "tokens,":
                            output_tokens = int(parts[j-1])
                        elif part == "tokens-per-sec":
                            tokens_per_sec = float(parts[j-1])
                elif "Peak memory:" in line:
                    # Parse: "Peak memory: X.XX GB"
                    parts = line.split()
                    peak_memory = float(parts[2])
            
            # Extract generated text
            generated_text = ""
            if "==========" in result.stdout:
                text_parts = result.stdout.split("==========")
                if len(text_parts) >= 3:
                    generated_text = text_parts[1].strip()
            
            results.append({
                "prompt": prompt,
                "output_tokens": output_tokens,
                "generation_time": gen_time,
                "tokens_per_sec": tokens_per_sec,
                "generated_text": generated_text,
                "text_length": len(generated_text),
                "peak_memory_gb": peak_memory
            })
            
            print(f"  📝 Generated {output_tokens} tokens in {gen_time:.2f}s")
            print(f"  ⚡ {tokens_per_sec:.2f} tokens/sec")
            print(f"  💾 Peak memory: {peak_memory:.2f} GB")
            print(f"  📏 Text length: {len(generated_text)} characters")
            print(f"  Preview: {generated_text[:100]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return results

def compare_results(pytorch_results, mlx_results):
    """Compare and display results"""
    print("\n" + "="*80)
    print("📊 NATURAL GENERATION COMPARISON - Gemma-3-12B on M3 Ultra")
    print("="*80)
    
    print(f"\n{'Prompt Type':<30} {'PyTorch Tokens':<15} {'MLX Tokens':<15} {'PyTorch tok/s':<15} {'MLX tok/s':<15}")
    print("-"*90)
    
    prompt_types = ["Haiku (short)", "Water cycle (medium)", "Recipe (long)"]
    
    for i, prompt_type in enumerate(prompt_types):
        if i < len(pytorch_results) and i < len(mlx_results):
            pt_tokens = pytorch_results[i]['output_tokens']
            mlx_tokens = mlx_results[i]['output_tokens']
            pt_speed = pytorch_results[i]['tokens_per_sec']
            mlx_speed = mlx_results[i]['tokens_per_sec']
            
            print(f"{prompt_type:<30} {pt_tokens:<15} {mlx_tokens:<15} {pt_speed:<15.2f} {mlx_speed:<15.2f}")
    
    # Calculate averages
    if pytorch_results and mlx_results:
        avg_pt_speed = sum(r['tokens_per_sec'] for r in pytorch_results) / len(pytorch_results)
        avg_mlx_speed = sum(r['tokens_per_sec'] for r in mlx_results) / len(mlx_results)
        
        avg_pt_tokens = sum(r['output_tokens'] for r in pytorch_results) / len(pytorch_results)
        avg_mlx_tokens = sum(r['output_tokens'] for r in mlx_results) / len(mlx_results)
        
        print("\n" + "-"*90)
        print(f"{'AVERAGES':<30} {avg_pt_tokens:<15.1f} {avg_mlx_tokens:<15.1f} {avg_pt_speed:<15.2f} {avg_mlx_speed:<15.2f}")
        
        print(f"\n🚀 MLX is {avg_mlx_speed/avg_pt_speed:.2f}x faster than PyTorch MPS")
    
    # Show actual generated text comparison for first prompt
    print("\n" + "="*80)
    print("📝 GENERATION QUALITY COMPARISON (Haiku prompt)")
    print("="*80)
    
    if pytorch_results and mlx_results:
        print("\nPyTorch output:")
        print(pytorch_results[0]['generated_text'][:300])
        
        print("\n\nMLX output:")
        print(mlx_results[0]['generated_text'][:300])
    
    # Save results
    all_results = {
        "pytorch_mps": pytorch_results,
        "mlx": mlx_results
    }
    
    with open("natural_generation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Results saved to natural_generation_results.json")

def main():
    print("🚀 Natural Generation Benchmark")
    print("💻 M3 Ultra with 256GB RAM")
    print("🎯 Comparing PyTorch MPS vs MLX with natural stopping\n")
    
    # Run benchmarks
    pytorch_results = benchmark_pytorch_natural()
    mlx_results = benchmark_mlx_natural()
    
    # Compare
    compare_results(pytorch_results, mlx_results)
    
    print("\n🎉 Benchmark complete!")

if __name__ == "__main__":
    main()