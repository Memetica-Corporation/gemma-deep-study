#!/usr/bin/env python3
"""
Quality comparison between FP16 and 4-bit quantized models
Testing on various prompt types to assess degradation
"""

import subprocess
import json
from typing import Dict, List

TEST_PROMPTS = [
    {
        "type": "factual",
        "prompt": "What is the capital of France and when was the Eiffel Tower built?"
    },
    {
        "type": "reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning."
    },
    {
        "type": "creative",
        "prompt": "Write a haiku about artificial intelligence:"
    },
    {
        "type": "code",
        "prompt": "Write a Python function to find the nth Fibonacci number using recursion:"
    },
    {
        "type": "explanation",
        "prompt": "Explain the difference between machine learning and deep learning in simple terms:"
    }
]

def generate_text(model_path: str, prompt: str, max_tokens: int = 150) -> Dict:
    """Generate text using MLX model"""
    cmd = [
        "mlx_lm.generate",
        "--model", model_path,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", "0.0"  # Deterministic for fair comparison
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract generated text
    generated = ""
    if "==========" in result.stdout:
        parts = result.stdout.split("==========")
        if len(parts) >= 3:
            generated = parts[1].strip()
    
    # Parse metrics
    metrics = {}
    for line in result.stdout.split('\n'):
        if "Generation:" in line:
            if "tokens-per-sec" in line:
                # Extract speed
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "tokens-per-sec":
                        metrics['speed'] = float(parts[i-1])
        elif "Peak memory:" in line:
            parts = line.split()
            metrics['memory'] = float(parts[2])
    
    return {
        "text": generated,
        "metrics": metrics
    }

def compare_quality():
    """Compare quality between FP16 and 4-bit models"""
    
    fp16_model = "./models_mlx/gemma-3-12b-it"
    q4_model = "./models_mlx/gemma-3-12b-it-q4-working"
    
    results = []
    
    print("🔬 Quality Comparison: FP16 vs 4-bit Quantized")
    print("="*60)
    
    for test in TEST_PROMPTS:
        print(f"\n📝 Testing: {test['type'].upper()}")
        print(f"Prompt: {test['prompt'][:60]}...")
        print("-"*60)
        
        # Generate with FP16
        print("Generating with FP16...")
        fp16_result = generate_text(fp16_model, test['prompt'])
        
        # Generate with 4-bit
        print("Generating with 4-bit...")
        q4_result = generate_text(q4_model, test['prompt'])
        
        # Compare
        result = {
            "type": test['type'],
            "prompt": test['prompt'],
            "fp16": {
                "text": fp16_result['text'],
                "speed": fp16_result['metrics'].get('speed', 0),
                "memory": fp16_result['metrics'].get('memory', 0)
            },
            "q4": {
                "text": q4_result['text'],
                "speed": q4_result['metrics'].get('speed', 0),
                "memory": q4_result['metrics'].get('memory', 0)
            }
        }
        
        results.append(result)
        
        # Display comparison
        print(f"\n🔷 FP16 Output ({result['fp16']['speed']:.1f} tok/s):")
        print(result['fp16']['text'][:200])
        
        print(f"\n🔶 4-bit Output ({result['q4']['speed']:.1f} tok/s):")
        print(result['q4']['text'][:200])
        
        # Simple similarity check
        if result['fp16']['text'][:100] == result['q4']['text'][:100]:
            print("\n✅ Outputs are identical (first 100 chars)")
        else:
            print("\n⚠️ Outputs differ")
    
    # Save detailed results
    with open("quality_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze quality differences"""
    print("\n" + "="*60)
    print("📊 QUALITY ANALYSIS SUMMARY")
    print("="*60)
    
    identical_count = 0
    similar_count = 0
    different_count = 0
    
    for result in results:
        fp16_text = result['fp16']['text']
        q4_text = result['q4']['text']
        
        # Check similarity
        if fp16_text == q4_text:
            identical_count += 1
        elif fp16_text[:50] == q4_text[:50]:
            similar_count += 1
        else:
            different_count += 1
    
    total = len(results)
    print(f"\n📈 Similarity Statistics:")
    print(f"  Identical outputs: {identical_count}/{total} ({100*identical_count/total:.1f}%)")
    print(f"  Similar (same start): {similar_count}/{total} ({100*similar_count/total:.1f}%)")
    print(f"  Different: {different_count}/{total} ({100*different_count/total:.1f}%)")
    
    # Performance comparison
    avg_fp16_speed = sum(r['fp16']['speed'] for r in results) / len(results)
    avg_q4_speed = sum(r['q4']['speed'] for r in results) / len(results)
    
    print(f"\n⚡ Performance:")
    print(f"  FP16 average: {avg_fp16_speed:.1f} tok/s")
    print(f"  4-bit average: {avg_q4_speed:.1f} tok/s")
    print(f"  Speedup: {avg_q4_speed/avg_fp16_speed:.2f}x")
    
    print(f"\n💾 Memory:")
    print(f"  FP16: {results[0]['fp16']['memory']:.1f} GB")
    print(f"  4-bit: {results[0]['q4']['memory']:.1f} GB")
    print(f"  Savings: {(1 - results[0]['q4']['memory']/results[0]['fp16']['memory'])*100:.1f}%")

def main():
    print("🚀 Gemma-3-12B Quality Comparison Test")
    print("Comparing FP16 vs 4-bit quantized outputs\n")
    
    results = compare_quality()
    analyze_results(results)
    
    print("\n✅ Results saved to quality_comparison_results.json")
    print("\n🎯 Conclusion: Check the outputs above for quality assessment")

if __name__ == "__main__":
    main()