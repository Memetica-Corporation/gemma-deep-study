#!/usr/bin/env python3
"""
Test inference with Gemma-3 models
Includes benchmarking for MPS vs CPU performance
"""

import torch
import time
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from rich.console import Console
from rich.table import Table
from typing import Dict


console = Console()


def load_model(model_path: str, device: str = "auto"):
    """Load Gemma model and tokenizer"""
    console.print(f"[cyan]Loading model from {model_path}...[/cyan]")
    
    # Determine device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    console.print(f"[green]Using device: {device}[/green]")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device if device != "mps" else None,
        low_cpu_mem_usage=True
    )
    
    if device == "mps":
        model = model.to("mps")
    
    return model, tokenizer, device


def benchmark_inference(
    model, 
    tokenizer, 
    device: str,
    prompt: str,
    max_length: int = 100,
    num_runs: int = 3
) -> Dict:
    """Benchmark inference performance"""
    
    # Prepare input
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "mps":
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    elif device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    # Warmup
    console.print("[yellow]Warming up...[/yellow]")
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False
        )
    
    # Benchmark
    times = []
    tokens_generated = []
    
    console.print(f"[cyan]Running {num_runs} inference passes...[/cyan]")
    
    for i in range(num_runs):
        torch.mps.synchronize() if device == "mps" else None
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        torch.mps.synchronize() if device == "mps" else None
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        
        # Count tokens generated
        num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
        tokens_generated.append(num_tokens)
        
        console.print(f"  Run {i+1}: {elapsed:.2f}s, {num_tokens} tokens")
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_tokens = np.mean(tokens_generated)
    tokens_per_second = avg_tokens / avg_time
    
    return {
        "avg_time": avg_time,
        "std_time": std_time,
        "avg_tokens": avg_tokens,
        "tokens_per_second": tokens_per_second,
        "device": device
    }


def test_generation_quality(model, tokenizer, device: str):
    """Test generation quality with various prompts"""
    
    test_prompts = [
        "The future of artificial intelligence is",
        "def fibonacci(n):",
        "The key differences between supervised and unsupervised learning are",
        "Once upon a time in a distant galaxy",
    ]
    
    console.print("\n[bold cyan]Generation Quality Test[/bold cyan]")
    console.print("=" * 60)
    
    for prompt in test_prompts:
        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        elif device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        console.print(f"[green]Response:[/green] {response}\n")
        console.print("-" * 60)


def analyze_model_architecture(model):
    """Analyze model architecture details"""
    
    console.print("\n[bold cyan]Model Architecture Analysis[/bold cyan]")
    console.print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layers
    layer_info = {}
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type not in layer_info:
            layer_info[module_type] = 0
        layer_info[module_type] += 1
    
    # Create table
    table = Table(title="Model Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Model Size (GB)", f"{total_params * 4 / 1e9:.2f}")
    
    console.print(table)
    
    # Layer breakdown
    layer_table = Table(title="Layer Breakdown")
    layer_table.add_column("Layer Type", style="cyan")
    layer_table.add_column("Count", style="green")
    
    for layer_type, count in sorted(layer_info.items(), key=lambda x: x[1], reverse=True)[:10]:
        layer_table.add_row(layer_type, str(count))
    
    console.print(layer_table)
    
    # Attention configuration
    if hasattr(model.config, 'num_attention_heads'):
        console.print(f"\n[bold]Attention Configuration:[/bold]")
        console.print(f"  Attention Heads: {model.config.num_attention_heads}")
        console.print(f"  Hidden Size: {model.config.hidden_size}")
        console.print(f"  Intermediate Size: {model.config.intermediate_size}")
        console.print(f"  Max Position Embeddings: {model.config.max_position_embeddings}")
        
        # Check for local/global attention pattern
        if hasattr(model.config, 'sliding_window'):
            console.print(f"  Sliding Window: {model.config.sliding_window}")


def compare_devices(model_path: str):
    """Compare performance across different devices"""
    
    console.print("\n[bold cyan]Device Comparison[/bold cyan]")
    console.print("=" * 60)
    
    results = []
    prompt = "Explain the concept of attention in transformers:"
    
    # Test on available devices
    devices_to_test = []
    if torch.backends.mps.is_available():
        devices_to_test.append("mps")
    devices_to_test.append("cpu")
    
    for device in devices_to_test:
        console.print(f"\n[yellow]Testing on {device}...[/yellow]")
        
        try:
            model, tokenizer, _ = load_model(model_path, device)
            benchmark_result = benchmark_inference(
                model, tokenizer, device, prompt, 
                max_length=50, num_runs=3
            )
            results.append(benchmark_result)
            
            # Clear model from memory
            del model
            torch.mps.empty_cache() if device == "mps" else None
            
        except Exception as e:
            console.print(f"[red]Error on {device}: {e}[/red]")
    
    # Display comparison
    if results:
        table = Table(title="Device Performance Comparison")
        table.add_column("Device", style="cyan")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Tokens/s", style="green")
        
        for result in results:
            table.add_row(
                result["device"],
                f"{result['avg_time']:.2f} ± {result['std_time']:.2f}",
                f"{result['tokens_per_second']:.1f}"
            )
        
        console.print(table)
        
        # Calculate speedup
        if len(results) > 1 and results[0]["device"] == "mps":
            speedup = results[1]["avg_time"] / results[0]["avg_time"]
            console.print(f"\n[bold green]MPS Speedup: {speedup:.2f}x[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Test Gemma-3 inference")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gemma-3-4b",
        help="Path to model directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests"
    )
    parser.add_argument(
        "--compare-devices",
        action="store_true",
        help="Compare performance across devices"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze model architecture"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain how attention mechanisms work in transformers:",
        help="Custom prompt for testing"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    console.print("""
╔══════════════════════════════════════╗
║     Gemma-3 Inference Tester         ║
╚══════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        console.print(f"[red]Model not found at {args.model_path}[/red]")
        console.print("Run: python scripts/download_model.py --model gemma-3-4b")
        return
    
    # Compare devices if requested
    if args.compare_devices:
        compare_devices(args.model_path)
        return
    
    # Load model
    model, tokenizer, device = load_model(args.model_path, args.device)
    
    # Analyze architecture if requested
    if args.analyze:
        analyze_model_architecture(model)
    
    # Run benchmark if requested
    if args.benchmark:
        result = benchmark_inference(
            model, tokenizer, device, args.prompt, 
            args.max_length, num_runs=5
        )
        
        console.print("\n[bold]Benchmark Results:[/bold]")
        console.print(f"  Average time: {result['avg_time']:.2f}s ± {result['std_time']:.2f}s")
        console.print(f"  Tokens/second: {result['tokens_per_second']:.1f}")
        console.print(f"  Device: {result['device']}")
    
    # Test generation quality
    test_generation_quality(model, tokenizer, device)
    
    # Interactive mode
    console.print("\n[bold cyan]Interactive Mode[/bold cyan] (type 'exit' to quit)")
    while True:
        prompt = console.input("\n[bold]Enter prompt:[/bold] ")
        if prompt.lower() == 'exit':
            break
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "mps":
            inputs = {k: v.to("mps") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        console.print(f"\n[green]Response:[/green] {response}")


if __name__ == "__main__":
    main()