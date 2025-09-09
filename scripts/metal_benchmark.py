#!/usr/bin/env python3
"""
Mac Metal/MLX Optimization Benchmarks for Gemma-3
Compares PyTorch MPS, MLX, and CPU performance
"""

import torch
import time
import argparse
import numpy as np
from pathlib import Path
import json
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# MLX imports (optional)
try:
    import mlx
    from mlx_lm import load, generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("MLX not available. Install with: pip install mlx mlx-lm")

console = Console()


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    num_runs: int = 5
    warmup_runs: int = 2
    memory_tracking: bool = True
    power_tracking: bool = False  # Requires additional tools
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8]
        if self.sequence_lengths is None:
            self.sequence_lengths = [128, 256, 512, 1024]


class MetalBenchmark:
    """Benchmark suite for Metal optimization"""
    
    def __init__(self, model_path: str, config: BenchmarkConfig):
        self.model_path = model_path
        self.config = config
        self.results = {}
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        
        stats = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
        
        # Mac-specific unified memory
        if torch.backends.mps.is_available():
            # MPS memory stats
            stats['mps_allocated_gb'] = torch.mps.driver_allocated_memory() / (1024**3) if hasattr(torch.mps, 'driver_allocated_memory') else 0
            
        return stats
        
    def benchmark_pytorch_mps(self) -> Dict:
        """Benchmark PyTorch with MPS backend"""
        console.print("[cyan]Benchmarking PyTorch MPS...[/cyan]")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Move to MPS
        device = torch.device("mps")
        model = model.to(device)
        
        results = {
            'inference_times': {},
            'memory_usage': {},
            'throughput': {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for batch_size in self.config.batch_sizes:
                for seq_len in self.config.sequence_lengths:
                    task = progress.add_task(
                        f"Testing batch={batch_size}, seq_len={seq_len}",
                        total=self.config.num_runs
                    )
                    
                    # Create dummy input
                    input_ids = torch.randint(
                        0, tokenizer.vocab_size,
                        (batch_size, seq_len),
                        device=device
                    )
                    
                    times = []
                    
                    # Warmup
                    for _ in range(self.config.warmup_runs):
                        with torch.no_grad():
                            _ = model(input_ids)
                        torch.mps.synchronize()
                    
                    # Benchmark
                    for run in range(self.config.num_runs):
                        
                        torch.mps.synchronize()
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            _ = model(input_ids)
                            
                        torch.mps.synchronize()
                        end_time = time.perf_counter()
                        
                        elapsed = end_time - start_time
                        times.append(elapsed)
                        
                        mem_after = self.get_memory_usage()
                        
                        progress.update(task, advance=1)
                    
                    # Calculate statistics
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results['inference_times'][key] = {
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times)
                    }
                    
                    # Throughput (tokens/second)
                    tokens_processed = batch_size * seq_len
                    results['throughput'][key] = tokens_processed / np.mean(times)
                    
                    # Memory usage
                    results['memory_usage'][key] = {
                        'peak_gb': mem_after.get('mps_allocated_gb', 0)
                    }
                    
        # Cleanup
        del model
        torch.mps.empty_cache()
        
        return results
        
    def benchmark_pytorch_cpu(self) -> Dict:
        """Benchmark PyTorch with CPU backend"""
        console.print("[cyan]Benchmarking PyTorch CPU...[/cyan]")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        results = {
            'inference_times': {},
            'memory_usage': {},
            'throughput': {}
        }
        
        # Test smaller configurations for CPU
        cpu_batch_sizes = [1, 2]
        cpu_seq_lengths = [128, 256]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            for batch_size in cpu_batch_sizes:
                for seq_len in cpu_seq_lengths:
                    task = progress.add_task(
                        f"Testing batch={batch_size}, seq_len={seq_len}",
                        total=self.config.num_runs
                    )
                    
                    # Create dummy input
                    input_ids = torch.randint(
                        0, tokenizer.vocab_size,
                        (batch_size, seq_len)
                    )
                    
                    times = []
                    
                    # Warmup
                    for _ in range(self.config.warmup_runs):
                        with torch.no_grad():
                            _ = model(input_ids)
                    
                    # Benchmark
                    for run in range(self.config.num_runs):
                        mem_before = self.get_memory_usage()
                        
                        start_time = time.perf_counter()
                        
                        with torch.no_grad():
                            _ = model(input_ids)
                            
                        end_time = time.perf_counter()
                        
                        elapsed = end_time - start_time
                        times.append(elapsed)
                        
                        mem_after = self.get_memory_usage()
                        
                        progress.update(task, advance=1)
                    
                    # Calculate statistics
                    key = f"batch_{batch_size}_seq_{seq_len}"
                    results['inference_times'][key] = {
                        'mean': np.mean(times),
                        'std': np.std(times),
                        'min': np.min(times),
                        'max': np.max(times)
                    }
                    
                    # Throughput
                    tokens_processed = batch_size * seq_len
                    results['throughput'][key] = tokens_processed / np.mean(times)
                    
                    # Memory usage
                    results['memory_usage'][key] = {
                        'peak_gb': mem_after['used_gb'] - mem_before['used_gb']
                    }
                    
        # Cleanup
        del model
        
        return results
        
    def benchmark_mlx(self) -> Optional[Dict]:
        """Benchmark MLX framework"""
        if not MLX_AVAILABLE:
            console.print("[yellow]MLX not available, skipping...[/yellow]")
            return None
            
        console.print("[cyan]Benchmarking MLX...[/cyan]")
        
        try:
            # Load model in MLX format
            model, tokenizer = load(self.model_path)
            
            results = {
                'inference_times': {},
                'memory_usage': {},
                'throughput': {}
            }
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                
                for batch_size in self.config.batch_sizes:
                    for seq_len in self.config.sequence_lengths:
                        task = progress.add_task(
                            f"Testing batch={batch_size}, seq_len={seq_len}",
                            total=self.config.num_runs
                        )
                        
                        # Create dummy input
                        prompt = "The " * (seq_len // 2)  # Simple prompt
                        
                        times = []
                        
                        # Warmup
                        for _ in range(self.config.warmup_runs):
                            _ = generate(model, tokenizer, prompt, max_tokens=10)
                        
                        # Benchmark
                        for run in range(self.config.num_runs):
                            start_time = time.perf_counter()
                            
                            _ = generate(
                                model, tokenizer, prompt,
                                max_tokens=50,
                                verbose=False
                            )
                            
                            end_time = time.perf_counter()
                            
                            elapsed = end_time - start_time
                            times.append(elapsed)
                            
                            progress.update(task, advance=1)
                        
                        # Calculate statistics
                        key = f"batch_{batch_size}_seq_{seq_len}"
                        results['inference_times'][key] = {
                            'mean': np.mean(times),
                            'std': np.std(times),
                            'min': np.min(times),
                            'max': np.max(times)
                        }
                        
                        # Approximate throughput
                        tokens_processed = seq_len + 50  # Input + generated
                        results['throughput'][key] = tokens_processed / np.mean(times)
                        
            return results
            
        except Exception as e:
            console.print(f"[red]MLX benchmark failed: {e}[/red]")
            return None
            
    def compare_results(self):
        """Compare and visualize benchmark results"""
        
        # Create comparison table
        table = Table(title="Performance Comparison")
        table.add_column("Configuration", style="cyan")
        table.add_column("PyTorch MPS", style="green")
        table.add_column("PyTorch CPU", style="yellow")
        if 'mlx' in self.results and self.results['mlx']:
            table.add_column("MLX", style="magenta")
            
        # Add rows for each configuration
        all_configs = set()
        for backend in self.results.values():
            if backend and 'throughput' in backend:
                all_configs.update(backend['throughput'].keys())
                
        for config in sorted(all_configs):
            row = [config]
            
            # MPS throughput
            if 'pytorch_mps' in self.results:
                mps_throughput = self.results['pytorch_mps']['throughput'].get(config, 0)
                row.append(f"{mps_throughput:.1f} tok/s")
            else:
                row.append("N/A")
                
            # CPU throughput
            if 'pytorch_cpu' in self.results:
                cpu_throughput = self.results['pytorch_cpu']['throughput'].get(config, 0)
                row.append(f"{cpu_throughput:.1f} tok/s")
            else:
                row.append("N/A")
                
            # MLX throughput
            if 'mlx' in self.results and self.results['mlx']:
                mlx_throughput = self.results['mlx']['throughput'].get(config, 0)
                row.append(f"{mlx_throughput:.1f} tok/s")
                
            table.add_row(*row)
            
        console.print(table)
        
        # Calculate speedups
        if 'pytorch_mps' in self.results and 'pytorch_cpu' in self.results:
            console.print("\n[bold]Speedup Analysis:[/bold]")
            
            for config in all_configs:
                if (config in self.results['pytorch_mps']['throughput'] and 
                    config in self.results['pytorch_cpu']['throughput']):
                    
                    mps_throughput = self.results['pytorch_mps']['throughput'][config]
                    cpu_throughput = self.results['pytorch_cpu']['throughput'][config]
                    
                    if cpu_throughput > 0:
                        speedup = mps_throughput / cpu_throughput
                        console.print(f"  {config}: MPS is {speedup:.2f}x faster than CPU")
                        
    def plot_results(self, save_path: Optional[str] = None):
        """Create visualization of benchmark results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Throughput comparison
        backends = []
        throughputs = []
        
        for backend_name, backend_results in self.results.items():
            if backend_results and 'throughput' in backend_results:
                backends.append(backend_name)
                avg_throughput = np.mean(list(backend_results['throughput'].values()))
                throughputs.append(avg_throughput)
                
        axes[0, 0].bar(backends, throughputs, color=['green', 'yellow', 'magenta'][:len(backends)])
        axes[0, 0].set_ylabel('Average Throughput (tokens/s)')
        axes[0, 0].set_title('Backend Performance Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Batch size scaling
        if 'pytorch_mps' in self.results:
            batch_sizes = []
            mps_times = []
            
            for config, times in self.results['pytorch_mps']['inference_times'].items():
                batch_size = int(config.split('_')[1])
                if batch_size not in batch_sizes:
                    batch_sizes.append(batch_size)
                    mps_times.append(times['mean'])
                    
            axes[0, 1].plot(batch_sizes, mps_times, 'o-', label='MPS', linewidth=2)
            
            if 'pytorch_cpu' in self.results:
                cpu_times = []
                for batch_size in batch_sizes[:2]:  # CPU only tested smaller batches
                    config = f"batch_{batch_size}_seq_128"
                    if config in self.results['pytorch_cpu']['inference_times']:
                        cpu_times.append(self.results['pytorch_cpu']['inference_times'][config]['mean'])
                        
                if cpu_times:
                    axes[0, 1].plot(batch_sizes[:len(cpu_times)], cpu_times, 
                                  's-', label='CPU', linewidth=2)
                    
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Inference Time (s)')
            axes[0, 1].set_title('Batch Size Scaling')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # Plot 3: Memory usage
        if 'pytorch_mps' in self.results:
            configs = []
            memory_usage = []
            
            for config, mem in self.results['pytorch_mps']['memory_usage'].items():
                configs.append(config.replace('batch_', 'B').replace('_seq_', '/S'))
                memory_usage.append(mem['peak_gb'])
                
            axes[1, 0].bar(range(len(configs)), memory_usage, tick_label=configs)
            axes[1, 0].set_xlabel('Configuration')
            axes[1, 0].set_ylabel('Peak Memory (GB)')
            axes[1, 0].set_title('Memory Usage (MPS)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
        # Plot 4: Efficiency metrics
        axes[1, 1].axis('off')
        
        efficiency_text = "Efficiency Metrics\n" + "="*30 + "\n\n"
        
        if 'pytorch_mps' in self.results:
            avg_throughput = np.mean(list(self.results['pytorch_mps']['throughput'].values()))
            efficiency_text += f"MPS Avg Throughput: {avg_throughput:.1f} tok/s\n"
            
        if 'pytorch_cpu' in self.results:
            avg_throughput = np.mean(list(self.results['pytorch_cpu']['throughput'].values()))
            efficiency_text += f"CPU Avg Throughput: {avg_throughput:.1f} tok/s\n"
            
        if 'mlx' in self.results and self.results['mlx']:
            avg_throughput = np.mean(list(self.results['mlx']['throughput'].values()))
            efficiency_text += f"MLX Avg Throughput: {avg_throughput:.1f} tok/s\n"
            
        # Add memory efficiency
        efficiency_text += f"\nSystem Memory: {self.get_memory_usage()['total_gb']:.1f} GB\n"
        efficiency_text += f"Available: {self.get_memory_usage()['available_gb']:.1f} GB\n"
        
        axes[1, 1].text(0.1, 0.5, efficiency_text, fontsize=11,
                       family='monospace', verticalalignment='center')
        
        plt.suptitle('Metal/MLX Optimization Benchmarks', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        
    def run_all_benchmarks(self):
        """Run all available benchmarks"""
        
        # PyTorch MPS
        if torch.backends.mps.is_available():
            self.results['pytorch_mps'] = self.benchmark_pytorch_mps()
        else:
            console.print("[yellow]MPS not available[/yellow]")
            
        # PyTorch CPU
        self.results['pytorch_cpu'] = self.benchmark_pytorch_cpu()
        
        # MLX
        if MLX_AVAILABLE:
            mlx_results = self.benchmark_mlx()
            if mlx_results:
                self.results['mlx'] = mlx_results
                
        # Compare results
        self.compare_results()
        
        # Visualize
        self.plot_results(save_path="visualizations/metal_benchmark.png")
        
        # Save results
        self.save_results("visualizations/benchmark_results.json")
        
    def save_results(self, filepath: str):
        """Save benchmark results to JSON"""
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
            
        serializable_results = convert_to_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        console.print(f"[green]Results saved to {filepath}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Metal/MLX optimization benchmarks")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gemma-3-4b",
        help="Path to model"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["mps", "cpu", "mlx"],
        help="Backends to benchmark"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        type=int,
        default=[128, 256, 512],
        help="Sequence lengths to test"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs"
    )
    
    args = parser.parse_args()
    
    console.print("""
╔══════════════════════════════════════╗
║   Metal/MLX Optimization Benchmark    ║
╚══════════════════════════════════════╝
    """)
    
    # Check model exists
    if not Path(args.model_path).exists():
        console.print(f"[red]Model not found at {args.model_path}[/red]")
        return
        
    # Create benchmark config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.seq_lengths,
        num_runs=args.num_runs
    )
    
    # Run benchmarks
    benchmark = MetalBenchmark(args.model_path, config)
    benchmark.run_all_benchmarks()
    
    console.print("\n[bold green]Benchmarking complete![/bold green]")


if __name__ == "__main__":
    main()