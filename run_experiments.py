#!/usr/bin/env python3
"""
Main experiment runner for Gemma-3 deep architecture study
Orchestrates all experiments and generates comprehensive reports
"""

import argparse
import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

# Import experiment modules
sys.path.append(str(Path(__file__).parent))

from experiments.lora.rank1_dynamic_merger import Rank1MergeConfig
from training.relora import create_relora_trainer
from analyze.capacity_meter import run_capacity_experiment
from analyze.attention_visualizer import run_attention_analysis
from experiments.attention.set_block_decoder import SBDConfig, create_sbd_accelerator

console = Console()


def display_menu():
    """Display interactive experiment menu"""
    
    menu_text = """
[bold cyan]Gemma-3 Deep Architecture Study[/bold cyan]
    
Available Experiments:
    
1. [green]LoRA Experiments[/green]
   a. Dynamic Rank-1 Merging
   b. ReLoRA Training
   c. Compare LoRA techniques
   
2. [yellow]Capacity Analysis[/yellow]
   a. Measure model capacity
   b. Track memorization vs generalization
   c. Detect grokking behavior
   
3. [blue]Attention Analysis[/blue]
   a. Visualize local vs global patterns
   b. Analyze attention flow
   c. Interactive attention explorer
   
4. [magenta]Inference Optimization[/magenta]
   a. Set Block Decoding benchmark
   b. Metal/MLX performance comparison
   c. Quantization experiments
   
5. [red]Full Pipeline[/red]
   Run complete experiment suite
   
6. [dim]Exit[/dim]
    """
    
    console.print(Panel(menu_text, title="Experiment Menu", border_style="bold"))
    
    return Prompt.ask(
        "Select experiment",
        choices=["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4a", "4b", "4c", "5", "6"],
        default="6"
    )


def run_rank1_experiment(model_path: str):
    """Run dynamic rank-1 merging experiment"""
    console.print("\n[bold cyan]Dynamic Rank-1 LoRA Merging Experiment[/bold cyan]")
    
    from transformers import AutoModelForCausalLM
    from torch.utils.data import DataLoader
    
    # Load model
    console.print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create config
    config = Rank1MergeConfig(
        merge_frequency=1,
        alpha=1.0,
        track_svd=True,
        svd_frequency=100
    )
    
    # Create trainer
    from experiments.lora.rank1_dynamic_merger import Rank1MergeTrainer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Rank1MergeTrainer(model, optimizer, config)
    
    console.print("[green]Rank-1 merger initialized![/green]")
    console.print(f"Merge frequency: {config.merge_frequency}")
    console.print(f"Alpha: {config.alpha}")
    
    # Create dummy data for demonstration
    console.print("\nCreating sample dataset...")
    from analyze.capacity_meter import BitStringDataset
    dataset = BitStringDataset(num_samples=100, sequence_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Train for one epoch
    console.print("\nTraining with rank-1 merging...")
    avg_loss = trainer.train_epoch(dataloader)
    
    console.print(f"\n[green]Training complete![/green]")
    console.print(f"Average loss: {avg_loss:.4f}")
    console.print(f"Statistics: {trainer.merger.get_statistics()}")
    
    # Visualize rank evolution
    if Confirm.ask("Visualize rank evolution?"):
        trainer.merger.plot_rank_evolution()
        

def run_relora_experiment(model_path: str):
    """Run ReLoRA training experiment"""
    console.print("\n[bold cyan]ReLoRA Training Experiment[/bold cyan]")
    
    from transformers import AutoModelForCausalLM
    from torch.utils.data import DataLoader
    
    # Load model
    console.print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Create ReLoRA trainer
    trainer = create_relora_trainer(model, learning_rate=1e-4, total_steps=5000)
    
    console.print("[green]ReLoRA trainer initialized![/green]")
    console.print(f"Warmup steps: {trainer.config.warmup_steps}")
    console.print(f"Reset interval: {trainer.config.reset_interval}")
    console.print(f"LoRA rank: {trainer.config.lora_rank}")
    
    # Create dummy data
    console.print("\nCreating sample dataset...")
    from analyze.capacity_meter import BitStringDataset
    dataset = BitStringDataset(num_samples=100, sequence_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Run warmup
    console.print("\nRunning warmup phase...")
    trainer.warmup_phase(dataloader, num_steps=100)
    
    # Run ReLoRA phase
    console.print("\nRunning ReLoRA phase...")
    trainer.relora_phase(dataloader, num_resets=2)
    
    console.print(f"\n[green]ReLoRA training complete![/green]")
    console.print(f"Total resets: {trainer.reset_count}")
    
    # Visualize training
    if Confirm.ask("Visualize training history?"):
        trainer.plot_training_history()
        

def run_capacity_analysis(model_path: str):
    """Run model capacity analysis"""
    console.print("\n[bold cyan]Model Capacity Analysis[/bold cyan]")
    
    from transformers import AutoModelForCausalLM
    
    # Load model
    console.print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Run capacity experiment
    analyzer, train_dataset, test_dataset = run_capacity_experiment(
        model, num_samples=500, sequence_length=128
    )
    
    # Simulate training for analysis
    console.print("\nSimulating training to measure memorization...")
    for step in range(10):
        # Simulate training step
        loss = 2.0 - (step * 0.1)  # Decreasing loss
        analyzer.track_training_step(step, loss, train_dataset, test_dataset)
        
    # Get report
    report = analyzer.get_capacity_report()
    
    console.print("\n[bold]Capacity Report:[/bold]")
    console.print(f"Bits per parameter: {report.bits_per_param}")
    console.print(f"Total capacity: {report.total_capacity_bits:,} bits")
    console.print(f"Memorization ratio: {report.memorization_ratio:.2%}")
    console.print(f"Generalization ratio: {report.generalization_ratio:.2%}")
    
    # Visualize
    if Confirm.ask("Visualize capacity analysis?"):
        analyzer.plot_capacity_analysis()
        

def run_attention_analysis(model_path: str):
    """Run attention pattern analysis"""
    console.print("\n[bold cyan]Attention Pattern Analysis[/bold cyan]")
    
    sample_text = Prompt.ask(
        "Enter text to analyze",
        default="The attention mechanism in transformers allows models to focus on relevant parts."
    )
    
    # Run analysis
    analyzer, patterns, flow_analysis = run_attention_analysis(model_path, sample_text)
    
    console.print(f"\n[green]Analysis complete![/green]")
    console.print(f"Extracted {len(patterns)} attention patterns")
    console.print(f"Information bottlenecks: {len(flow_analysis['information_bottlenecks'])}")
    console.print(f"Attention highways: {len(flow_analysis['attention_highways'])}")
    
    # Interactive exploration
    if Confirm.ask("Launch interactive attention explorer?"):
        for pattern in patterns[:3]:
            if pattern.head_idx == -1:  # Averaged patterns
                fig = analyzer.create_interactive_visualization(pattern)
                fig.show()
                

def run_sbd_benchmark(model_path: str):
    """Run Set Block Decoding benchmark"""
    console.print("\n[bold cyan]Set Block Decoding Benchmark[/bold cyan]")
    
    # Create SBD config
    config = SBDConfig(
        block_size=4,
        masking_strategy="adaptive",
        masking_ratio=0.5,
        confidence_threshold=0.8
    )
    
    # Create accelerator
    console.print("Initializing SBD accelerator...")
    accelerator = create_sbd_accelerator(model_path, config)
    
    console.print(f"Block size: {config.block_size}")
    console.print(f"Masking strategy: {config.masking_strategy}")
    console.print(f"Confidence threshold: {config.confidence_threshold}")
    
    # Run benchmark
    prompt = Prompt.ask(
        "Enter prompt for generation",
        default="Explain the concept of attention in transformers:"
    )
    
    console.print("\nRunning benchmark...")
    results = accelerator.benchmark_vs_standard(
        prompt, max_new_tokens=50, num_runs=3
    )
    
    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(f"SBD Speed: {results['sbd_avg_speed']:.1f} tokens/s")
    console.print(f"Standard Speed: {results['standard_avg_speed']:.1f} tokens/s")
    console.print(f"Speedup: {results['speedup_factor']:.2f}x")
    

def run_full_pipeline(model_path: str):
    """Run complete experiment pipeline"""
    console.print("\n[bold cyan]Running Full Experiment Pipeline[/bold cyan]")
    
    experiments = [
        ("Capacity Analysis", run_capacity_analysis),
        ("Attention Analysis", run_attention_analysis),
        ("Rank-1 LoRA", run_rank1_experiment),
        ("ReLoRA Training", run_relora_experiment),
        ("SBD Benchmark", run_sbd_benchmark)
    ]
    
    for name, experiment_fn in experiments:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]{name}[/bold]")
        console.print('='*60)
        
        try:
            experiment_fn(model_path)
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            if not Confirm.ask("Continue with next experiment?"):
                break
                
    console.print("\n[bold green]Pipeline complete![/bold green]")
    

def main():
    parser = argparse.ArgumentParser(description="Gemma-3 experiment runner")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gemma-3-4b",
        help="Path to model"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["rank1", "relora", "capacity", "attention", "sbd", "full", "interactive"],
        default="interactive",
        help="Experiment to run"
    )
    
    args = parser.parse_args()
    
    console.print("""
╔══════════════════════════════════════╗
║   Gemma-3 Deep Architecture Study    ║
║         Experiment Runner            ║
╚══════════════════════════════════════╝
    """)
    
    # Check model exists
    if not Path(args.model_path).exists():
        console.print(f"[red]Model not found at {args.model_path}[/red]")
        console.print("Run: python scripts/download_model.py --model gemma-3-4b")
        return
        
    # Create output directories
    Path("visualizations").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    
    # Run experiments
    if args.experiment == "interactive":
        while True:
            choice = display_menu()
            
            if choice == "6":
                console.print("[dim]Exiting...[/dim]")
                break
            elif choice == "1a":
                run_rank1_experiment(args.model_path)
            elif choice == "1b":
                run_relora_experiment(args.model_path)
            elif choice == "2a":
                run_capacity_analysis(args.model_path)
            elif choice == "3a":
                run_attention_analysis(args.model_path)
            elif choice == "4a":
                run_sbd_benchmark(args.model_path)
            elif choice == "5":
                run_full_pipeline(args.model_path)
                
            if not Confirm.ask("\nRun another experiment?"):
                break
                
    else:
        # Run specific experiment
        experiment_map = {
            "rank1": run_rank1_experiment,
            "relora": run_relora_experiment,
            "capacity": run_capacity_analysis,
            "attention": run_attention_analysis,
            "sbd": run_sbd_benchmark,
            "full": run_full_pipeline
        }
        
        experiment_fn = experiment_map[args.experiment]
        experiment_fn(args.model_path)
        
    console.print("\n[bold green]Thank you for using Gemma-3 Deep Architecture Study![/bold green]")


if __name__ == "__main__":
    main()