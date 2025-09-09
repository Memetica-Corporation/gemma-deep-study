#!/usr/bin/env python3
"""
Generate interactive architecture visualization for Gemma-3
Creates both data export and launches web server
"""

import argparse
import json
import webbrowser
import http.server
import socketserver
import threading
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from analyze.model_architecture_explorer import GemmaArchitectureAnalyzer
from rich.console import Console

console = Console()


def generate_visualization(model_path: str, output_dir: str = "visualizations"):
    """Generate architecture visualization files"""
    
    console.print("[cyan]Analyzing Gemma-3 architecture...[/cyan]")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Analyze model
    analyzer = GemmaArchitectureAnalyzer(model_path)
    architecture_data = analyzer.analyze_architecture()
    
    # Export JSON data
    json_path = Path(output_dir) / "architecture_data.json"
    analyzer.export_to_json(str(json_path))
    
    # Create Plotly visualization
    fig = analyzer.create_plotly_visualization()
    plotly_path = Path(output_dir) / "architecture_plot.html"
    fig.write_html(str(plotly_path))
    
    # Print summary
    console.print("\n[bold green]✅ Visualization Generated![/bold green]")
    console.print(f"\n[bold]Model Statistics:[/bold]")
    
    stats_table = f"""
    Total Parameters: {architecture_data['total_params']:,}
    Model Size: {architecture_data['memory_info']['model_size_gb']:.2f} GB
    Hidden Size: {architecture_data['hidden_size']}
    Layers: {architecture_data['num_hidden_layers']}
    Attention Heads: {architecture_data['num_attention_heads']}
    KV Heads: {architecture_data['num_key_value_heads']}
    Vocab Size: {architecture_data['vocab_size']:,}
    Max Context: {architecture_data['max_position_embeddings']:,} tokens
    """
    
    console.print(stats_table)
    
    # Attention pattern info
    attention_info = architecture_data['attention_info']
    console.print(f"\n[bold]Attention Pattern:[/bold]")
    console.print(f"  Local Layers: {len(attention_info['local_layers'])}")
    console.print(f"  Global Layers: {len(attention_info['global_layers'])}")
    console.print(f"  Ratio: {attention_info['local_to_global_ratio']:.1f}:1")
    
    # Memory info
    mem_info = architecture_data['memory_info']
    console.print(f"\n[bold]Memory Requirements:[/bold]")
    console.print(f"  BF16: {mem_info['model_size_gb']:.2f} GB")
    console.print(f"  INT8: {mem_info['int8_size_gb']:.2f} GB")
    console.print(f"  INT4: {mem_info['int4_size_gb']:.2f} GB")
    console.print(f"  KV-Cache (128K): {mem_info.get('kv_cache_131072_gb', 'N/A'):.2f} GB")
    console.print(f"  KV-Cache (128K, optimized): {mem_info.get('kv_cache_131072_optimized_gb', 'N/A'):.2f} GB")
    
    return architecture_data


def serve_visualization(port: int = 8080):
    """Start local web server for visualization"""
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="visualizations", **kwargs)
            
        def log_message(self, format, *args):
            # Suppress request logs
            pass
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        console.print(f"\n[bold cyan]🌐 Web server started![/bold cyan]")
        console.print(f"View visualization at: [link]http://localhost:{port}/gemma_architecture.html[/link]")
        console.print("[dim]Press Ctrl+C to stop the server[/dim]\n")
        
        # Open browser
        webbrowser.open(f"http://localhost:{port}/gemma_architecture.html")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Server stopped[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Generate Gemma-3 architecture visualization")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/gemma-3-4b",
        help="Path to model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Output directory for visualization files"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start web server to view visualization"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for web server"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    console.print("""
╔══════════════════════════════════════╗
║  Gemma-3 Architecture Visualizer     ║
╚══════════════════════════════════════╝
    """)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        console.print(f"[red]Model not found at {args.model_path}[/red]")
        console.print("Run: python scripts/download_model.py --model gemma-3-4b")
        return
    
    # Generate visualization
    architecture_data = generate_visualization(args.model_path, args.output_dir)
    
    # Serve if requested
    if args.serve:
        serve_visualization(args.port)
    else:
        console.print(f"\n[bold]To view the visualization:[/bold]")
        console.print(f"1. Run: python {__file__} --serve")
        console.print(f"2. Or open: {Path(args.output_dir).absolute()}/gemma_architecture.html")


if __name__ == "__main__":
    main()