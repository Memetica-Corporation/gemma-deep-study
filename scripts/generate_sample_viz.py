#!/usr/bin/env python3
"""
Generate sample architecture visualization without requiring model download
"""

import json
from pathlib import Path
import webbrowser
import http.server
import socketserver
from rich.console import Console

console = Console()

# Sample architecture data for Gemma-3 4B
SAMPLE_DATA = {
    "model_type": "gemma",
    "vocab_size": 256000,
    "hidden_size": 3072,
    "intermediate_size": 12288,
    "num_hidden_layers": 42,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "rope_theta": 10000,
    "sliding_window": 1024,
    "total_params": 4000000000,
    "trainable_params": 4000000000,
    "attention_info": {
        "total_attention_layers": 42,
        "local_layers": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40],
        "global_layers": [5, 11, 17, 23, 29, 35, 41],
        "attention_heads": 16,
        "kv_heads": 8,
        "head_dim": 192,
        "local_to_global_ratio": 5.0
    },
    "memory_info": {
        "model_size_gb": 8.0,
        "fp32_size_gb": 16.0,
        "int8_size_gb": 4.0,
        "int4_size_gb": 2.0,
        "kv_cache_1024_gb": 0.5,
        "kv_cache_4096_gb": 2.0,
        "kv_cache_16384_gb": 8.0,
        "kv_cache_131072_gb": 64.0,
        "kv_cache_131072_optimized_gb": 12.8
    },
    "layers": [
        {
            "name": "model.embed_tokens",
            "layer_type": "Embedding",
            "params": 786432000,
            "input_shape": [1],
            "output_shape": [1, 3072],
            "attributes": {},
            "is_attention": False,
            "is_local": False,
            "is_global": False,
            "description": "Token embeddings: Converts input tokens to dense vectors"
        }
    ]
}

# Add sample layers
for i in range(42):
    is_global = (i + 1) % 6 == 0
    layer_type = "global-attention" if is_global else "local-attention"
    
    # Add attention layers
    for component in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        SAMPLE_DATA["layers"].append({
            "name": f"model.layers.{i}.self_attn.{component}",
            "layer_type": "Linear",
            "params": 9437184,
            "input_shape": [1, 3072],
            "output_shape": [1, 3072],
            "attributes": {"in_features": 3072, "out_features": 3072},
            "is_attention": True,
            "is_local": not is_global,
            "is_global": is_global,
            "description": f"{'Global' if is_global else 'Local (1024 tokens)'} attention {component}"
        })
    
    # Add MLP layers
    for component in ["gate_proj", "up_proj", "down_proj"]:
        out_size = 12288 if component != "down_proj" else 3072
        in_size = 3072 if component != "down_proj" else 12288
        SAMPLE_DATA["layers"].append({
            "name": f"model.layers.{i}.mlp.{component}",
            "layer_type": "Linear",
            "params": in_size * out_size,
            "input_shape": [1, in_size],
            "output_shape": [1, out_size],
            "attributes": {"in_features": in_size, "out_features": out_size},
            "is_attention": False,
            "is_local": False,
            "is_global": False,
            "description": f"MLP {component}"
        })

def generate_sample_visualization():
    """Generate visualization with sample data"""
    
    console.print("[cyan]Generating sample visualization...[/cyan]")
    
    # Create visualizations directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # Save sample data
    with open("visualizations/architecture_data.json", "w") as f:
        json.dump(SAMPLE_DATA, f, indent=2)
    
    console.print("[green]✅ Sample data generated![/green]")
    console.print(f"\n[bold]Model Statistics (Sample):[/bold]")
    console.print(f"  Total Parameters: 4B")
    console.print(f"  Hidden Size: 3072")
    console.print(f"  Layers: 42")
    console.print(f"  Attention Heads: 16")
    console.print(f"  KV Heads: 8 (Grouped Query Attention)")
    console.print(f"  Max Context: 128K tokens")
    console.print(f"  Local Window: 1024 tokens")
    console.print(f"  Local:Global Ratio: 5:1")

def serve_visualization(port=8080):
    """Serve the visualization"""
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="visualizations", **kwargs)
        
        def log_message(self, format, *args):
            pass  # Suppress logs
    
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

if __name__ == "__main__":
    console.print("""
╔══════════════════════════════════════╗
║  Gemma-3 Architecture Visualizer     ║
║         (Sample Data)                ║
╚══════════════════════════════════════╝
    """)
    
    generate_sample_visualization()
    serve_visualization()