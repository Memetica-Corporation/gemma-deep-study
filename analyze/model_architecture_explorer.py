#!/usr/bin/env python3
"""
Gemma-3 Model Architecture Explorer
Generates an interactive web visualization of the model architecture
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, asdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class LayerInfo:
    """Information about a model layer"""
    name: str
    layer_type: str
    params: int
    input_shape: List[int]
    output_shape: List[int]
    attributes: Dict[str, Any]
    is_attention: bool
    is_local: bool
    is_global: bool
    description: str


class GemmaArchitectureAnalyzer:
    """Analyze and visualize Gemma-3 architecture"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # CPU for analysis
        )
        self.layers_info = []
        self.architecture_data = {}
        
    def analyze_architecture(self) -> Dict:
        """Perform complete architecture analysis"""
        
        # Extract config information
        self.architecture_data = {
            'model_type': self.config.model_type,
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'intermediate_size': self.config.intermediate_size,
            'num_hidden_layers': self.config.num_hidden_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'num_key_value_heads': getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads),
            'max_position_embeddings': self.config.max_position_embeddings,
            'rope_theta': getattr(self.config, 'rope_theta', 10000),
            'sliding_window': getattr(self.config, 'sliding_window', None),
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        # Analyze each layer
        self._analyze_layers()
        
        # Analyze attention patterns
        self._analyze_attention_patterns()
        
        # Calculate memory requirements
        self._calculate_memory_requirements()
        
        return self.architecture_data
        
    def _analyze_layers(self):
        """Analyze individual layers"""
        layers_info = []
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = self._extract_layer_info(name, module)
                if layer_info:
                    layers_info.append(layer_info)
                    
        self.layers_info = layers_info
        self.architecture_data['layers'] = [asdict(l) for l in layers_info]
        
    def _extract_layer_info(self, name: str, module: nn.Module) -> Optional[LayerInfo]:
        """Extract information from a single layer"""
        
        # Skip container modules
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            return None
            
        # Determine layer type
        layer_type = type(module).__name__
        
        # Count parameters
        params = sum(p.numel() for p in module.parameters())
        if params == 0:
            return None
            
        # Determine if it's attention-related
        is_attention = any(x in name.lower() for x in ['attention', 'attn', 'q_proj', 'k_proj', 'v_proj', 'o_proj'])
        
        # Determine if local or global (Gemma-3 pattern)
        layer_idx = self._extract_layer_index(name)
        is_local = False
        is_global = False
        
        if layer_idx is not None:
            # Gemma-3: every 6th layer is global, others are local
            is_global = (layer_idx + 1) % 6 == 0
            is_local = not is_global and is_attention
            
        # Extract attributes
        attributes = {}
        if hasattr(module, 'in_features'):
            attributes['in_features'] = module.in_features
        if hasattr(module, 'out_features'):
            attributes['out_features'] = module.out_features
        if hasattr(module, 'hidden_size'):
            attributes['hidden_size'] = module.hidden_size
        if hasattr(module, 'num_heads'):
            attributes['num_heads'] = module.num_heads
            
        # Generate description
        description = self._generate_layer_description(name, module, layer_type, is_local, is_global)
        
        # Estimate shapes (simplified)
        input_shape = []
        output_shape = []
        
        if isinstance(module, nn.Linear):
            input_shape = [1, module.in_features]
            output_shape = [1, module.out_features]
        elif isinstance(module, nn.Embedding):
            input_shape = [1]
            output_shape = [1, module.embedding_dim]
            
        return LayerInfo(
            name=name,
            layer_type=layer_type,
            params=params,
            input_shape=input_shape,
            output_shape=output_shape,
            attributes=attributes,
            is_attention=is_attention,
            is_local=is_local,
            is_global=is_global,
            description=description
        )
        
    def _extract_layer_index(self, name: str) -> Optional[int]:
        """Extract layer index from name"""
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return None
        
    def _generate_layer_description(self, name: str, module: nn.Module, layer_type: str, 
                                   is_local: bool, is_global: bool) -> str:
        """Generate human-readable description of layer"""
        
        if 'embed' in name.lower():
            return "Token embeddings: Converts input tokens to dense vectors"
            
        elif 'q_proj' in name:
            window_type = "local (1024 tokens)" if is_local else "global (full sequence)"
            return f"Query projection for {window_type} attention"
            
        elif 'k_proj' in name:
            window_type = "local (1024 tokens)" if is_local else "global (full sequence)"
            return f"Key projection for {window_type} attention"
            
        elif 'v_proj' in name:
            window_type = "local (1024 tokens)" if is_local else "global (full sequence)"
            return f"Value projection for {window_type} attention"
            
        elif 'o_proj' in name:
            return "Output projection: Combines attention heads"
            
        elif 'gate_proj' in name:
            return "Gate projection: Controls information flow in MLP"
            
        elif 'up_proj' in name:
            return "Up projection: Expands to higher dimension in MLP"
            
        elif 'down_proj' in name:
            return "Down projection: Reduces dimension back in MLP"
            
        elif 'norm' in name.lower():
            return "Layer normalization: Stabilizes training"
            
        elif 'lm_head' in name:
            return "Language model head: Projects to vocabulary for next token prediction"
            
        else:
            return f"{layer_type} layer"
            
    def _analyze_attention_patterns(self):
        """Analyze attention patterns in the model"""
        
        attention_info = {
            'total_attention_layers': 0,
            'local_layers': [],
            'global_layers': [],
            'attention_heads': self.architecture_data['num_attention_heads'],
            'kv_heads': self.architecture_data['num_key_value_heads'],
            'head_dim': self.architecture_data['hidden_size'] // self.architecture_data['num_attention_heads']
        }
        
        for i in range(self.architecture_data['num_hidden_layers']):
            if (i + 1) % 6 == 0:
                attention_info['global_layers'].append(i)
            else:
                attention_info['local_layers'].append(i)
                
        attention_info['total_attention_layers'] = len(attention_info['local_layers']) + len(attention_info['global_layers'])
        attention_info['local_to_global_ratio'] = len(attention_info['local_layers']) / max(1, len(attention_info['global_layers']))
        
        self.architecture_data['attention_info'] = attention_info
        
    def _calculate_memory_requirements(self):
        """Calculate memory requirements for different scenarios"""
        
        param_bytes = self.architecture_data['total_params'] * 2  # bfloat16
        
        memory_info = {
            'model_size_gb': param_bytes / (1024**3),
            'fp32_size_gb': self.architecture_data['total_params'] * 4 / (1024**3),
            'int8_size_gb': self.architecture_data['total_params'] / (1024**3),
            'int4_size_gb': self.architecture_data['total_params'] * 0.5 / (1024**3),
        }
        
        # Estimate KV cache for different sequence lengths
        num_layers = self.architecture_data['num_hidden_layers']
        hidden_size = self.architecture_data['hidden_size']
        num_kv_heads = self.architecture_data['num_key_value_heads']
        head_dim = hidden_size // self.architecture_data['num_attention_heads']
        
        for seq_len in [1024, 4096, 16384, 131072]:  # 128K max
            # KV cache size per token
            kv_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # 2 for K and V, 2 bytes for bf16
            total_kv_cache = seq_len * kv_per_token
            
            # Account for local vs global optimization
            # Local layers only store 1024 tokens, global stores full sequence
            local_layers = len(self.architecture_data['attention_info']['local_layers'])
            global_layers = len(self.architecture_data['attention_info']['global_layers'])
            
            optimized_cache = (
                local_layers * min(1024, seq_len) * num_kv_heads * head_dim * 2 * 2 +  # Local
                global_layers * seq_len * num_kv_heads * head_dim * 2 * 2  # Global
            )
            
            memory_info[f'kv_cache_{seq_len}_gb'] = total_kv_cache / (1024**3)
            memory_info[f'kv_cache_{seq_len}_optimized_gb'] = optimized_cache / (1024**3)
            
        self.architecture_data['memory_info'] = memory_info
        
    def generate_layer_graph(self) -> Dict:
        """Generate graph representation of model layers"""
        
        nodes = []
        edges = []
        
        # Create nodes for each layer type
        layer_groups = {}
        for layer in self.layers_info:
            layer_idx = self._extract_layer_index(layer.name)
            if layer_idx is not None:
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = []
                layer_groups[layer_idx].append(layer)
                
        # Create simplified nodes
        for idx, layers in sorted(layer_groups.items()):
            # Group layers by type
            has_attention = any(l.is_attention for l in layers)
            is_local = any(l.is_local for l in layers)
            is_global = any(l.is_global for l in layers)
            
            node = {
                'id': f'layer_{idx}',
                'label': f'Layer {idx}',
                'type': 'local' if is_local else 'global' if is_global else 'feedforward',
                'params': sum(l.params for l in layers),
                'components': [l.layer_type for l in layers]
            }
            nodes.append(node)
            
            # Create edges
            if idx > 0:
                edges.append({
                    'source': f'layer_{idx-1}',
                    'target': f'layer_{idx}'
                })
                
        return {'nodes': nodes, 'edges': edges}
        
    def create_plotly_visualization(self) -> go.Figure:
        """Create interactive Plotly visualization"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Parameter Distribution', 'Layer Types', 
                          'Memory Requirements', 'Attention Pattern'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. Parameter distribution
        layer_names = []
        param_counts = []
        colors = []
        
        for layer in self.layers_info[:30]:  # Top 30 layers
            if layer.params > 1000:  # Skip tiny layers
                layer_names.append(layer.name.split('.')[-1])
                param_counts.append(layer.params)
                if layer.is_local:
                    colors.append('lightblue')
                elif layer.is_global:
                    colors.append('lightcoral')
                else:
                    colors.append('lightgreen')
                    
        fig.add_trace(
            go.Bar(x=layer_names, y=param_counts, marker_color=colors,
                  name='Parameters'),
            row=1, col=1
        )
        
        # 2. Layer type distribution
        layer_types = {}
        for layer in self.layers_info:
            lt = layer.layer_type
            if lt not in layer_types:
                layer_types[lt] = 0
            layer_types[lt] += 1
            
        fig.add_trace(
            go.Pie(labels=list(layer_types.keys()), 
                  values=list(layer_types.values())),
            row=1, col=2
        )
        
        # 3. Memory requirements
        memory_data = self.architecture_data['memory_info']
        configs = ['fp32', 'int8', 'int4']
        sizes = [memory_data[f'{c}_size_gb'] for c in configs]
        
        fig.add_trace(
            go.Bar(x=configs, y=sizes, name='Model Size (GB)'),
            row=2, col=1
        )
        
        # 4. Attention pattern
        layers = list(range(self.architecture_data['num_hidden_layers']))
        types = ['Local' if (i+1) % 6 != 0 else 'Global' for i in layers]
        colors = ['blue' if t == 'Local' else 'red' for t in types]
        
        fig.add_trace(
            go.Scatter(x=layers, y=[1]*len(layers), mode='markers',
                      marker=dict(size=10, color=colors),
                      text=types, name='Attention Type'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Gemma-3 Architecture Analysis",
            showlegend=False,
            height=800
        )
        
        return fig
        
    def export_to_json(self, output_path: str):
        """Export architecture data to JSON"""
        
        # Convert to serializable format
        def make_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            return obj
            
        data = make_serializable(self.architecture_data)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Architecture data exported to {output_path}")


def analyze_model(model_path: str) -> Dict:
    """Main function to analyze model architecture"""
    
    print("Analyzing Gemma-3 architecture...")
    analyzer = GemmaArchitectureAnalyzer(model_path)
    
    architecture_data = analyzer.analyze_architecture()
    
    # Export data
    analyzer.export_to_json("visualizations/architecture_data.json")
    
    # Create visualization
    fig = analyzer.create_plotly_visualization()
    fig.write_html("visualizations/architecture_plot.html")
    
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {architecture_data['total_params']:,}")
    print(f"  Hidden Size: {architecture_data['hidden_size']}")
    print(f"  Layers: {architecture_data['num_hidden_layers']}")
    print(f"  Attention Heads: {architecture_data['num_attention_heads']}")
    print(f"  Model Size: {architecture_data['memory_info']['model_size_gb']:.2f} GB")
    
    return architecture_data


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/gemma-3-4b"
    analyze_model(model_path)