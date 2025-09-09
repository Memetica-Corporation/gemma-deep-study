"""
Attention Pattern Visualization for Gemma-3
Focuses on understanding the 5:1 local/global attention pattern
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json


@dataclass
class AttentionPattern:
    """Container for attention pattern data"""
    layer_idx: int
    head_idx: int
    attention_weights: torch.Tensor
    is_local: bool
    is_global: bool
    tokens: List[str]


class GemmaAttentionAnalyzer:
    """Analyze Gemma-3's unique attention patterns"""
    
    def __init__(self, model_path: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            output_attentions=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.attention_patterns = []
        self.layer_types = self._identify_layer_types()
        
    def _identify_layer_types(self) -> Dict[int, str]:
        """Identify which layers are local vs global attention"""
        layer_types = {}
        
        # Gemma-3 pattern: 5 local, 1 global, repeat
        for i in range(len(self.model.model.layers)):
            if (i + 1) % 6 == 0:
                layer_types[i] = "global"
            else:
                layer_types[i] = "local"
                
        return layer_types
        
    def extract_attention_patterns(
        self,
        text: str,
        layers_to_analyze: Optional[List[int]] = None
    ) -> List[AttentionPattern]:
        """Extract attention patterns from specified layers"""
        
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Forward pass with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        attentions = outputs.attentions  # Tuple of attention matrices
        
        patterns = []
        layers_to_analyze = layers_to_analyze or range(len(attentions))
        
        for layer_idx in layers_to_analyze:
            if layer_idx >= len(attentions):
                continue
                
            attention = attentions[layer_idx]  # Shape: (batch, heads, seq_len, seq_len)
            layer_type = self.layer_types.get(layer_idx, "unknown")
            
            # Average across heads for visualization
            avg_attention = attention.mean(dim=1).squeeze(0)
            
            # Store pattern
            pattern = AttentionPattern(
                layer_idx=layer_idx,
                head_idx=-1,  # -1 indicates averaged
                attention_weights=avg_attention,
                is_local=(layer_type == "local"),
                is_global=(layer_type == "global"),
                tokens=tokens
            )
            patterns.append(pattern)
            
            # Also store individual head patterns for detailed analysis
            for head_idx in range(attention.shape[1]):
                head_pattern = AttentionPattern(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    attention_weights=attention[0, head_idx],
                    is_local=(layer_type == "local"),
                    is_global=(layer_type == "global"),
                    tokens=tokens
                )
                patterns.append(head_pattern)
                
        self.attention_patterns = patterns
        return patterns
        
    def visualize_attention_matrix(
        self,
        pattern: AttentionPattern,
        save_path: Optional[str] = None
    ):
        """Visualize single attention pattern as heatmap"""
        
        plt.figure(figsize=(12, 10))
        
        # Convert to numpy
        attention = pattern.attention_weights.cpu().numpy()
        
        # Create heatmap
        sns.heatmap(
            attention,
            xticklabels=pattern.tokens,
            yticklabels=pattern.tokens,
            cmap='Blues' if pattern.is_local else 'Reds',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        layer_type = "Local" if pattern.is_local else "Global"
        head_info = f"Head {pattern.head_idx}" if pattern.head_idx >= 0 else "Averaged"
        plt.title(f"Layer {pattern.layer_idx} ({layer_type}) - {head_info}")
        plt.xlabel("Keys (To)")
        plt.ylabel("Queries (From)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        
    def analyze_local_vs_global_patterns(self) -> Dict:
        """Analyze differences between local and global attention patterns"""
        
        local_patterns = [p for p in self.attention_patterns if p.is_local and p.head_idx == -1]
        global_patterns = [p for p in self.attention_patterns if p.is_global and p.head_idx == -1]
        
        if not local_patterns or not global_patterns:
            return {}
            
        analysis = {
            'local': {},
            'global': {}
        }
        
        # Analyze local patterns
        for pattern in local_patterns:
            weights = pattern.attention_weights.cpu().numpy()
            
            # Calculate attention spread (entropy)
            entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()
            
            # Calculate effective attention window
            seq_len = weights.shape[0]
            avg_distance = 0
            for i in range(seq_len):
                distances = np.abs(np.arange(seq_len) - i)
                avg_distance += np.sum(weights[i] * distances)
            avg_distance /= seq_len
            
            if 'entropy' not in analysis['local']:
                analysis['local']['entropy'] = []
                analysis['local']['avg_distance'] = []
                
            analysis['local']['entropy'].append(entropy)
            analysis['local']['avg_distance'].append(avg_distance)
            
        # Analyze global patterns
        for pattern in global_patterns:
            weights = pattern.attention_weights.cpu().numpy()
            
            entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()
            
            seq_len = weights.shape[0]
            avg_distance = 0
            for i in range(seq_len):
                distances = np.abs(np.arange(seq_len) - i)
                avg_distance += np.sum(weights[i] * distances)
            avg_distance /= seq_len
            
            if 'entropy' not in analysis['global']:
                analysis['global']['entropy'] = []
                analysis['global']['avg_distance'] = []
                
            analysis['global']['entropy'].append(entropy)
            analysis['global']['avg_distance'].append(avg_distance)
            
        # Calculate statistics
        for pattern_type in ['local', 'global']:
            if analysis[pattern_type]:
                analysis[pattern_type]['mean_entropy'] = np.mean(analysis[pattern_type]['entropy'])
                analysis[pattern_type]['mean_distance'] = np.mean(analysis[pattern_type]['avg_distance'])
                
        return analysis
        
    def visualize_layer_comparison(self, save_path: Optional[str] = None):
        """Compare local vs global attention patterns across layers"""
        
        analysis = self.analyze_local_vs_global_patterns()
        
        if not analysis:
            print("No patterns to analyze")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Entropy comparison
        local_entropies = analysis['local'].get('entropy', [])
        global_entropies = analysis['global'].get('entropy', [])
        
        axes[0, 0].boxplot([local_entropies, global_entropies], 
                          labels=['Local', 'Global'])
        axes[0, 0].set_ylabel('Attention Entropy')
        axes[0, 0].set_title('Attention Focus Comparison')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average attention distance
        local_distances = analysis['local'].get('avg_distance', [])
        global_distances = analysis['global'].get('avg_distance', [])
        
        axes[0, 1].boxplot([local_distances, global_distances],
                          labels=['Local', 'Global'])
        axes[0, 1].set_ylabel('Average Attention Distance')
        axes[0, 1].set_title('Attention Range Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Layer-wise entropy
        layers = []
        entropies = []
        colors = []
        
        for pattern in self.attention_patterns:
            if pattern.head_idx == -1:  # Only averaged patterns
                layers.append(pattern.layer_idx)
                weights = pattern.attention_weights.cpu().numpy()
                entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1).mean()
                entropies.append(entropy)
                colors.append('blue' if pattern.is_local else 'red')
                
        axes[1, 0].scatter(layers, entropies, c=colors, alpha=0.7, s=50)
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Attention Entropy')
        axes[1, 0].set_title('Entropy Across Layers')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Local'),
            Patch(facecolor='red', label='Global')
        ]
        axes[1, 0].legend(handles=legend_elements)
        
        # Plot 4: Statistics summary
        axes[1, 1].axis('off')
        
        stats_text = "Attention Pattern Statistics\n" + "="*30 + "\n\n"
        stats_text += "Local Layers:\n"
        stats_text += f"  Mean Entropy: {analysis['local'].get('mean_entropy', 0):.3f}\n"
        stats_text += f"  Mean Distance: {analysis['local'].get('mean_distance', 0):.2f}\n\n"
        stats_text += "Global Layers:\n"
        stats_text += f"  Mean Entropy: {analysis['global'].get('mean_entropy', 0):.3f}\n"
        stats_text += f"  Mean Distance: {analysis['global'].get('mean_distance', 0):.2f}\n"
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                       family='monospace', verticalalignment='center')
        
        plt.suptitle('Gemma-3 Local vs Global Attention Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        
    def create_interactive_visualization(self, pattern: AttentionPattern) -> go.Figure:
        """Create interactive Plotly visualization"""
        
        attention = pattern.attention_weights.cpu().numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=attention,
            x=pattern.tokens,
            y=pattern.tokens,
            colorscale='Blues' if pattern.is_local else 'Reds',
            hovertemplate='From: %{y}<br>To: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))
        
        layer_type = "Local" if pattern.is_local else "Global"
        head_info = f"Head {pattern.head_idx}" if pattern.head_idx >= 0 else "Averaged"
        
        fig.update_layout(
            title=f"Layer {pattern.layer_idx} ({layer_type}) - {head_info}",
            xaxis_title="Keys (To)",
            yaxis_title="Queries (From)",
            width=800,
            height=700
        )
        
        return fig
        
    def analyze_attention_flow(self, text: str) -> Dict:
        """Analyze how attention flows through the network"""
        
        patterns = self.extract_attention_patterns(text)
        
        flow_analysis = {
            'layer_connectivity': [],
            'information_bottlenecks': [],
            'attention_highways': []
        }
        
        # Track how attention changes across layers
        prev_attention = None
        
        for pattern in patterns:
            if pattern.head_idx != -1:  # Skip individual heads
                continue
                
            attention = pattern.attention_weights.cpu().numpy()
            
            if prev_attention is not None:
                # Calculate attention flow change
                flow_change = np.abs(attention - prev_attention).mean()
                flow_analysis['layer_connectivity'].append({
                    'from_layer': pattern.layer_idx - 1,
                    'to_layer': pattern.layer_idx,
                    'flow_change': float(flow_change),
                    'is_transition': self.layer_types.get(pattern.layer_idx - 1) != self.layer_types.get(pattern.layer_idx)
                })
                
            # Identify bottlenecks (positions with concentrated attention)
            max_attention_per_pos = attention.max(axis=0)
            bottleneck_positions = np.where(max_attention_per_pos > 0.5)[0]
            
            if len(bottleneck_positions) > 0:
                flow_analysis['information_bottlenecks'].append({
                    'layer': pattern.layer_idx,
                    'positions': bottleneck_positions.tolist(),
                    'tokens': [pattern.tokens[i] for i in bottleneck_positions],
                    'max_weights': max_attention_per_pos[bottleneck_positions].tolist()
                })
                
            # Identify highways (consistent attention paths)
            diagonal_strength = np.mean(np.diag(attention))
            if diagonal_strength > 0.3:
                flow_analysis['attention_highways'].append({
                    'layer': pattern.layer_idx,
                    'type': 'diagonal',
                    'strength': float(diagonal_strength)
                })
                
            prev_attention = attention
            
        return flow_analysis
        
    def save_analysis(self, analysis: Dict, filepath: str):
        """Save analysis results to JSON"""
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
            
        serializable_analysis = convert_to_serializable(analysis)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
            
        print(f"Analysis saved to {filepath}")


def run_attention_analysis(model_path: str, sample_text: str = None):
    """Run complete attention analysis"""
    
    if sample_text is None:
        sample_text = """The attention mechanism in transformers allows the model to focus on 
        different parts of the input sequence when processing each token. This selective 
        focus enables the model to capture long-range dependencies and relationships."""
    
    print("Initializing Gemma Attention Analyzer...")
    analyzer = GemmaAttentionAnalyzer(model_path)
    
    print(f"Analyzing text: {sample_text[:50]}...")
    patterns = analyzer.extract_attention_patterns(sample_text, layers_to_analyze=list(range(12)))
    
    print(f"Extracted {len(patterns)} attention patterns")
    
    # Visualize sample patterns
    for pattern in patterns[:3]:  # First 3 averaged patterns
        if pattern.head_idx == -1:
            analyzer.visualize_attention_matrix(
                pattern,
                save_path=f"visualizations/attention_layer_{pattern.layer_idx}.png"
            )
    
    # Compare local vs global
    print("\nComparing local vs global attention patterns...")
    analyzer.visualize_layer_comparison(
        save_path="visualizations/local_vs_global_comparison.png"
    )
    
    # Analyze attention flow
    print("\nAnalyzing attention flow...")
    flow_analysis = analyzer.analyze_attention_flow(sample_text)
    
    # Save results
    Path("visualizations").mkdir(exist_ok=True)
    analyzer.save_analysis(flow_analysis, "visualizations/attention_flow_analysis.json")
    
    print("\nAnalysis complete! Check visualizations/ directory for results.")
    
    return analyzer, patterns, flow_analysis


if __name__ == "__main__":
    print("Attention Visualizer initialized")
    print("This module analyzes Gemma-3's 5:1 local/global attention patterns")
    print("Import and use with your Gemma model for attention analysis")