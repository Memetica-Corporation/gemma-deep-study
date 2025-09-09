"""
State-of-the-art Visualization Tools for Model Analysis
Interactive, publication-quality visualizations for deep understanding
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class InteractiveVisualizer:
    """
    Create interactive visualizations for model analysis
    Publication-quality, web-based interactive plots
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figures = {}
        
    def visualize_attention_patterns(self, attention_weights: torch.Tensor,
                                    layer_name: str = "attention",
                                    tokens: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive attention heatmap with advanced features
        """
        
        # Convert to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        # Handle multi-head attention
        if attention_weights.ndim == 4:
            # Average over heads for main view
            avg_attention = attention_weights.mean(axis=1)
        else:
            avg_attention = attention_weights
            
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Attention Pattern',
                'Attention Entropy by Position',
                'Head Diversity Analysis',
                'Attention Distance Distribution'
            ),
            specs=[[{'type': 'heatmap'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'violin'}]]
        )
        
        # Main attention heatmap
        fig.add_trace(
            go.Heatmap(
                z=avg_attention[0] if avg_attention.ndim > 2 else avg_attention,
                x=tokens if tokens else list(range(avg_attention.shape[-1])),
                y=tokens if tokens else list(range(avg_attention.shape[-2])),
                colorscale='Viridis',
                text=np.round(avg_attention[0] if avg_attention.ndim > 2 else avg_attention, 3),
                hovertemplate='From: %{y}<br>To: %{x}<br>Weight: %{text}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Attention entropy
        entropy = -np.sum(avg_attention * np.log(avg_attention + 1e-10), axis=-1)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(entropy[0]))) if entropy.ndim > 1 else list(range(len(entropy))),
                y=entropy[0] if entropy.ndim > 1 else entropy,
                mode='lines+markers',
                name='Entropy',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Head diversity (if multi-head)
        if attention_weights.ndim == 4:
            num_heads = attention_weights.shape[1]
            head_similarities = []
            
            for i in range(num_heads):
                for j in range(i+1, num_heads):
                    similarity = np.corrcoef(
                        attention_weights[0, i].flatten(),
                        attention_weights[0, j].flatten()
                    )[0, 1]
                    head_similarities.append(similarity)
                    
            fig.add_trace(
                go.Bar(
                    x=list(range(len(head_similarities))),
                    y=head_similarities,
                    name='Head Pair Similarity',
                    marker_color='lightblue'
                ),
                row=2, col=1
            )
        
        # Attention distance distribution
        seq_len = avg_attention.shape[-1]
        distances = []
        weights = []
        
        for i in range(seq_len):
            for j in range(seq_len):
                distances.append(abs(i - j))
                weights.append(avg_attention[0, i, j] if avg_attention.ndim > 2 else avg_attention[i, j])
                
        fig.add_trace(
            go.Violin(
                x=distances,
                y=weights,
                name='Distance vs Weight',
                box_visible=True,
                meanline_visible=True
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Attention Analysis: {layer_name}",
            height=800,
            showlegend=False,
            hovermode='closest'
        )
        
        # Save figure
        output_path = self.output_dir / f"attention_{layer_name}.html"
        fig.write_html(str(output_path))
        
        self.figures[f"attention_{layer_name}"] = fig
        return fig
    
    def visualize_layer_dynamics(self, layer_stats: List[Dict],
                                model_name: str = "model") -> go.Figure:
        """
        Visualize layer-wise statistics and dynamics
        """
        
        # Prepare data
        layers = []
        metrics = {
            'gradient_norm': [],
            'activation_sparsity': [],
            'attention_entropy': [],
            'effective_rank': []
        }
        
        for stat in layer_stats:
            layers.append(f"Layer {stat.get('layer_idx', 0)}")
            metrics['gradient_norm'].append(stat.get('gradient_norm', 0))
            metrics['activation_sparsity'].append(stat.get('activation_sparsity', 0))
            metrics['attention_entropy'].append(stat.get('attention_entropy', 0))
            
            # Compute effective rank from singular values
            if 'singular_values' in stat and len(stat['singular_values']) > 0:
                s = np.array(stat['singular_values'])
                s_normalized = s / (s.sum() + 1e-10)
                entropy = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
                effective_rank = np.exp(entropy)
                metrics['effective_rank'].append(effective_rank)
            else:
                metrics['effective_rank'].append(0)
                
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gradient Flow',
                'Activation Sparsity',
                'Attention Entropy',
                'Effective Rank'
            ),
            specs=[[{'secondary_y': False}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': False}]]
        )
        
        # Gradient norms
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=metrics['gradient_norm'],
                mode='lines+markers',
                name='Gradient Norm',
                line=dict(color='blue', width=3),
                marker=dict(size=10),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        # Activation sparsity
        fig.add_trace(
            go.Bar(
                x=layers,
                y=metrics['activation_sparsity'],
                name='Sparsity',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Attention entropy
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=metrics['attention_entropy'],
                mode='lines+markers',
                name='Entropy',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ),
            row=2, col=1
        )
        
        # Effective rank
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=metrics['effective_rank'],
                mode='lines+markers',
                name='Effective Rank',
                line=dict(color='purple', width=2),
                marker=dict(size=8),
                fill='tonexty'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Layer Dynamics Analysis: {model_name}",
            height=800,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Gradient Norm", row=1, col=1)
        fig.update_yaxes(title_text="Sparsity", row=1, col=2)
        fig.update_yaxes(title_text="Entropy", row=2, col=1)
        fig.update_yaxes(title_text="Rank", row=2, col=2)
        
        # Save
        output_path = self.output_dir / f"layer_dynamics_{model_name}.html"
        fig.write_html(str(output_path))
        
        self.figures[f"layer_dynamics_{model_name}"] = fig
        return fig
    
    def visualize_capacity_analysis(self, capacity_data: Dict,
                                   dataset_sizes: List[int]) -> go.Figure:
        """
        Visualize model capacity and memorization analysis
        Based on "How much do language models memorize?" paper
        """
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Capacity vs Dataset Size',
                'Memorization to Generalization Transition',
                'Double Descent Curve',
                'Layer-wise Capacity Distribution'
            )
        )
        
        # Capacity curve
        model_capacity = capacity_data.get('total_capacity', 0)
        dataset_bits = [size * 8 for size in dataset_sizes]  # Convert to bits
        
        fig.add_trace(
            go.Scatter(
                x=dataset_sizes,
                y=[model_capacity] * len(dataset_sizes),
                mode='lines',
                name='Model Capacity',
                line=dict(color='blue', width=3, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dataset_sizes,
                y=dataset_bits,
                mode='lines',
                name='Dataset Size (bits)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Memorization vs Generalization
        transition_point = capacity_data.get('transition_point', 0)
        memorization_curve = [
            1.0 if size < transition_point else np.exp(-(size - transition_point) / transition_point)
            for size in dataset_sizes
        ]
        generalization_curve = [
            0.0 if size < transition_point else 1 - np.exp(-(size - transition_point) / transition_point)
            for size in dataset_sizes
        ]
        
        fig.add_trace(
            go.Scatter(
                x=dataset_sizes,
                y=memorization_curve,
                mode='lines',
                name='Memorization',
                line=dict(color='orange', width=2),
                fill='tozeroy'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=dataset_sizes,
                y=generalization_curve,
                mode='lines',
                name='Generalization',
                line=dict(color='green', width=2),
                fill='tozeroy'
            ),
            row=1, col=2
        )
        
        # Double descent curve
        loss_curve = []
        for size in dataset_sizes:
            if size < transition_point * 0.8:
                # Underfitting
                loss = 1.0 - (size / (transition_point * 0.8)) * 0.3
            elif size < transition_point * 1.2:
                # Interpolation threshold
                loss = 0.7 + 0.3 * ((size - transition_point * 0.8) / (transition_point * 0.4))
            else:
                # Over-parameterized regime
                loss = 1.0 - 0.4 * (1 - np.exp(-(size - transition_point * 1.2) / transition_point))
                
            loss_curve.append(loss)
            
        fig.add_trace(
            go.Scatter(
                x=dataset_sizes,
                y=loss_curve,
                mode='lines',
                name='Test Loss',
                line=dict(color='purple', width=3)
            ),
            row=2, col=1
        )
        
        # Layer-wise capacity
        layer_capacities = capacity_data.get('layer_capacities', {})
        if layer_capacities:
            layers = list(layer_capacities.keys())
            capacities = list(layer_capacities.values())
            
            fig.add_trace(
                go.Bar(
                    x=layers,
                    y=capacities,
                    name='Layer Capacity',
                    marker_color='teal'
                ),
                row=2, col=2
            )
            
        # Update layout
        fig.update_layout(
            title="Model Capacity and Memorization Analysis",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Dataset Size", row=1, col=1)
        fig.update_xaxes(title_text="Dataset Size", row=1, col=2)
        fig.update_xaxes(title_text="Dataset Size", row=2, col=1)
        fig.update_xaxes(title_text="Layer", row=2, col=2)
        
        fig.update_yaxes(title_text="Bits", row=1, col=1, type="log")
        fig.update_yaxes(title_text="Proportion", row=1, col=2)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        fig.update_yaxes(title_text="Capacity (bits)", row=2, col=2)
        
        # Save
        output_path = self.output_dir / "capacity_analysis.html"
        fig.write_html(str(output_path))
        
        self.figures["capacity_analysis"] = fig
        return fig
    
    def visualize_lora_dynamics(self, lora_stats: Dict) -> go.Figure:
        """
        Visualize LoRA training dynamics including rank evolution
        """
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Rank Evolution',
                'Subspace Trajectory',
                'Parameter Update Distribution',
                'Layer-wise LoRA Impact',
                'Gradient Alignment',
                'Effective Rank vs Iteration'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter3d'}, {'type': 'violin'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Rank evolution
        iterations = lora_stats.get('iterations', [])
        ranks = lora_stats.get('ranks', [])
        
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=ranks,
                mode='lines+markers',
                name='Rank',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Subspace trajectory (3D)
        if 'subspace_coords' in lora_stats:
            coords = lora_stats['subspace_coords']
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0] if len(coords) > 0 else [],
                    y=coords[:, 1] if len(coords) > 0 else [],
                    z=coords[:, 2] if len(coords) > 0 else [],
                    mode='lines+markers',
                    name='Trajectory',
                    line=dict(color=iterations, colorscale='Viridis', width=4),
                    marker=dict(size=4)
                ),
                row=1, col=2
            )
            
        # Parameter update distribution
        if 'param_updates' in lora_stats:
            updates = lora_stats['param_updates']
            fig.add_trace(
                go.Violin(
                    y=updates.flatten() if hasattr(updates, 'flatten') else updates,
                    name='Updates',
                    box_visible=True,
                    meanline_visible=True
                ),
                row=1, col=3
            )
            
        # Layer-wise impact heatmap
        if 'layer_impacts' in lora_stats:
            impacts = lora_stats['layer_impacts']
            fig.add_trace(
                go.Heatmap(
                    z=impacts,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=1
            )
            
        # Gradient alignment
        if 'gradient_alignment' in lora_stats:
            alignment = lora_stats['gradient_alignment']
            fig.add_trace(
                go.Scatter(
                    x=iterations[:len(alignment)],
                    y=alignment,
                    mode='lines',
                    name='Alignment',
                    line=dict(color='green', width=2)
                ),
                row=2, col=2
            )
            
        # Effective rank over time
        if 'effective_ranks' in lora_stats:
            eff_ranks = lora_stats['effective_ranks']
            fig.add_trace(
                go.Scatter(
                    x=iterations[:len(eff_ranks)],
                    y=eff_ranks,
                    mode='lines+markers',
                    name='Effective Rank',
                    line=dict(color='purple', width=2),
                    fill='tozeroy'
                ),
                row=2, col=3
            )
            
        # Update layout
        fig.update_layout(
            title="LoRA Training Dynamics",
            height=900,
            showlegend=False
        )
        
        # Save
        output_path = self.output_dir / "lora_dynamics.html"
        fig.write_html(str(output_path))
        
        self.figures["lora_dynamics"] = fig
        return fig
    
    def create_architecture_graph(self, architecture_map: Dict) -> go.Figure:
        """
        Create an interactive graph visualization of model architecture
        """
        
        # Build node and edge lists
        nodes = []
        edges = []
        node_sizes = []
        node_colors = []
        
        # Add embedding nodes
        for name, info in architecture_map.get('embeddings', {}).items():
            nodes.append(name)
            node_sizes.append(np.log(info['params'] + 1) * 5)
            node_colors.append('blue')
            
        # Add attention nodes
        for layer in architecture_map.get('attention_layers', []):
            nodes.append(layer['name'])
            node_sizes.append(np.log(layer['params'] + 1) * 5)
            color = 'red' if layer['type'] == 'global' else 'orange'
            node_colors.append(color)
            
        # Add MLP nodes
        for layer in architecture_map.get('mlp_layers', []):
            nodes.append(layer['name'])
            node_sizes.append(np.log(layer['params'] + 1) * 5)
            node_colors.append('green')
            
        # Create edges (simplified sequential connections)
        for i in range(len(nodes) - 1):
            edges.append((i, i + 1))
            
        # Create network graph
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(range(len(nodes)))
        G.add_edges_from(edges)
        
        # Get positions using spring layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Extract positions
        x_nodes = [pos[i][0] for i in range(len(nodes))]
        y_nodes = [pos[i][1] for i in range(len(nodes))]
        
        # Create edge traces
        edge_trace = []
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='gray'),
                    hoverinfo='none'
                )
            )
            
        # Create node trace
        node_trace = go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            ),
            text=[n.split('.')[-1] for n in nodes],  # Show only last part of name
            textposition="top center",
            hovertext=nodes,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title="Model Architecture Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800
        )
        
        # Save
        output_path = self.output_dir / "architecture_graph.html"
        fig.write_html(str(output_path))
        
        self.figures["architecture_graph"] = fig
        return fig


class LayerDynamicsPlotter:
    """
    Advanced plotting for layer-wise dynamics and analysis
    """
    
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_gradient_flow(self, gradient_norms: Dict[str, float],
                          save_path: Optional[str] = None):
        """
        Visualize gradient flow through the network
        """
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Extract layer names and norms
        layers = list(gradient_norms.keys())
        norms = list(gradient_norms.values())
        
        # Main gradient plot
        ax1.bar(range(len(layers)), norms, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Gradient Norm', fontsize=12)
        ax1.set_title('Gradient Flow Through Network', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add threshold lines
        ax1.axhline(y=1e-7, color='red', linestyle='--', label='Vanishing threshold')
        ax1.axhline(y=100, color='orange', linestyle='--', label='Exploding threshold')
        ax1.legend()
        
        # Log scale plot
        ax2.semilogy(range(len(layers)), norms, 'o-', color='darkblue', linewidth=2)
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Gradient Norm (log scale)', fontsize=12)
        ax2.set_title('Gradient Flow (Log Scale)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_attention_statistics(self, attention_stats: List[Dict],
                                 save_path: Optional[str] = None):
        """
        Plot comprehensive attention statistics
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract statistics
        layers = []
        entropy = []
        sparsity = []
        locality = []
        diversity = []
        distance = []
        
        for stat in attention_stats:
            layers.append(stat.get('layer', 0))
            entropy.append(stat.get('entropy', 0))
            sparsity.append(stat.get('sparsity', 0))
            locality.append(stat.get('locality', 0))
            diversity.append(stat.get('head_diversity', 0))
            distance.append(stat.get('attention_distance', 0))
            
        # Entropy plot
        axes[0, 0].plot(layers, entropy, 'o-', color='red', linewidth=2)
        axes[0, 0].set_title('Attention Entropy')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Entropy')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Sparsity plot
        axes[0, 1].bar(layers, sparsity, color='green', alpha=0.7)
        axes[0, 1].set_title('Attention Sparsity')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Sparsity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Locality plot
        axes[0, 2].plot(layers, locality, 's-', color='blue', linewidth=2)
        axes[0, 2].set_title('Attention Locality')
        axes[0, 2].set_xlabel('Layer')
        axes[0, 2].set_ylabel('Locality Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Head diversity plot
        axes[1, 0].plot(layers, diversity, '^-', color='purple', linewidth=2)
        axes[1, 0].set_title('Head Diversity')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Diversity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Attention distance plot
        axes[1, 1].bar(layers, distance, color='orange', alpha=0.7)
        axes[1, 1].set_title('Mean Attention Distance')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Combined normalized plot
        axes[1, 2].plot(layers, entropy / np.max(entropy), label='Entropy', linewidth=2)
        axes[1, 2].plot(layers, sparsity / np.max(sparsity), label='Sparsity', linewidth=2)
        axes[1, 2].plot(layers, locality / np.max(locality), label='Locality', linewidth=2)
        axes[1, 2].plot(layers, diversity / np.max(diversity), label='Diversity', linewidth=2)
        axes[1, 2].set_title('Normalized Metrics')
        axes[1, 2].set_xlabel('Layer')
        axes[1, 2].set_ylabel('Normalized Value')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Attention Mechanism Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def plot_training_dynamics(self, training_stats: Dict,
                              save_path: Optional[str] = None):
        """
        Visualize training dynamics over time
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curve
        if 'loss' in training_stats:
            axes[0, 0].plot(training_stats['loss'], linewidth=2, color='blue')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            
        # Learning rate schedule
        if 'learning_rate' in training_stats:
            axes[0, 1].plot(training_stats['learning_rate'], linewidth=2, color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
            
        # Gradient norm
        if 'gradient_norm' in training_stats:
            axes[1, 0].plot(training_stats['gradient_norm'], linewidth=2, color='red')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Norm')
            axes[1, 0].grid(True, alpha=0.3)
            
        # Parameter update ratio
        if 'param_update_ratio' in training_stats:
            axes[1, 1].plot(training_stats['param_update_ratio'], linewidth=2, color='purple')
            axes[1, 1].set_title('Parameter Update Ratio')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Update/Param Ratio')
            axes[1, 1].grid(True, alpha=0.3)
            
        plt.suptitle('Training Dynamics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()