"""
Core Architecture Analysis Components
Deep introspection tools for Gemma-3 transformer architecture
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class LayerStatistics:
    """Comprehensive statistics for a single transformer layer"""
    layer_idx: int
    attention_entropy: float
    activation_sparsity: float
    gradient_norm: float
    weight_rank: int
    singular_values: np.ndarray
    neuron_activation_rate: np.ndarray
    attention_pattern_diversity: float
    local_vs_global_ratio: Optional[float] = None
    
    
class GemmaArchitecture:
    """
    Advanced architecture analyzer for Gemma-3 models
    Provides deep insights into model structure and behavior
    """
    
    def __init__(self, model: nn.Module, config: Optional[Dict] = None):
        self.model = model
        self.config = config or self._infer_config(model)
        self.layer_stats = []
        self.hooks = []
        self._setup_architecture_map()
        
    def _infer_config(self, model: nn.Module) -> Dict:
        """Infer model configuration from architecture"""
        config = {
            'hidden_size': None,
            'num_layers': 0,
            'num_heads': None,
            'vocab_size': None,
            'max_position_embeddings': None,
            'local_attention_window': 1024,
            'global_attention_layers': [],
        }
        
        # Analyze model structure
        for name, module in model.named_modules():
            if 'embed' in name.lower() and hasattr(module, 'weight'):
                if config['vocab_size'] is None:
                    config['vocab_size'] = module.weight.shape[0]
                    config['hidden_size'] = module.weight.shape[1]
            elif 'attention' in name.lower():
                config['num_layers'] += 1
                # Detect local vs global attention
                if hasattr(module, 'window_size'):
                    if module.window_size > config['local_attention_window']:
                        config['global_attention_layers'].append(config['num_layers'] - 1)
                        
        return config
    
    def _setup_architecture_map(self):
        """Create detailed map of model architecture"""
        self.architecture_map = {
            'embeddings': {},
            'attention_layers': [],
            'mlp_layers': [],
            'normalization_layers': [],
            'output_layers': {}
        }
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding):
                self.architecture_map['embeddings'][name] = {
                    'vocab_size': module.num_embeddings,
                    'embedding_dim': module.embedding_dim,
                    'params': module.weight.numel()
                }
            elif 'attention' in module.__class__.__name__.lower():
                layer_info = self._analyze_attention_layer(name, module)
                self.architecture_map['attention_layers'].append(layer_info)
            elif isinstance(module, (nn.Linear, nn.Conv1d)):
                if 'mlp' in name.lower() or 'feed' in name.lower():
                    self.architecture_map['mlp_layers'].append({
                        'name': name,
                        'in_features': module.in_features if hasattr(module, 'in_features') else module.in_channels,
                        'out_features': module.out_features if hasattr(module, 'out_features') else module.out_channels,
                        'params': sum(p.numel() for p in module.parameters())
                    })
                    
    def _analyze_attention_layer(self, name: str, module: nn.Module) -> Dict:
        """Deep analysis of attention layer structure"""
        info = {
            'name': name,
            'type': 'unknown',
            'num_heads': None,
            'head_dim': None,
            'window_size': None,
            'params': sum(p.numel() for p in module.parameters())
        }
        
        # Detect attention type
        if hasattr(module, 'window_size'):
            info['window_size'] = module.window_size
            info['type'] = 'local' if module.window_size <= 1024 else 'global'
        
        # Extract head configuration
        for sub_name, sub_module in module.named_modules():
            if isinstance(sub_module, nn.Linear):
                if 'q_proj' in sub_name or 'query' in sub_name:
                    out_dim = sub_module.out_features
                    if self.config['hidden_size']:
                        info['num_heads'] = out_dim // (self.config['hidden_size'] // 8)  # Estimate
                        info['head_dim'] = out_dim // info['num_heads']
                    break
                    
        return info
    
    def analyze_layer(self, layer_idx: int, inputs: torch.Tensor, 
                     outputs: torch.Tensor, gradients: Optional[torch.Tensor] = None) -> LayerStatistics:
        """Comprehensive analysis of a single layer"""
        
        # Compute attention entropy
        attention_entropy = self._compute_attention_entropy(outputs)
        
        # Measure activation sparsity
        activation_sparsity = (outputs.abs() < 1e-6).float().mean().item()
        
        # Gradient analysis
        grad_norm = 0.0
        if gradients is not None:
            grad_norm = gradients.norm(2).item()
        
        # Weight matrix analysis (SVD)
        weight_rank, singular_values = self._analyze_weight_rank(layer_idx)
        
        # Neuron activation patterns
        neuron_activation_rate = self._compute_neuron_activation_rate(outputs)
        
        # Attention pattern diversity
        attention_diversity = self._compute_attention_diversity(layer_idx, outputs)
        
        # Local vs global attention ratio
        local_global_ratio = None
        if layer_idx in self.config.get('global_attention_layers', []):
            local_global_ratio = self._compute_local_global_ratio(outputs)
        
        return LayerStatistics(
            layer_idx=layer_idx,
            attention_entropy=attention_entropy,
            activation_sparsity=activation_sparsity,
            gradient_norm=grad_norm,
            weight_rank=weight_rank,
            singular_values=singular_values,
            neuron_activation_rate=neuron_activation_rate,
            attention_pattern_diversity=attention_diversity,
            local_vs_global_ratio=local_global_ratio
        )
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Calculate entropy of attention distribution"""
        if attention_weights.dim() < 2:
            return 0.0
        
        # Normalize to probability distribution
        probs = torch.softmax(attention_weights.view(-1, attention_weights.shape[-1]), dim=-1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        return entropy.item()
    
    def _analyze_weight_rank(self, layer_idx: int) -> Tuple[int, np.ndarray]:
        """Analyze effective rank of weight matrices using SVD"""
        # Get weight matrix from layer
        layer = self._get_layer_by_idx(layer_idx)
        if layer is None or not hasattr(layer, 'weight'):
            return 0, np.array([])
        
        weight = layer.weight.detach().cpu().numpy()
        
        # Compute SVD
        _, s, _ = np.linalg.svd(weight, full_matrices=False)
        
        # Compute effective rank (99% energy threshold)
        cumsum = np.cumsum(s**2)
        total_energy = cumsum[-1]
        rank = np.argmax(cumsum >= 0.99 * total_energy) + 1
        
        return rank, s
    
    def _compute_neuron_activation_rate(self, activations: torch.Tensor) -> np.ndarray:
        """Compute activation rate for each neuron"""
        # Reshape to (batch * seq_len, hidden_dim)
        act_flat = activations.view(-1, activations.shape[-1])
        
        # Compute activation rate (non-zero activations)
        activation_rate = (act_flat.abs() > 1e-6).float().mean(dim=0)
        
        return activation_rate.cpu().numpy()
    
    def _compute_attention_diversity(self, layer_idx: int, outputs: torch.Tensor) -> float:
        """Measure diversity of attention patterns"""
        # Simplified diversity metric based on output variance
        variance = outputs.var(dim=-1).mean()
        return variance.item()
    
    def _compute_local_global_ratio(self, outputs: torch.Tensor) -> float:
        """Compute ratio of local vs global attention contribution"""
        # Placeholder for actual local/global decomposition
        # In practice, this would require access to attention weights
        return 0.5  # Default 50/50 split
    
    def _get_layer_by_idx(self, layer_idx: int) -> Optional[nn.Module]:
        """Get layer module by index"""
        layers = [m for n, m in self.model.named_modules() if 'layer' in n.lower()]
        if layer_idx < len(layers):
            return layers[layer_idx]
        return None
    
    def register_hooks(self):
        """Register forward and backward hooks for comprehensive analysis"""
        
        def forward_hook(module, inputs, outputs, layer_idx):
            # Store activation statistics
            stats = self.analyze_layer(layer_idx, inputs[0], outputs)
            self.layer_stats.append(stats)
            
        def backward_hook(module, grad_input, grad_output, layer_idx):
            # Update gradient statistics
            if self.layer_stats and len(self.layer_stats) > layer_idx:
                self.layer_stats[layer_idx].gradient_norm = grad_output[0].norm(2).item()
        
        # Register hooks on all layers
        for idx, (name, module) in enumerate(self.model.named_modules()):
            if 'layer' in name.lower():
                # Use partial to capture layer index
                from functools import partial
                fwd_hook = module.register_forward_hook(
                    partial(forward_hook, layer_idx=idx)
                )
                bwd_hook = module.register_full_backward_hook(
                    partial(backward_hook, layer_idx=idx)
                )
                self.hooks.extend([fwd_hook, bwd_hook])
                
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def get_capacity_estimate(self) -> float:
        """
        Estimate model capacity in bits per parameter
        Based on "How much do language models memorize?" paper
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Theoretical capacity ~ 3.6 bits per parameter for transformers
        theoretical_capacity = total_params * 3.6
        
        # Adjust based on architecture specifics
        adjustment_factor = 1.0
        
        # Reduce capacity for local attention layers (less expressive)
        num_local = len(self.architecture_map['attention_layers']) - len(
            self.config.get('global_attention_layers', [])
        )
        if num_local > 0:
            adjustment_factor *= 0.85
            
        # Increase for deeper models (more composition)
        depth_bonus = math.log(self.config['num_layers']) / 10
        adjustment_factor *= (1 + depth_bonus)
        
        return theoretical_capacity * adjustment_factor
    
    def compute_memorization_threshold(self, dataset_size: int) -> float:
        """
        Compute when model transitions from memorization to generalization
        Returns the data/parameter ratio at which grokking occurs
        """
        capacity = self.get_capacity_estimate()
        dataset_bits = dataset_size * 8  # Rough estimate
        
        # Grokking occurs when dataset exceeds capacity
        ratio = dataset_bits / capacity
        
        return ratio


class LayerAnalyzer:
    """
    Advanced layer-wise analysis tools
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_representations = defaultdict(list)
        self.attention_patterns = defaultdict(list)
        
    def extract_representations(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate representations from all layers"""
        representations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                representations[name] = output.detach()
            return hook
        
        hooks = []
        for name, module in self.model.named_modules():
            if any(key in name.lower() for key in ['layer', 'block']):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
            
        return representations
    
    def compute_representation_similarity(self, layer1: str, layer2: str) -> float:
        """Compute cosine similarity between layer representations"""
        if layer1 not in self.layer_representations or layer2 not in self.layer_representations:
            return 0.0
        
        rep1 = torch.cat(self.layer_representations[layer1], dim=0)
        rep2 = torch.cat(self.layer_representations[layer2], dim=0)
        
        # Flatten and normalize
        rep1_flat = rep1.view(rep1.shape[0], -1)
        rep2_flat = rep2.view(rep2.shape[0], -1)
        
        rep1_norm = torch.nn.functional.normalize(rep1_flat, p=2, dim=1)
        rep2_norm = torch.nn.functional.normalize(rep2_flat, p=2, dim=1)
        
        # Compute similarity
        similarity = (rep1_norm * rep2_norm).sum(dim=1).mean()
        
        return similarity.item()
    
    def analyze_gradient_flow(self, loss: torch.Tensor) -> Dict[str, float]:
        """Analyze gradient flow through layers"""
        gradient_norms = {}
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradient_norms[name] = param.grad.norm(2).item()
                
        return gradient_norms
    
    def compute_layer_importance(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute importance of each layer using gradient-based attribution
        """
        importance_scores = {}
        
        # Get baseline output
        baseline_output = self.model(inputs)
        baseline_loss = torch.nn.functional.cross_entropy(baseline_output, targets)
        
        for name, module in self.model.named_modules():
            if 'layer' not in name.lower():
                continue
                
            # Temporarily zero out layer
            original_weight = None
            if hasattr(module, 'weight'):
                original_weight = module.weight.data.clone()
                module.weight.data.zero_()
            
            # Compute perturbed loss
            perturbed_output = self.model(inputs)
            perturbed_loss = torch.nn.functional.cross_entropy(perturbed_output, targets)
            
            # Restore weights
            if original_weight is not None:
                module.weight.data = original_weight
            
            # Importance is the change in loss
            importance_scores[name] = abs(perturbed_loss.item() - baseline_loss.item())
            
        return importance_scores


class AttentionMechanism:
    """
    Specialized analyzer for Gemma-3's hybrid attention mechanism
    """
    
    def __init__(self, local_window: int = 1024, global_ratio: float = 0.2):
        self.local_window = local_window
        self.global_ratio = global_ratio
        self.attention_cache = []
        
    def analyze_attention_pattern(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weight patterns"""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        analysis = {
            'entropy': self._compute_entropy(attention_weights),
            'sparsity': self._compute_sparsity(attention_weights),
            'locality': self._measure_locality(attention_weights),
            'head_diversity': self._compute_head_diversity(attention_weights),
            'attention_distance': self._compute_mean_attention_distance(attention_weights)
        }
        
        return analysis
    
    def _compute_entropy(self, weights: torch.Tensor) -> float:
        """Compute attention entropy"""
        probs = weights.softmax(dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        return entropy.item()
    
    def _compute_sparsity(self, weights: torch.Tensor) -> float:
        """Measure attention sparsity"""
        threshold = 1.0 / weights.shape[-1]  # 1/seq_len threshold
        sparse_ratio = (weights < threshold).float().mean()
        return sparse_ratio.item()
    
    def _measure_locality(self, weights: torch.Tensor) -> float:
        """Measure how local the attention patterns are"""
        seq_len = weights.shape[-2]
        
        # Create distance matrix
        positions = torch.arange(seq_len, device=weights.device)
        distance_matrix = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
        
        # Weight by attention and compute mean distance
        weighted_distance = (weights * distance_matrix.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        mean_distance = weighted_distance.mean()
        
        # Normalize by sequence length
        locality_score = 1.0 - (mean_distance / seq_len).item()
        
        return locality_score
    
    def _compute_head_diversity(self, weights: torch.Tensor) -> float:
        """Measure diversity across attention heads"""
        num_heads = weights.shape[1]
        
        if num_heads == 1:
            return 0.0
        
        # Compute pairwise cosine similarity between heads
        weights_flat = weights.view(weights.shape[0], num_heads, -1)
        weights_norm = torch.nn.functional.normalize(weights_flat, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(weights_norm, weights_norm.transpose(-1, -2))
        
        # Average off-diagonal elements (excluding self-similarity)
        mask = 1 - torch.eye(num_heads, device=weights.device)
        avg_similarity = (similarity_matrix * mask).sum() / (num_heads * (num_heads - 1))
        
        # Diversity is inverse of similarity
        diversity = 1.0 - avg_similarity.item()
        
        return diversity
    
    def _compute_mean_attention_distance(self, weights: torch.Tensor) -> float:
        """Compute mean distance attended to"""
        seq_len = weights.shape[-1]
        
        positions = torch.arange(seq_len, device=weights.device).float()
        pos_from = positions.unsqueeze(1).expand(-1, seq_len)
        pos_to = positions.unsqueeze(0).expand(seq_len, -1)
        
        distances = (pos_to - pos_from).abs()
        
        # Weight by attention values
        mean_distance = (weights * distances).sum(dim=-1).mean()
        
        return mean_distance.item()
    
    def decompose_local_global(self, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose attention into local and global components
        """
        seq_len = attention_weights.shape[-1]
        
        # Create local attention mask
        local_mask = torch.zeros_like(attention_weights)
        for i in range(seq_len):
            start = max(0, i - self.local_window // 2)
            end = min(seq_len, i + self.local_window // 2 + 1)
            local_mask[..., i, start:end] = 1.0
        
        # Separate local and global attention
        local_attention = attention_weights * local_mask
        global_attention = attention_weights * (1 - local_mask)
        
        return local_attention, global_attention
    
    def optimize_attention_pattern(self, attention_weights: torch.Tensor, 
                                  importance_threshold: float = 0.01) -> torch.Tensor:
        """
        Optimize attention pattern by pruning low-importance connections
        """
        # Keep only attention weights above threshold
        mask = attention_weights > importance_threshold
        optimized = attention_weights * mask
        
        # Renormalize
        optimized = optimized / (optimized.sum(dim=-1, keepdim=True) + 1e-10)
        
        return optimized