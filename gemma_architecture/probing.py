"""
Advanced Probing Tools for Deep Model Analysis
Implements state-of-the-art mechanistic interpretability techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import einops
from scipy import stats
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ProbeResult:
    """Results from probing analysis"""
    layer_name: str
    probe_type: str
    accuracy: float
    feature_importance: np.ndarray
    learned_representations: torch.Tensor
    metadata: Dict[str, Any]


class ActivationProbe:
    """
    Advanced activation probing for understanding learned representations
    Implements techniques from mechanistic interpretability research
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.activation_cache = {}
        self.probe_models = {}
        self.hooks = []
        
    def register_activation_hooks(self, layer_patterns: List[str]):
        """Register hooks to capture activations from specified layers"""
        
        def make_hook(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    output = output[0]
                self.activation_cache[name] = output.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in layer_patterns):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)
                
    def clear_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache = {}
        
    def train_linear_probe(self, layer_name: str, labels: torch.Tensor, 
                          probe_dim: Optional[int] = None) -> ProbeResult:
        """Train a linear probe on layer activations"""
        
        if layer_name not in self.activation_cache:
            raise ValueError(f"No activations cached for layer {layer_name}")
            
        activations = self.activation_cache[layer_name]
        
        # Flatten activations if needed
        if activations.dim() > 2:
            batch_size = activations.shape[0]
            activations = activations.view(batch_size, -1)
            
        # Optionally reduce dimensionality
        if probe_dim and probe_dim < activations.shape[1]:
            pca = PCA(n_components=probe_dim)
            activations_np = activations.numpy()
            activations = torch.from_numpy(pca.fit_transform(activations_np))
            
        # Create and train probe
        probe = nn.Linear(activations.shape[1], labels.max().item() + 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
        
        # Training loop
        probe.train()
        for epoch in range(100):
            logits = probe(activations)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Evaluate probe
        probe.eval()
        with torch.no_grad():
            logits = probe(activations)
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            
        # Extract feature importance
        feature_importance = probe.weight.abs().mean(dim=0).numpy()
        
        # Store probe for later use
        self.probe_models[layer_name] = probe
        
        return ProbeResult(
            layer_name=layer_name,
            probe_type='linear',
            accuracy=accuracy,
            feature_importance=feature_importance,
            learned_representations=probe.weight.detach(),
            metadata={'num_features': activations.shape[1]}
        )
        
    def find_interpretable_neurons(self, layer_name: str, 
                                  threshold: float = 0.95) -> List[int]:
        """
        Find neurons that respond to interpretable concepts
        Uses activation maximization and statistical analysis
        """
        
        if layer_name not in self.activation_cache:
            return []
            
        activations = self.activation_cache[layer_name]
        
        # Flatten if needed
        if activations.dim() > 2:
            activations = activations.view(activations.shape[0], -1)
            
        interpretable_neurons = []
        
        for neuron_idx in range(activations.shape[1]):
            neuron_acts = activations[:, neuron_idx]
            
            # Check for sparsity (interpretable neurons are often sparse)
            sparsity = (neuron_acts == 0).float().mean()
            
            # Check for bimodality (often indicates feature detection)
            if neuron_acts.std() > 0:
                # Normalize for statistical test
                normalized = (neuron_acts - neuron_acts.mean()) / neuron_acts.std()
                _, p_value = stats.normaltest(normalized.numpy())
                
                # Non-normal distribution suggests interpretability
                if p_value < 0.05 and sparsity > 0.5:
                    interpretable_neurons.append(neuron_idx)
                    
        return interpretable_neurons
    
    def decompose_representations(self, layer_name: str, 
                                 method: str = 'pca') -> Dict[str, np.ndarray]:
        """
        Decompose layer representations into interpretable components
        Supports PCA, ICA, and sparse coding
        """
        
        if layer_name not in self.activation_cache:
            raise ValueError(f"No activations for layer {layer_name}")
            
        activations = self.activation_cache[layer_name]
        
        # Flatten
        if activations.dim() > 2:
            batch_size = activations.shape[0]
            activations = activations.view(batch_size, -1)
            
        activations_np = activations.numpy()
        
        results = {}
        
        if method in ['pca', 'all']:
            # PCA decomposition
            pca = PCA(n_components=min(50, activations_np.shape[1]))
            components_pca = pca.fit_transform(activations_np)
            results['pca_components'] = components_pca
            results['pca_explained_variance'] = pca.explained_variance_ratio_
            
        if method in ['ica', 'all']:
            # ICA decomposition
            ica = FastICA(n_components=min(20, activations_np.shape[1]))
            components_ica = ica.fit_transform(activations_np)
            results['ica_components'] = components_ica
            results['ica_mixing_matrix'] = ica.mixing_
            
        if method in ['sparse', 'all']:
            # Sparse coding (simplified version)
            from sklearn.decomposition import DictionaryLearning
            dict_learning = DictionaryLearning(n_components=min(30, activations_np.shape[1]))
            sparse_codes = dict_learning.fit_transform(activations_np)
            results['sparse_codes'] = sparse_codes
            results['dictionary'] = dict_learning.components_
            
        return results
    
    def compute_neuron_feature_correlation(self, layer_name: str, 
                                          features: torch.Tensor) -> np.ndarray:
        """
        Compute correlation between neurons and input features
        Helps identify what each neuron encodes
        """
        
        if layer_name not in self.activation_cache:
            raise ValueError(f"No activations for layer {layer_name}")
            
        activations = self.activation_cache[layer_name]
        
        # Flatten if needed
        if activations.dim() > 2:
            activations = activations.view(activations.shape[0], -1)
        if features.dim() > 2:
            features = features.view(features.shape[0], -1)
            
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(
            activations.T.numpy(),
            features.T.numpy()
        )
        
        # Extract neuron-feature correlations
        n_neurons = activations.shape[1]
        neuron_feature_corr = correlation_matrix[:n_neurons, n_neurons:]
        
        return neuron_feature_corr


class GradientTracker:
    """
    Advanced gradient analysis for understanding learning dynamics
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradient_history = defaultdict(list)
        self.gradient_stats = {}
        
    def track_gradients(self, loss: torch.Tensor):
        """Track gradients for all parameters"""
        
        # Compute gradients
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        
        # Store gradients
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().clone()
                self.gradient_history[name].append(grad)
                
    def compute_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute comprehensive gradient statistics"""
        
        stats = {}
        
        for name, grad_list in self.gradient_history.items():
            if not grad_list:
                continue
                
            # Stack gradients
            grads = torch.stack(grad_list)
            
            stats[name] = {
                'mean_norm': grads.norm(2, dim=tuple(range(1, grads.dim()))).mean().item(),
                'std_norm': grads.norm(2, dim=tuple(range(1, grads.dim()))).std().item(),
                'max_norm': grads.norm(2, dim=tuple(range(1, grads.dim()))).max().item(),
                'min_norm': grads.norm(2, dim=tuple(range(1, grads.dim()))).min().item(),
                'gradient_variance': grads.var().item(),
                'gradient_snr': (grads.mean() / (grads.std() + 1e-8)).abs().item()
            }
            
        self.gradient_stats = stats
        return stats
    
    def detect_gradient_issues(self) -> Dict[str, List[str]]:
        """Detect common gradient problems"""
        
        issues = {
            'vanishing_gradients': [],
            'exploding_gradients': [],
            'dead_neurons': [],
            'oscillating_gradients': []
        }
        
        for name, grad_list in self.gradient_history.items():
            if len(grad_list) < 2:
                continue
                
            grads = torch.stack(grad_list)
            grad_norms = grads.norm(2, dim=tuple(range(1, grads.dim())))
            
            # Check for vanishing gradients
            if grad_norms.mean() < 1e-7:
                issues['vanishing_gradients'].append(name)
                
            # Check for exploding gradients  
            if grad_norms.mean() > 100:
                issues['exploding_gradients'].append(name)
                
            # Check for dead neurons (consistently zero gradients)
            if (grads == 0).all():
                issues['dead_neurons'].append(name)
                
            # Check for oscillating gradients
            if len(grad_list) > 10:
                # Compute autocorrelation
                grad_flat = grads.view(len(grad_list), -1)
                mean_grad = grad_flat.mean(dim=0)
                centered = grad_flat - mean_grad
                
                autocorr = []
                for lag in range(1, min(5, len(grad_list))):
                    corr = (centered[:-lag] * centered[lag:]).mean()
                    autocorr.append(corr.item())
                    
                # Negative autocorrelation indicates oscillation
                if autocorr and np.mean(autocorr) < -0.5:
                    issues['oscillating_gradients'].append(name)
                    
        return issues
    
    def compute_gradient_alignment(self, layer1: str, layer2: str) -> float:
        """
        Compute alignment between gradients of different layers
        High alignment suggests coupled learning
        """
        
        if layer1 not in self.gradient_history or layer2 not in self.gradient_history:
            return 0.0
            
        grads1 = torch.stack(self.gradient_history[layer1])
        grads2 = torch.stack(self.gradient_history[layer2])
        
        # Flatten and normalize
        grads1_flat = grads1.view(len(grads1), -1)
        grads2_flat = grads2.view(len(grads2), -1)
        
        grads1_norm = F.normalize(grads1_flat, p=2, dim=1)
        grads2_norm = F.normalize(grads2_flat, p=2, dim=1)
        
        # Compute cosine similarity
        alignment = (grads1_norm * grads2_norm).sum(dim=1).mean()
        
        return alignment.item()


class RepresentationAnalyzer:
    """
    Analyze learned representations and their evolution during training
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.representation_history = defaultdict(list)
        self.similarity_matrices = {}
        
    def capture_representations(self, inputs: torch.Tensor, layer_patterns: List[str]):
        """Capture representations from specified layers"""
        
        representations = {}
        
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                representations[name] = output.detach().cpu()
            return hook
            
        hooks = []
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in layer_patterns):
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)
                
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
            
        # Clean up
        for hook in hooks:
            hook.remove()
            
        # Store representations
        for name, rep in representations.items():
            self.representation_history[name].append(rep)
            
    def compute_representation_dynamics(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze how representations change over time
        """
        
        dynamics = {}
        
        for layer_name, rep_list in self.representation_history.items():
            if len(rep_list) < 2:
                continue
                
            # Compute various metrics
            dynamics[layer_name] = {}
            
            # Representation drift (how much they change)
            drift_scores = []
            for i in range(1, len(rep_list)):
                prev_rep = rep_list[i-1].view(rep_list[i-1].shape[0], -1)
                curr_rep = rep_list[i].view(rep_list[i].shape[0], -1)
                
                # Normalize
                prev_norm = F.normalize(prev_rep, p=2, dim=1)
                curr_norm = F.normalize(curr_rep, p=2, dim=1)
                
                # Compute similarity
                similarity = (prev_norm * curr_norm).sum(dim=1).mean()
                drift = 1 - similarity.item()
                drift_scores.append(drift)
                
            dynamics[layer_name]['mean_drift'] = np.mean(drift_scores)
            dynamics[layer_name]['drift_variance'] = np.var(drift_scores)
            
            # Representation rank (complexity)
            latest_rep = rep_list[-1].view(rep_list[-1].shape[0], -1)
            _, s, _ = torch.svd(latest_rep)
            
            # Effective rank
            s_normalized = s / s.sum()
            entropy = -(s_normalized * torch.log(s_normalized + 1e-10)).sum()
            effective_rank = torch.exp(entropy).item()
            
            dynamics[layer_name]['effective_rank'] = effective_rank
            dynamics[layer_name]['rank_ratio'] = effective_rank / len(s)
            
        return dynamics
    
    def compute_cka_similarity(self, layer1: str, layer2: str) -> float:
        """
        Compute Centered Kernel Alignment (CKA) similarity between layers
        State-of-the-art method for comparing representations
        """
        
        if layer1 not in self.representation_history or layer2 not in self.representation_history:
            return 0.0
            
        # Get latest representations
        rep1 = self.representation_history[layer1][-1]
        rep2 = self.representation_history[layer2][-1]
        
        # Flatten
        rep1_flat = rep1.view(rep1.shape[0], -1)
        rep2_flat = rep2.view(rep2.shape[0], -1)
        
        # Compute Gram matrices
        gram1 = torch.mm(rep1_flat, rep1_flat.T)
        gram2 = torch.mm(rep2_flat, rep2_flat.T)
        
        # Center Gram matrices
        n = gram1.shape[0]
        centering = torch.eye(n) - torch.ones(n, n) / n
        
        gram1_centered = centering @ gram1 @ centering
        gram2_centered = centering @ gram2 @ centering
        
        # Compute CKA
        numerator = torch.trace(gram1_centered @ gram2_centered)
        denominator = torch.sqrt(
            torch.trace(gram1_centered @ gram1_centered) * 
            torch.trace(gram2_centered @ gram2_centered)
        )
        
        cka = (numerator / (denominator + 1e-10)).item()
        
        return cka
    
    def analyze_feature_emergence(self, threshold: float = 0.1) -> Dict[str, List[int]]:
        """
        Track when specific features emerge during training
        """
        
        emerging_features = {}
        
        for layer_name, rep_list in self.representation_history.items():
            if len(rep_list) < 2:
                continue
                
            emerging_features[layer_name] = []
            
            # Track feature activation over time
            initial_rep = rep_list[0].view(rep_list[0].shape[0], -1)
            
            for t, rep in enumerate(rep_list[1:], 1):
                curr_rep = rep.view(rep.shape[0], -1)
                
                # Find newly activated features
                initial_active = (initial_rep.abs() > threshold).float().mean(dim=0)
                curr_active = (curr_rep.abs() > threshold).float().mean(dim=0)
                
                # Features that became active
                newly_active = ((curr_active > 0.5) & (initial_active < 0.1)).nonzero()
                
                for feature_idx in newly_active:
                    emerging_features[layer_name].append({
                        'feature_idx': feature_idx.item(),
                        'emergence_time': t,
                        'activation_strength': curr_active[feature_idx].item()
                    })
                    
        return emerging_features
    
    def compute_representation_collapse(self) -> Dict[str, float]:
        """
        Detect representation collapse (all representations becoming similar)
        Important for detecting training issues
        """
        
        collapse_scores = {}
        
        for layer_name, rep_list in self.representation_history.items():
            if not rep_list:
                continue
                
            latest_rep = rep_list[-1]
            
            # Reshape to (batch, features)
            if latest_rep.dim() > 2:
                latest_rep = latest_rep.view(latest_rep.shape[0], -1)
                
            # Compute pairwise similarities
            rep_norm = F.normalize(latest_rep, p=2, dim=1)
            similarity_matrix = torch.mm(rep_norm, rep_norm.T)
            
            # Remove diagonal
            mask = 1 - torch.eye(similarity_matrix.shape[0])
            off_diagonal = similarity_matrix * mask
            
            # High mean similarity indicates collapse
            mean_similarity = off_diagonal.sum() / (mask.sum() + 1e-10)
            
            collapse_scores[layer_name] = mean_similarity.item()
            
        return collapse_scores