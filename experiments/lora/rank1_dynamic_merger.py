"""
Dynamic Rank-1 LoRA Merger
Implementation of the novel technique where rank-1 LoRA is initialized and merged at each step.
Based on the principle that gradient descent updates are inherently rank-1 for batch size 1.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
try:
    from scipy.linalg import svd
except Exception:
    svd = None


@dataclass
class Rank1MergeConfig:
    """Configuration for rank-1 dynamic merging"""
    merge_frequency: int = 1  # Merge every N steps
    alpha: float = 1.0  # LoRA scaling factor
    track_svd: bool = True  # Track effective rank via SVD
    svd_frequency: int = 100  # Compute SVD every N steps
    learning_rate_multiplier: float = 2.0  # LR multiplier during merge


class Rank1LoRALayer(nn.Module):
    """Single rank-1 LoRA layer"""
    
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Rank-1 decomposition: W = u @ v^T
        self.lora_u = nn.Parameter(torch.randn(out_features, 1) * 0.01)
        self.lora_v = nn.Parameter(torch.randn(1, in_features) * 0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rank-1 update"""
        # Efficient rank-1 matrix-vector multiplication
        return self.alpha * (self.lora_u @ (self.lora_v @ x.T)).T
    
    def get_weight_update(self) -> torch.Tensor:
        """Get the full weight update matrix"""
        return self.alpha * (self.lora_u @ self.lora_v)


class DynamicRank1Merger:
    """Manages dynamic rank-1 LoRA merging during training"""
    
    def __init__(self, model: nn.Module, config: Rank1MergeConfig):
        self.model = model
        self.config = config
        self.step = 0
        self.merge_history = []
        self.svd_history = []
        
        # Track which layers have LoRA
        self.lora_layers: Dict[str, Rank1LoRALayer] = {}
        self.base_layers: Dict[str, nn.Module] = {}
        
        # Initialize LoRA for linear layers
        self._initialize_lora_layers()
        
    def _initialize_lora_layers(self):
        """Initialize rank-1 LoRA for all linear layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.base_layers[name] = module
                # Create rank-1 LoRA for this layer
                lora = Rank1LoRALayer(
                    module.in_features,
                    module.out_features,
                    self.config.alpha
                )
                self.lora_layers[name] = lora
                
    def merge_and_reinit(self):
        """Merge current LoRA weights and reinitialize"""
        merge_info = {}
        
        for name, base_layer in self.base_layers.items():
            lora_layer = self.lora_layers[name]
            
            # Get weight update from LoRA
            weight_update = lora_layer.get_weight_update()
            
            # Merge into base weights
            with torch.no_grad():
                base_layer.weight.add_(weight_update)
            
            # Track merge info
            merge_info[name] = {
                'update_norm': torch.norm(weight_update).item(),
                'weight_norm': torch.norm(base_layer.weight).item()
            }
            
            # Reinitialize LoRA parameters
            nn.init.normal_(lora_layer.lora_u, std=0.01)
            nn.init.normal_(lora_layer.lora_v, std=0.01)
            
        self.merge_history.append(merge_info)
        
    def compute_effective_rank(self, layer_name: str) -> Tuple[int, np.ndarray]:
        """Compute effective rank of weight matrix via SVD"""
        with torch.no_grad():
            weight = self.base_layers[layer_name].weight
            if svd is not None:
                s = torch.tensor(svd(weight.cpu().numpy(), full_matrices=False)[1])
            else:
                # Fallback to torch.linalg.svdvals
                s = torch.linalg.svdvals(weight)
            
            # Compute effective rank (number of singular values > threshold)
            threshold = s[0] * 1e-3 if s[0] > 0 else 1e-10
            effective_rank = np.sum(s > threshold)
            
            return effective_rank, s
            
    def step_update(self, loss: torch.Tensor):
        """Called after each optimization step"""
        self.step += 1
        
        # Merge if needed
        if self.step % self.config.merge_frequency == 0:
            self.merge_and_reinit()
            
        # Track SVD if needed
        if self.config.track_svd and self.step % self.config.svd_frequency == 0:
            svd_info = {}
            for name in self.base_layers.keys():
                rank, singular_values = self.compute_effective_rank(name)
                svd_info[name] = {
                    'effective_rank': rank,
                    'top_10_singular': singular_values[:10].tolist()
                }
            self.svd_history.append({
                'step': self.step,
                'loss': loss.item(),
                'svd_info': svd_info
            })
            
    def plot_rank_evolution(self, layer_names: Optional[List[str]] = None):
        """Plot effective rank evolution over training"""
        if not self.svd_history:
            print("No SVD history to plot")
            return
            
        if layer_names is None:
            layer_names = list(self.base_layers.keys())[:3]  # Plot first 3 layers
            
        fig, axes = plt.subplots(len(layer_names), 2, figsize=(12, 4*len(layer_names)))
        if len(layer_names) == 1:
            axes = axes.reshape(1, -1)
            
        for idx, layer_name in enumerate(layer_names):
            steps = []
            ranks = []
            losses = []
            
            for record in self.svd_history:
                if layer_name in record['svd_info']:
                    steps.append(record['step'])
                    ranks.append(record['svd_info'][layer_name]['effective_rank'])
                    losses.append(record['loss'])
                    
            # Plot effective rank
            axes[idx, 0].plot(steps, ranks, 'b-', linewidth=2)
            axes[idx, 0].set_xlabel('Training Step')
            axes[idx, 0].set_ylabel('Effective Rank')
            axes[idx, 0].set_title(f'{layer_name}: Rank Evolution')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot loss vs rank
            axes[idx, 1].scatter(ranks, losses, alpha=0.5, c=steps, cmap='viridis')
            axes[idx, 1].set_xlabel('Effective Rank')
            axes[idx, 1].set_ylabel('Loss')
            axes[idx, 1].set_title(f'{layer_name}: Loss vs Rank')
            axes[idx, 1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('visualizations/rank1_evolution.png', dpi=150)
        plt.show()
        
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        stats = {
            'total_merges': len(self.merge_history),
            'current_step': self.step,
        }
        
        if self.svd_history:
            latest_svd = self.svd_history[-1]['svd_info']
            avg_rank = np.mean([
                info['effective_rank'] 
                for info in latest_svd.values()
            ])
            stats['average_effective_rank'] = avg_rank
            
        return stats


class Rank1MergeTrainer:
    """Training loop with rank-1 dynamic merging"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Rank1MergeConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.merger = DynamicRank1Merger(model, config)
        self.config = config
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step with rank-1 merging"""
        self.model.train()
        
        # Forward pass with LoRA
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Apply LoRA updates to gradients
        for name, base_layer in self.merger.base_layers.items():
            lora_layer = self.merger.lora_layers[name]
            if base_layer.weight.grad is not None:
                # Add LoRA gradient contribution
                lora_grad = lora_layer.get_weight_update()
                base_layer.weight.grad.add_(lora_grad)
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Merge and track
        self.merger.step_update(loss)
        
        return loss.item()
        
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            if num_batches % 100 == 0:
                print(f"Batch {num_batches}: Loss = {loss:.4f}, "
                      f"Stats = {self.merger.get_statistics()}")
                
        return total_loss / num_batches


if __name__ == "__main__":
    # Example usage
    print("Rank-1 Dynamic Merger initialized")
    print("This module implements the novel rank-1 LoRA merging technique")
    print("Import and use with your Gemma model for training")