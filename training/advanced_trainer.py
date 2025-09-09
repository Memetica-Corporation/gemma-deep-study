"""
State-of-the-art Training Framework for Gemma-3
Implements cutting-edge optimization and training techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import math
import wandb
from pathlib import Path
import json
from tqdm.auto import tqdm
import time


@dataclass
class TrainingConfig:
    """Advanced training configuration"""
    # Model
    model_name: str = "gemma-3-4b"
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    gradient_clip: float = 1.0
    
    # LoRA specific
    lora_rank: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # ReLoRA specific  
    relora_enabled: bool = False
    relora_reset_interval: int = 1000
    relora_warmup_steps: int = 100
    relora_lr_multiplier: float = 2.0
    
    # Dynamic Rank-1 merging
    rank1_merge_enabled: bool = False
    rank1_merge_frequency: int = 1
    rank1_merge_decay: float = 0.999
    
    # Training
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 500
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    
    # Advanced features
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    flash_attention: bool = True
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # Memory optimization
    cpu_offload: bool = False
    activation_checkpointing: bool = True
    optimizer_state_sharding: bool = False
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Data
    max_length: int = 2048
    dataset_name: Optional[str] = None
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "gemma-3-frontier"
    wandb_entity: Optional[str] = None
    
    # Hardware
    device: str = "mps"  # mps for Mac, cuda for GPU
    num_workers: int = 4


class AdvancedLoRAModule(nn.Module):
    """
    Advanced LoRA implementation with dynamic rank and merging capabilities
    """
    
    def __init__(self, 
                 base_layer: nn.Module,
                 rank: int,
                 alpha: float,
                 dropout: float = 0.0,
                 merge_weights: bool = False):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False
        self.merge_weights = merge_weights
        
        # Get dimensions
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type: {type(base_layer)}")
            
        # Create LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Statistics tracking
        self.update_count = 0
        self.gradient_norms = []
        self.activation_stats = defaultdict(list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward
        result = self.base_layer(x)
        
        if not self.merged:
            # Apply LoRA
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling
            
        # Track statistics
        self.update_count += 1
        if self.training:
            self.activation_stats['mean'].append(result.mean().item())
            self.activation_stats['std'].append(result.std().item())
            
        return result
    
    def merge(self):
        """Merge LoRA weights into base layer"""
        if not self.merged:
            if isinstance(self.base_layer, nn.Linear):
                self.base_layer.weight.data += (
                    self.lora_B @ self.lora_A * self.scaling
                )
            self.merged = True
            
    def unmerge(self):
        """Unmerge LoRA weights from base layer"""
        if self.merged:
            if isinstance(self.base_layer, nn.Linear):
                self.base_layer.weight.data -= (
                    self.lora_B @ self.lora_A * self.scaling
                )
            self.merged = False
            
    def get_effective_rank(self) -> int:
        """Compute effective rank using SVD"""
        weight_delta = self.lora_B @ self.lora_A
        # Prefer torch.linalg.svdvals; fall back to torch.svd for older versions
        try:
            s = torch.linalg.svdvals(weight_delta)
        except Exception:
            _, s, _ = torch.svd(weight_delta)
        
        # Compute effective rank (99% energy)
        cumsum = torch.cumsum(s**2, dim=0)
        total = cumsum[-1]
        rank = torch.sum(cumsum < 0.99 * total).item() + 1
        
        return rank


class ReLoRATrainer:
    """
    Implements ReLoRA: High-Rank Training Through Low-Rank Updates
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup LoRA modules
        self.lora_modules = self._setup_lora_modules()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        # Enable CUDA GradScaler only when on CUDA; use torch.autocast for MPS/CUDA
        use_cuda_amp = config.mixed_precision and torch.device(config.device).type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
        
        # Statistics
        self.training_stats = defaultdict(list)
        self.step = 0
        self.epoch = 0
        
        # Setup experiment tracking
        if config.use_wandb:
            self._init_wandb()
            
    def _setup_lora_modules(self) -> Dict[str, AdvancedLoRAModule]:
        """Setup LoRA modules for target layers"""
        lora_modules = {}
        
        for name, module in self.model.named_modules():
            if any(target in name for target in self.config.lora_target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA module
                    lora_module = AdvancedLoRAModule(
                        base_layer=module,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    
                    # Replace in model
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, child_name, lora_module)
                    
                    lora_modules[name] = lora_module
                    
        print(f"Setup {len(lora_modules)} LoRA modules")
        return lora_modules
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper parameter groups"""
        
        # Separate LoRA and base parameters
        lora_params = []
        base_params = []
        
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                base_params.append(param)
                
        # Create parameter groups
        param_groups = [
            {'params': lora_params, 'lr': self.config.learning_rate},
            {'params': base_params, 'lr': self.config.learning_rate * 0.1}  # Lower LR for base
        ]
        
        optimizer = AdamW(
            param_groups,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.relora_reset_interval if self.config.relora_enabled else 1000,
            T_mult=1,
            eta_min=self.config.learning_rate * 0.1
        )
        return scheduler
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            config=self.config.__dict__,
            name=f"{self.config.model_name}_relora_{self.config.lora_rank}"
        )
        wandb.watch(self.model, log_freq=100)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision context (device-aware)
        autocast_dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32
        use_autocast = self.config.mixed_precision and self.device.type in ("cuda", "mps")
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=use_autocast):
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
        # Backward pass
        if self.scaler and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # Gradient accumulation
        if (self.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clip > 0:
                if self.scaler and self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
                
            # Optimizer step
            if self.scaler and self.scaler.is_enabled():
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
                
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # ReLoRA reset
            if self.config.relora_enabled and self.step % self.config.relora_reset_interval == 0:
                self._relora_reset()
                
            # Rank-1 merging
            if self.config.rank1_merge_enabled and self.step % self.config.rank1_merge_frequency == 0:
                self._rank1_merge()
                
        # Update statistics
        self.training_stats['loss'].append(loss.item())
        self.training_stats['learning_rate'].append(self.scheduler.get_last_lr()[0])
        
        # Log to wandb
        if self.config.use_wandb and self.step % self.config.logging_steps == 0:
            wandb.log({
                'loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0],
                'step': self.step
            })
            
        self.step += 1
        return loss.item()
    
    def _relora_reset(self):
        """
        ReLoRA reset: merge current LoRA, initialize new LoRA
        """
        print(f"ReLoRA reset at step {self.step}")
        
        for name, lora_module in self.lora_modules.items():
            # Merge current LoRA into base
            lora_module.merge()
            
            # Reset LoRA parameters
            nn.init.kaiming_uniform_(lora_module.lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_module.lora_B)
            
            # Unmerge for continued training
            lora_module.unmerge()
            
        # Boost learning rate temporarily
        for param_group in self.optimizer.param_groups:
            if 'lora_' in str(param_group['params'][0]):
                param_group['lr'] *= self.config.relora_lr_multiplier
                
    def _rank1_merge(self):
        """
        Dynamic Rank-1 merging: frequently merge rank-1 updates
        """
        
        for name, lora_module in self.lora_modules.items():
            # Get current gradients as rank-1 update
            if lora_module.lora_A.grad is not None and lora_module.lora_B.grad is not None:
                # Compute rank-1 approximation of gradient
                grad_A = lora_module.lora_A.grad.mean(dim=0, keepdim=True)
                grad_B = lora_module.lora_B.grad.mean(dim=1, keepdim=True)
                
                # Apply rank-1 update to base weights
                rank1_update = grad_B @ grad_A
                lora_module.base_layer.weight.data -= (
                    self.config.learning_rate * rank1_update * lora_module.scaling
                )
                
                # Decay LoRA weights
                lora_module.lora_A.data *= self.config.rank1_merge_decay
                lora_module.lora_B.data *= self.config.rank1_merge_decay
                
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model"""
        
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                total_loss += loss.item()
                total_steps += 1
                
        avg_loss = total_loss / total_steps
        
        if self.config.use_wandb:
            wandb.log({'eval_loss': avg_loss, 'step': self.step})
            
        return {'eval_loss': avg_loss}
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint"""
        
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"checkpoint_{self.step}.pt"
            
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_stats': dict(self.training_stats)
        }
        
        # Save LoRA-specific information
        lora_info = {}
        for name, lora_module in self.lora_modules.items():
            lora_info[name] = {
                'effective_rank': lora_module.get_effective_rank(),
                'update_count': lora_module.update_count,
                'activation_stats': dict(lora_module.activation_stats)
            }
        checkpoint['lora_info'] = lora_info
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        
        print(f"Loaded checkpoint from {path} (step {self.step})")


"""
Note: SetBlockDecoder is provided in experiments/attention/set_block_decoder.py.
This duplicate implementation has been removed to avoid divergence.
"""


class MemoryEfficientTrainer:
    """
    Memory-efficient training with advanced techniques
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup gradient checkpointing
        if config.gradient_checkpointing:
            self._enable_gradient_checkpointing()
            
        # Setup activation checkpointing
        if config.activation_checkpointing:
            self._enable_activation_checkpointing()
            
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency"""
        
        def checkpoint_forward(module, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                module._forward_impl, *args, **kwargs
            )
            
        # Apply to transformer blocks
        for name, module in self.model.named_modules():
            if 'block' in name.lower() or 'layer' in name.lower():
                module._forward_impl = module.forward
                module.forward = lambda *args, m=module, **kwargs: checkpoint_forward(m, *args, **kwargs)
                
    def _enable_activation_checkpointing(self):
        """Enable activation checkpointing"""
        
        # This is model-specific, example for transformer blocks
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            
    def compute_memory_usage(self) -> Dict[str, float]:
        """Compute current memory usage"""
        
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved
            }
        elif self.device.type == 'mps':
            # MPS memory tracking (approximate)
            allocated = torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0
            return {
                'allocated_gb': allocated,
                'reserved_gb': allocated * 1.2  # Estimate
            }
        else:
            return {'allocated_gb': 0, 'reserved_gb': 0}