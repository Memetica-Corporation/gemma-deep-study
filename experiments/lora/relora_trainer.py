"""
ReLoRA Implementation for Gemma-3
High-Rank Training Through Low-Rank Updates (arxiv:2307.05695)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class ReLoRAConfig:
    """Configuration for ReLoRA training"""
    warmup_steps: int = 1000  # Initial full-rank warmup
    reset_interval: int = 2000  # Steps between resets
    lora_rank: int = 16  # Rank for each LoRA update
    lora_alpha: float = 32.0  # LoRA scaling
    lora_dropout: float = 0.1
    lr_multiplier: float = 2.0  # LR multiplier during ReLoRA
    quick_warmup_steps: int = 100  # Quick warmup after reset
    target_modules: List[str] = None  # Which modules to apply LoRA
    
    def __post_init__(self):
        if self.target_modules is None:
            # Default: attention and MLP layers
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", 
                                  "gate_proj", "up_proj", "down_proj"]


class JaggedCosineScheduler:
    """Jagged cosine learning rate scheduler for ReLoRA"""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        base_lr: float,
        relora_lr_multiplier: float,
        warmup_steps: int,
        quick_warmup_steps: int,
        total_steps: int,
        reset_interval: int
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.relora_lr_multiplier = relora_lr_multiplier
        self.warmup_steps = warmup_steps
        self.quick_warmup_steps = quick_warmup_steps
        self.total_steps = total_steps
        self.reset_interval = reset_interval
        self.current_step = 0
        self.last_reset_step = 0
        self.in_relora = False
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Initial warmup
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        elif (self.current_step - self.last_reset_step) <= self.quick_warmup_steps:
            # Quick warmup after reset
            progress = (self.current_step - self.last_reset_step) / self.quick_warmup_steps
            target_lr = self.base_lr * self.relora_lr_multiplier
            lr = target_lr * progress
        else:
            # Cosine decay
            if self.in_relora:
                base = self.base_lr * self.relora_lr_multiplier
            else:
                base = self.base_lr
                
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = base * 0.5 * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        
    def on_reset(self):
        """Called when ReLoRA reset happens"""
        self.last_reset_step = self.current_step
        self.in_relora = True


class ReLoRATrainer:
    """ReLoRA training implementation"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ReLoRAConfig,
        optimizer: optim.Optimizer,
        scheduler: JaggedCosineScheduler
    ):
        self.base_model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.current_model = model
        self.step_count = 0
        self.reset_count = 0
        self.training_history = []
        
        # Track LoRA parameters for merging
        self.lora_params = []
        
    def warmup_phase(self, dataloader, num_steps: int):
        """Initial full-rank warmup training"""
        print(f"Starting warmup phase for {num_steps} steps...")
        
        self.current_model.train()
        total_loss = 0
        
        for step in range(num_steps):
            # Get batch
            batch = next(iter(dataloader))
            
            # Forward pass
            outputs = self.current_model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.step_count += 1
            
            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Warmup Step {step}/{num_steps}: Loss = {avg_loss:.4f}, "
                      f"LR = {self.scheduler.optimizer.param_groups[0]['lr']:.6f}")
                
        print(f"Warmup complete. Average loss: {total_loss/num_steps:.4f}")
        
    def initialize_lora(self):
        """Initialize LoRA on the current model"""
        print(f"Initializing LoRA with rank={self.config.lora_rank}")
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.current_model = get_peft_model(self.base_model, lora_config)
        
        # Update optimizer to include LoRA parameters
        lora_params = [p for n, p in self.current_model.named_parameters() 
                      if 'lora' in n.lower() and p.requires_grad]
        
        self.optimizer.add_param_group({
            'params': lora_params,
            'lr': self.optimizer.param_groups[0]['lr']
        })
        
        return lora_params
        
    def merge_and_reset(self):
        """Merge LoRA weights and reset for next iteration"""
        print(f"Performing ReLoRA reset #{self.reset_count + 1}")
        
        # Merge LoRA weights into base model
        self.current_model = self.current_model.merge_and_unload()
        self.base_model = self.current_model
        
        # Reset optimizer state for LoRA parameters
        # Remove LoRA param group
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups.pop()
            
        # Reinitialize LoRA
        lora_params = self.initialize_lora()
        
        # Notify scheduler about reset
        self.scheduler.on_reset()
        
        self.reset_count += 1
        
        return lora_params
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.current_model.train()
        
        # Forward pass
        outputs = self.current_model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.current_model.parameters(), 
            max_norm=1.0
        )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Update scheduler
        lr = self.scheduler.step()
        
        self.step_count += 1
        
        # Track history
        self.training_history.append({
            'step': self.step_count,
            'loss': loss.item(),
            'lr': lr,
            'reset_count': self.reset_count
        })
        
        return loss.item()
        
    def relora_phase(self, dataloader, num_resets: int):
        """Main ReLoRA training phase"""
        print(f"Starting ReLoRA phase with {num_resets} resets...")
        
        # Initialize first LoRA
        self.initialize_lora()
        
        for reset_idx in range(num_resets):
            print(f"\n=== ReLoRA Iteration {reset_idx + 1}/{num_resets} ===")
            
            # Train for reset_interval steps
            total_loss = 0
            for step in range(self.config.reset_interval):
                batch = next(iter(dataloader))
                loss = self.train_step(batch)
                total_loss += loss
                
                if step % 100 == 0:
                    avg_loss = total_loss / (step + 1)
                    print(f"Step {step}/{self.config.reset_interval}: "
                          f"Loss = {avg_loss:.4f}, "
                          f"LR = {self.scheduler.optimizer.param_groups[0]['lr']:.6f}")
                    
            # Merge and reset
            if reset_idx < num_resets - 1:  # Don't reset after last iteration
                self.merge_and_reset()
                
        print("\nReLoRA training complete!")
        
    def plot_training_history(self):
        """Plot training metrics"""
        if not self.training_history:
            print("No training history to plot")
            return
            
        history = self.training_history
        steps = [h['step'] for h in history]
        losses = [h['loss'] for h in history]
        lrs = [h['lr'] for h in history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot loss
        ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('ReLoRA Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Mark reset points
        reset_steps = [i * self.config.reset_interval + self.config.warmup_steps 
                      for i in range(1, self.reset_count)]
        for reset_step in reset_steps:
            if reset_step < max(steps):
                ax1.axvline(x=reset_step, color='r', linestyle='--', 
                          alpha=0.5, label='Reset' if reset_step == reset_steps[0] else '')
        
        if reset_steps:
            ax1.legend()
        
        # Plot learning rate
        ax2.plot(steps, lrs, 'g-', linewidth=2)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Jagged Cosine Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/relora_training.png', dpi=150)
        plt.show()
        
    def get_effective_rank(self) -> Dict[str, int]:
        """Compute effective rank of current model weights"""
        ranks = {}
        
        for name, param in self.base_model.named_parameters():
            if len(param.shape) == 2:  # Only for matrices
                with torch.no_grad():
                    U, S, V = torch.svd(param)
                    # Effective rank: singular values > threshold
                    threshold = S[0] * 1e-3 if S[0] > 0 else 1e-10
                    effective_rank = torch.sum(S > threshold).item()
                    ranks[name] = effective_rank
                    
        return ranks
        
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'model_state_dict': self.base_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'reset_count': self.reset_count,
            'config': self.config,
            'training_history': self.training_history
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")


def create_relora_trainer(model, learning_rate=1e-4, total_steps=10000):
    """Factory function to create ReLoRA trainer"""
    
    config = ReLoRAConfig(
        warmup_steps=1000,
        reset_interval=2000,
        lora_rank=16,
        lora_alpha=32.0,
        lr_multiplier=2.0
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    scheduler = JaggedCosineScheduler(
        optimizer=optimizer,
        base_lr=learning_rate,
        relora_lr_multiplier=config.lr_multiplier,
        warmup_steps=config.warmup_steps,
        quick_warmup_steps=config.quick_warmup_steps,
        total_steps=total_steps,
        reset_interval=config.reset_interval
    )
    
    trainer = ReLoRATrainer(model, config, optimizer, scheduler)
    
    return trainer


if __name__ == "__main__":
    print("ReLoRA Trainer initialized")
    print("This module implements High-Rank Training Through Low-Rank Updates")
    print("Import and use with your Gemma model for efficient training")