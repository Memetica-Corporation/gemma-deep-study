"""
Deprecated: ReLoRA implementation moved to training/relora.py
This file remains for backward compatibility and will import from the unified module.
"""

from training.relora import (
    ReLoRAConfig,
    JaggedCosineScheduler,
    ReLoRATrainer,
    create_relora_trainer,
)


__all__ = [
    'ReLoRAConfig',
    'JaggedCosineScheduler',
    'ReLoRATrainer',
    'create_relora_trainer',
]
def create_relora_trainer(model, learning_rate=1e-4, total_steps=10000):
    return __import__('training.relora', fromlist=['create_relora_trainer']).create_relora_trainer(
        model, learning_rate, total_steps
    )


if __name__ == "__main__":
    print("ReLoRA Trainer initialized")
    print("This module implements High-Rank Training Through Low-Rank Updates")
    print("Import and use with your Gemma model for efficient training")