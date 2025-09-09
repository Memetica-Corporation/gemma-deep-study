import sys
import types

# Provide a lightweight stub for wandb if not installed
if 'wandb' not in sys.modules:
    sys.modules['wandb'] = types.SimpleNamespace(
        init=lambda **kwargs: None,
        watch=lambda *args, **kwargs: None,
        log=lambda *args, **kwargs: None,
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from training.advanced_trainer import TrainingConfig, ReLoRATrainer


class DummyClassificationModel(nn.Module):
    def __init__(self, input_dim: int = 4, num_classes: int = 3):
        super().__init__()
        # Name contains 'q_proj' so LoRA targets will be picked up
        self.q_proj = nn.Linear(input_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        logits = self.q_proj(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SimpleNamespace(loss=loss if loss is not None else logits.mean(), logits=logits)


def test_train_step_accumulates_gradients_without_no_grad():
    torch.manual_seed(0)

    model = DummyClassificationModel(input_dim=4, num_classes=3)
    config = TrainingConfig(
        device='cpu',
        mixed_precision=True,  # Should not disable grads on CPU
        gradient_accumulation_steps=2,
        use_wandb=False,
        lora_target_modules=['q_proj'],
        batch_size=2,
    )

    trainer = ReLoRATrainer(model, config)

    batch = {
        'input_ids': torch.randn(2, 4),
        'labels': torch.randint(0, 3, (2,))
    }

    loss_val = trainer.train_step(batch)

    # Since accumulation_steps=2, optimizer step should not have run yet and grads must exist
    lora_grads = [p.grad for n, p in trainer.model.named_parameters() if 'lora_' in n]
    assert any(g is not None for g in lora_grads), "Expected LoRA parameter gradients to be computed"
    assert isinstance(loss_val, float)


