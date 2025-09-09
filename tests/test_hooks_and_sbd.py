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
from types import SimpleNamespace

from gemma_architecture.core import GemmaArchitecture


class TinyTwoLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 8)
        self.layer2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


def test_backward_hook_updates_gradient_norm():
    model = TinyTwoLayer()
    arch = GemmaArchitecture(model)
    arch.register_hooks()

    x = torch.randn(2, 8)
    y = model(x).sum()
    y.backward()

    # Hooks should have populated layer_stats with gradient norms
    assert any(ls.gradient_norm != 0.0 for ls in arch.layer_stats), "Expected non-zero gradient norms recorded by hooks"


def test_sbd_mask_fallback_token_id(monkeypatch):
    # Build a minimal tokenizer/model shim that lacks mask_token_id
    class Tok:
        mask_token_id = None
        eos_token_id = 2
        def __call__(self, text, return_tensors='pt'):
            return {'input_ids': torch.ones(1, 3, dtype=torch.long)}
        def decode(self, ids, skip_special_tokens=True):
            return 'ok'

    class Cfg:
        hidden_size = 8
        vocab_size = 10

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()
            self.lm_head = nn.Linear(8, 10)
            self.emb = nn.Embedding(10, 8)
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
            h = self.emb(input_ids)
            logits = self.lm_head(h)
            out = SimpleNamespace(logits=logits)
            if output_hidden_states:
                out.hidden_states = [h]
            return out

    from experiments.attention.set_block_decoder import SBDConfig, SetBlockDecoder

    model = LM()
    tok = Tok()
    sbd = SetBlockDecoder(model, tok, SBDConfig(block_size=2))

    # Should not raise due to missing mask_token_id; should use eos fallback
    text, stats = sbd.generate_with_sbd("hello", max_new_tokens=1)
    assert isinstance(text, str)
    assert 'tokens_generated' in stats


