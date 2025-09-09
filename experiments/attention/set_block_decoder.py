"""
Set Block Decoding (SBD) Implementation for Gemma-3
Based on "Set Block Decoding is a Language Model Inference Accelerator" (arxiv:2509.04185)
Accelerates generation by sampling multiple future tokens in parallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import numpy as np
import time
import matplotlib.pyplot as plt


@dataclass
class SBDConfig:
    """Configuration for Set Block Decoding"""
    block_size: int = 4  # Number of tokens to predict in parallel
    masking_strategy: str = "random"  # random, structured, adaptive
    masking_ratio: float = 0.5  # Ratio of tokens to mask in MATP
    confidence_threshold: float = 0.8  # Threshold for accepting predictions
    max_iterations: int = 3  # Max refinement iterations per block
    temperature: float = 0.7
    use_discrete_diffusion: bool = False  # Use diffusion-based solver


class MaskedTokenPredictor(nn.Module):
    """Masked token prediction head for SBD"""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Prediction head for masked tokens
        self.masked_lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        masked_positions: torch.Tensor
    ) -> torch.Tensor:
        """Predict masked tokens"""
        
        # Extract hidden states at masked positions
        batch_size, seq_len, _ = hidden_states.shape
        masked_hidden = hidden_states[masked_positions]
        
        # Predict tokens
        predictions = self.masked_lm_head(masked_hidden)
        
        return predictions


class SetBlockDecoder:
    """Set Block Decoding implementation for accelerated inference"""
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: SBDConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Initialize masked token predictor
        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        self.matp_head = MaskedTokenPredictor(hidden_size, vocab_size)
        
        # Move to same device as model
        device = next(model.parameters()).device
        self.matp_head = self.matp_head.to(device)
        self.device = device
        
        # Statistics tracking
        self.stats = {
            'blocks_generated': 0,
            'tokens_accepted': 0,
            'tokens_rejected': 0,
            'iterations_per_block': [],
            'speedup_factor': []
        }
        
    def create_block_mask(
        self,
        block_size: int,
        strategy: str = "random"
    ) -> torch.Tensor:
        """Create mask pattern for block prediction"""
        
        if strategy == "random":
            # Random masking
            mask = torch.rand(block_size) < self.config.masking_ratio
            # Ensure at least one token is masked
            if not mask.any():
                mask[torch.randint(0, block_size, (1,))] = True
                
        elif strategy == "structured":
            # Structured masking (e.g., every other token)
            mask = torch.zeros(block_size, dtype=torch.bool)
            mask[::2] = True
            
        elif strategy == "adaptive":
            # Adaptive masking based on position
            # Mask more tokens at the beginning of the block
            probs = torch.linspace(0.8, 0.2, block_size)
            mask = torch.rand(block_size) < probs
            
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")
            
        return mask
        
    def parallel_token_prediction(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict multiple tokens in parallel using MATP.
        Returns (predictions, confidences, mask) where mask is a 1D bool tensor of length block_size.
        """
        
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Create block at the end of sequence
        block_start = seq_len
        block_ids = torch.zeros(
            (batch_size, self.config.block_size),
            dtype=torch.long,
            device=self.device
        )
        
        # Initialize with mask tokens (robust fallback for causal LMs)
        mask_token_id = (
            self.tokenizer.mask_token_id
            or getattr(self.tokenizer, 'unk_token_id', None)
            or self.tokenizer.eos_token_id
        )
        block_ids.fill_(mask_token_id)
        
        # Concatenate with input
        extended_ids = torch.cat([input_ids, block_ids], dim=1)
        
        # Create attention mask
        if attention_mask is not None:
            block_attention = torch.ones(
                (batch_size, self.config.block_size),
                dtype=attention_mask.dtype,
                device=self.device
            )
            extended_attention = torch.cat([attention_mask, block_attention], dim=1)
        else:
            extended_attention = None
            
        # Get model hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=extended_ids,
                attention_mask=extended_attention,
                output_hidden_states=True
            )
            
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Predict tokens for masked positions
        mask = self.create_block_mask(self.config.block_size, self.config.masking_strategy)
        masked_positions = mask.nonzero(as_tuple=False).view(-1)
        
        predictions = []
        confidences = []
        
        for pos in masked_positions:
            # Get hidden state at this position
            hidden = hidden_states[:, block_start + pos]
            
            # Predict token
            logits = self.model.lm_head(hidden)
            probs = F.softmax(logits / self.config.temperature, dim=-1)
            
            # Sample token
            predicted_token = torch.multinomial(probs, 1).squeeze()
            confidence = probs.max().item()
            
            predictions.append(predicted_token)
            confidences.append(confidence)
            
        predictions = torch.stack(predictions) if predictions else torch.tensor([], device=self.device)
        confidences = torch.tensor(confidences)
        
        return predictions, confidences, mask
        
    def refine_block_predictions(
        self,
        input_ids: torch.Tensor,
        block_predictions: torch.Tensor,
        mask: torch.Tensor,
        iteration: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine block predictions iteratively"""
        
        if iteration >= self.config.max_iterations:
            return block_predictions, torch.ones(len(block_predictions))
            
        # Create new input with current predictions
        extended_ids = torch.cat([input_ids, block_predictions.unsqueeze(0)], dim=1)
        
        # Re-predict masked positions
        with torch.no_grad():
            outputs = self.model(input_ids=extended_ids, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states[-1]
        
        refined_predictions = block_predictions.clone()
        confidences = []
        
        # Refine predictions for low-confidence positions
        for i, is_masked in enumerate(mask):
            if is_masked:
                hidden = hidden_states[:, input_ids.shape[1] + i]
                logits = self.model.lm_head(hidden)
                probs = F.softmax(logits / self.config.temperature, dim=-1)
                
                confidence = probs.max().item()
                
                if confidence < self.config.confidence_threshold:
                    # Re-sample if low confidence
                    refined_predictions[i] = torch.multinomial(probs, 1).squeeze()
                    
                confidences.append(confidence)
                
        confidences = torch.tensor(confidences)
        
        # Recursive refinement if needed
        if confidences.mean() < self.config.confidence_threshold:
            return self.refine_block_predictions(
                input_ids, refined_predictions, mask, iteration + 1
            )
            
        return refined_predictions, confidences
        
    def generate_with_sbd(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Generate text using Set Block Decoding"""
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # Move inputs to device whether it's a BatchEncoding or a plain dict
        if hasattr(inputs, 'to'):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: (v.to(self.device) if hasattr(v, 'to') else v) for k, v in inputs.items()}
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        generated_ids = input_ids.clone()
        tokens_generated = 0
        
        start_time = time.perf_counter()
        
        while tokens_generated < max_new_tokens:
            # Predict block of tokens
            predictions, confidences, mask = self.parallel_token_prediction(
                generated_ids, attention_mask
            )
            
            if len(predictions) == 0:
                break
                
            # Create full block
            block = torch.zeros(self.config.block_size, dtype=torch.long, device=self.device)
            
            # Fill in predictions
            masked_positions = mask.nonzero(as_tuple=False).view(-1).to(block.device)
            if masked_positions.numel() > 0:
                block[masked_positions] = predictions[: masked_positions.numel()]
                
            # Fill non-masked positions with standard decoding
            non_masked = ~mask
            if non_masked.any():
                with torch.no_grad():
                    outputs = self.model(input_ids=generated_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1)
                    
                non_masked_positions = non_masked.nonzero().squeeze()
                if non_masked_positions.numel() > 0:
                    if non_masked_positions.dim() == 0:
                        non_masked_positions = non_masked_positions.unsqueeze(0)
                    block[non_masked_positions] = next_token
                    
            # Refine predictions
            block, final_confidences = self.refine_block_predictions(
                generated_ids, block, mask
            )
            
            # Accept high-confidence predictions
            accepted_tokens = []
            for i, (token, conf) in enumerate(zip(block, final_confidences)):
                if i >= len(final_confidences) or conf > self.config.confidence_threshold:
                    accepted_tokens.append(token)
                    self.stats['tokens_accepted'] += 1
                else:
                    self.stats['tokens_rejected'] += 1
                    break  # Stop at first low-confidence token
                    
            if accepted_tokens:
                accepted_tensor = torch.tensor(accepted_tokens, device=self.device).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, accepted_tensor], dim=1)
                tokens_generated += len(accepted_tokens)
                
                # Update attention mask
                if attention_mask is not None:
                    new_attention = torch.ones(
                        (1, len(accepted_tokens)),
                        dtype=attention_mask.dtype,
                        device=self.device
                    )
                    attention_mask = torch.cat([attention_mask, new_attention], dim=1)
                    
            else:
                # Fall back to standard decoding if no tokens accepted
                with torch.no_grad():
                    outputs = self.model(input_ids=generated_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                    
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                tokens_generated += 1
                
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)
                    ], dim=1)
                    
            self.stats['blocks_generated'] += 1
            
            # Check for EOS token
            if self.tokenizer.eos_token_id in generated_ids[0, -self.config.block_size:]:
                break
                
        end_time = time.perf_counter()
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate statistics
        elapsed_time = end_time - start_time
        tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else 0
        
        generation_stats = {
            'tokens_generated': tokens_generated,
            'time_elapsed': elapsed_time,
            'tokens_per_second': tokens_per_second,
            'blocks_generated': self.stats['blocks_generated'],
            'acceptance_rate': self.stats['tokens_accepted'] / max(1, self.stats['tokens_accepted'] + self.stats['tokens_rejected'])
        }
        
        return generated_text, generation_stats
        
    def benchmark_vs_standard(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        num_runs: int = 3
    ) -> Dict:
        """Benchmark SBD against standard autoregressive generation"""
        
        results = {
            'sbd': [],
            'standard': []
        }
        
        print("Benchmarking Set Block Decoding...")
        
        # Benchmark SBD
        for i in range(num_runs):
            _, stats = self.generate_with_sbd(prompt, max_new_tokens)
            results['sbd'].append(stats)
            print(f"  SBD Run {i+1}: {stats['tokens_per_second']:.1f} tokens/s")
            
        # Benchmark standard generation
        print("\nBenchmarking Standard Generation...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        for i in range(num_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
                
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
            tokens_per_second = tokens_generated / elapsed if elapsed > 0 else 0
            
            results['standard'].append({
                'tokens_generated': tokens_generated,
                'time_elapsed': elapsed,
                'tokens_per_second': tokens_per_second
            })
            
            print(f"  Standard Run {i+1}: {tokens_per_second:.1f} tokens/s")
            
        # Calculate statistics
        sbd_avg_speed = np.mean([r['tokens_per_second'] for r in results['sbd']])
        standard_avg_speed = np.mean([r['tokens_per_second'] for r in results['standard']])
        
        speedup = sbd_avg_speed / standard_avg_speed if standard_avg_speed > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Average SBD Speed: {sbd_avg_speed:.1f} tokens/s")
        print(f"Average Standard Speed: {standard_avg_speed:.1f} tokens/s")
        print(f"Speedup Factor: {speedup:.2f}x")
        print(f"Acceptance Rate: {results['sbd'][0]['acceptance_rate']:.2%}")
        
        return {
            'sbd_results': results['sbd'],
            'standard_results': results['standard'],
            'speedup_factor': speedup,
            'sbd_avg_speed': sbd_avg_speed,
            'standard_avg_speed': standard_avg_speed
        }
        
    def visualize_generation_pattern(self, text: str, save_path: Optional[str] = None):
        """Visualize the block generation pattern"""
        
        # Generate with tracking
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Track which tokens were generated in parallel
        generation_pattern = []
        
        # This would need more detailed tracking in actual generation
        # For now, create a sample pattern
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create visualization of parallel vs sequential generation
        # This is a simplified visualization
        
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('Token Position')
        ax.set_title('Set Block Decoding Pattern')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def create_sbd_accelerator(model_path: str, config: Optional[SBDConfig] = None):
    """Factory function to create SBD accelerator"""
    
    if config is None:
        config = SBDConfig()
        
    # Import transformers lazily to avoid heavy optional deps during import
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return SetBlockDecoder(model, tokenizer, config)


if __name__ == "__main__":
    print("Set Block Decoder initialized")
    print("This module implements Set Block Decoding for accelerated inference")
    print("Based on arxiv:2509.04185")
    print("Import and use with your Gemma model for faster generation")