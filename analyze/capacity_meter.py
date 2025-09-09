"""
Model Capacity and Memorization Analysis
Based on "How much do language models memorize?" (arxiv:2505.24832)
Measures ~3.6 bits per parameter capacity and tracks memorization vs generalization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json


@dataclass
class CapacityMetrics:
    """Metrics for model capacity analysis"""
    bits_per_param: float
    total_capacity_bits: int
    memorization_ratio: float
    generalization_ratio: float
    grokking_point: Optional[int]  # Step where grokking begins
    layer_capacities: Dict[str, float]


class BitStringDataset:
    """Random bitstring dataset for capacity measurement"""
    
    def __init__(self, num_samples: int, sequence_length: int, vocab_size: int = 2):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        
        # Generate random bitstrings
        self.data = torch.randint(0, vocab_size, (num_samples, sequence_length))
        
    def get_entropy(self) -> float:
        """Calculate dataset entropy in bits"""
        # For uniform random distribution
        return self.num_samples * self.sequence_length * np.log2(self.vocab_size)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.data[idx]
        }


class CapacityAnalyzer:
    """Analyze model capacity and memorization behavior"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.param_count = sum(p.numel() for p in model.parameters())
        self.training_history = []
        self.memorization_scores = []
        self.generalization_scores = []
        
    def estimate_capacity(self, bits_per_param: float = 3.6) -> int:
        """Estimate total model capacity in bits"""
        return int(self.param_count * bits_per_param)
        
    def measure_memorization(
        self, 
        train_dataset: BitStringDataset,
        test_dataset: Optional[BitStringDataset] = None
    ) -> Tuple[float, float]:
        """
        Measure memorization and generalization
        Returns: (memorization_score, generalization_score)
        """
        self.model.eval()
        
        # Measure on training data (memorization)
        train_loss = self._compute_dataset_loss(train_dataset)
        train_perplexity = torch.exp(train_loss)
        
        # Perfect memorization would give perplexity of 1
        # Random guessing gives perplexity of vocab_size
        memorization_score = 1.0 - (train_perplexity - 1) / (train_dataset.vocab_size - 1)
        memorization_score = max(0, min(1, memorization_score.item()))
        
        # Measure on test data (generalization) if available
        generalization_score = 0.0
        if test_dataset is not None:
            test_loss = self._compute_dataset_loss(test_dataset)
            test_perplexity = torch.exp(test_loss)
            generalization_score = 1.0 - (test_perplexity - 1) / (test_dataset.vocab_size - 1)
            generalization_score = max(0, min(1, generalization_score.item()))
            
        return memorization_score, generalization_score
        
    def _compute_dataset_loss(self, dataset) -> torch.Tensor:
        """Compute average loss over dataset"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(min(100, len(dataset))):  # Sample subset
                batch = dataset[i]
                input_ids = batch['input_ids'].unsqueeze(0)
                labels = batch['labels'].unsqueeze(0)
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                total_loss += outputs.loss
                num_batches += 1
                
        return total_loss / num_batches
        
    def track_training_step(
        self,
        step: int,
        loss: float,
        train_dataset: BitStringDataset,
        test_dataset: Optional[BitStringDataset] = None
    ):
        """Track metrics during training"""
        mem_score, gen_score = self.measure_memorization(train_dataset, test_dataset)
        
        self.memorization_scores.append(mem_score)
        self.generalization_scores.append(gen_score)
        
        self.training_history.append({
            'step': step,
            'loss': loss,
            'memorization': mem_score,
            'generalization': gen_score
        })
        
        # Detect grokking (when generalization starts improving)
        if len(self.generalization_scores) > 10:
            recent_improvement = (
                self.generalization_scores[-1] - 
                self.generalization_scores[-10]
            )
            if recent_improvement > 0.1:  # Significant improvement
                print(f"Potential grokking detected at step {step}")
                
    def analyze_layer_capacity(self) -> Dict[str, float]:
        """Analyze capacity distribution across layers"""
        layer_capacities = {}
        
        for name, param in self.model.named_parameters():
            if len(param.shape) >= 2:  # Focus on weight matrices
                # Estimate capacity based on parameter count and rank
                param_count = param.numel()
                
                # Compute effective rank via SVD
                with torch.no_grad():
                    if param.shape[0] * param.shape[1] < 10000:  # Small enough for SVD
                        U, S, V = torch.svd(param)
                        threshold = S[0] * 1e-3 if S[0] > 0 else 1e-10
                        effective_rank = torch.sum(S > threshold).item()
                        
                        # Capacity estimate: rank * log2(matrix_size)
                        capacity_bits = effective_rank * np.log2(param_count)
                    else:
                        # Approximate for large matrices
                        capacity_bits = param_count * 3.6
                        
                layer_capacities[name] = capacity_bits
                
        return layer_capacities
        
    def detect_double_descent(self) -> Optional[int]:
        """Detect double descent in training history"""
        if len(self.training_history) < 20:
            return None
            
        losses = [h['loss'] for h in self.training_history]
        
        # Simple detection: look for local maximum followed by decrease
        for i in range(10, len(losses) - 10):
            window_before = losses[i-10:i]
            window_after = losses[i:i+10]
            
            if (max(window_before) < losses[i] and 
                losses[i] > max(window_after) and
                np.mean(window_after) < np.mean(window_before)):
                return self.training_history[i]['step']
                
        return None
        
    def plot_capacity_analysis(self):
        """Visualize capacity and memorization analysis"""
        if not self.training_history:
            print("No training history to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = [h['step'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]
        mem_scores = [h['memorization'] for h in self.training_history]
        gen_scores = [h['generalization'] for h in self.training_history]
        
        # Plot 1: Loss curve
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark double descent if detected
        dd_point = self.detect_double_descent()
        if dd_point:
            axes[0, 0].axvline(x=dd_point, color='r', linestyle='--', 
                             label=f'Double Descent @ {dd_point}')
            axes[0, 0].legend()
            
        # Plot 2: Memorization vs Generalization
        axes[0, 1].plot(steps, mem_scores, 'r-', label='Memorization', linewidth=2)
        axes[0, 1].plot(steps, gen_scores, 'g-', label='Generalization', linewidth=2)
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Memorization vs Generalization')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Layer capacity distribution
        layer_caps = self.analyze_layer_capacity()
        if layer_caps:
            layer_names = list(layer_caps.keys())[:10]  # Top 10 layers
            capacities = [layer_caps[name] for name in layer_names]
            
            axes[1, 0].barh(range(len(layer_names)), capacities)
            axes[1, 0].set_yticks(range(len(layer_names)))
            axes[1, 0].set_yticklabels([n.split('.')[-1] for n in layer_names], 
                                      fontsize=8)
            axes[1, 0].set_xlabel('Capacity (bits)')
            axes[1, 0].set_title('Layer Capacity Distribution')
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
        # Plot 4: Grokking visualization
        if len(mem_scores) > 0 and len(gen_scores) > 0:
            axes[1, 1].scatter(mem_scores, gen_scores, c=steps, 
                             cmap='viridis', alpha=0.6, s=20)
            axes[1, 1].set_xlabel('Memorization Score')
            axes[1, 1].set_ylabel('Generalization Score')
            axes[1, 1].set_title('Memorization-Generalization Trajectory')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
            cbar.set_label('Training Step')
            
        plt.tight_layout()
        plt.savefig('visualizations/capacity_analysis.png', dpi=150)
        plt.show()
        
    def get_capacity_report(self) -> CapacityMetrics:
        """Generate comprehensive capacity report"""
        bits_per_param = 3.6
        total_capacity = self.estimate_capacity(bits_per_param)
        
        # Calculate average memorization and generalization
        avg_mem = np.mean(self.memorization_scores) if self.memorization_scores else 0
        avg_gen = np.mean(self.generalization_scores) if self.generalization_scores else 0
        
        # Detect grokking point
        grokking_point = None
        if len(self.generalization_scores) > 20:
            for i in range(10, len(self.generalization_scores)):
                if self.generalization_scores[i] > self.generalization_scores[i-10] + 0.1:
                    grokking_point = self.training_history[i]['step']
                    break
                    
        return CapacityMetrics(
            bits_per_param=bits_per_param,
            total_capacity_bits=total_capacity,
            memorization_ratio=avg_mem,
            generalization_ratio=avg_gen,
            grokking_point=grokking_point,
            layer_capacities=self.analyze_layer_capacity()
        )
        
    def save_analysis(self, filepath: str):
        """Save analysis results"""
        report = self.get_capacity_report()
        
        results = {
            'param_count': self.param_count,
            'bits_per_param': report.bits_per_param,
            'total_capacity_bits': report.total_capacity_bits,
            'memorization_ratio': report.memorization_ratio,
            'generalization_ratio': report.generalization_ratio,
            'grokking_point': report.grokking_point,
            'training_history': self.training_history,
            'layer_capacities': report.layer_capacities
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Analysis saved to {filepath}")


def run_capacity_experiment(model, num_samples=1000, sequence_length=128):
    """Run a complete capacity experiment"""
    
    print(f"Running capacity experiment with {num_samples} samples")
    
    # Create datasets
    train_dataset = BitStringDataset(num_samples, sequence_length)
    test_dataset = BitStringDataset(num_samples // 10, sequence_length)
    
    # Initialize analyzer
    analyzer = CapacityAnalyzer(model)
    
    # Get initial report
    report = analyzer.get_capacity_report()
    
    print(f"Model Parameters: {analyzer.param_count:,}")
    print(f"Estimated Capacity: {report.total_capacity_bits:,} bits")
    print(f"Dataset Entropy: {train_dataset.get_entropy():,.0f} bits")
    
    ratio = train_dataset.get_entropy() / report.total_capacity_bits
    print(f"Data/Capacity Ratio: {ratio:.2f}")
    
    if ratio > 1:
        print("Dataset exceeds model capacity - expect generalization")
    else:
        print("Model capacity exceeds dataset - expect memorization")
        
    return analyzer, train_dataset, test_dataset


if __name__ == "__main__":
    print("Capacity Meter initialized")
    print("This module analyzes model capacity and memorization based on arxiv:2505.24832")
    print("Import and use with your Gemma model for capacity analysis")