"""
Implementation of "How much do language models memorize?" (arxiv:2505.24832)
Advanced capacity measurement and memorization analysis for Gemma-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random


@dataclass
class CapacityMeasurement:
    """Results from capacity measurement experiments"""
    model_name: str
    total_parameters: int
    theoretical_capacity_bits: float
    measured_capacity_bits: float
    memorization_threshold: int
    generalization_onset: int
    double_descent_point: Optional[int]
    layer_capacities: Dict[str, float]
    bitstring_results: List[Dict[str, Any]]


class ModelCapacityAnalyzer:
    """
    Measure model capacity and memorization behavior
    Based on frontier research in understanding LLM memorization
    """
    
    def __init__(self, model: nn.Module, tokenizer: Any, device: str = 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Calculate model parameters
        self.total_params = sum(p.numel() for p in model.parameters())
        self.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Theoretical capacity (3.6 bits per parameter for transformers)
        self.theoretical_capacity = self.total_params * 3.6
        
        # Storage for experimental results
        self.memorization_results = []
        self.generalization_results = []
        
    def measure_bitstring_memorization(self, 
                                      num_strings: int = 1000,
                                      string_length: int = 100,
                                      epochs: int = 10) -> Dict[str, Any]:
        """
        Measure model's ability to memorize random bitstrings
        Core experiment from the paper
        """
        
        print(f"Measuring memorization capacity with {num_strings} bitstrings of length {string_length}")
        
        # Generate random bitstrings
        bitstrings = self._generate_random_bitstrings(num_strings, string_length)
        
        # Tokenize bitstrings
        tokenized = [self.tokenizer.encode(bs, return_tensors='pt').to(self.device) 
                     for bs in bitstrings]
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        memorization_curve = []
        accuracy_per_epoch = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            # Training loop
            self.model.train()
            for tokens in tqdm(tokenized, desc=f"Epoch {epoch+1}/{epochs}"):
                if tokens.shape[1] < 2:
                    continue
                    
                # Prepare inputs and targets
                inputs = tokens[:, :-1]
                targets = tokens[:, 1:]
                
                # Forward pass
                outputs = self.model(inputs)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    targets.view(-1)
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.numel()
                
            # Calculate epoch metrics
            avg_loss = epoch_loss / len(tokenized)
            accuracy = correct_predictions / total_predictions
            
            memorization_curve.append({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': accuracy,
                'bits_memorized': self._calculate_bits_memorized(accuracy, num_strings, string_length)
            })
            
            accuracy_per_epoch.append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            
        # Analyze results
        results = {
            'num_strings': num_strings,
            'string_length': string_length,
            'total_bits': num_strings * string_length,
            'memorization_curve': memorization_curve,
            'final_accuracy': accuracy_per_epoch[-1],
            'bits_memorized': self._calculate_bits_memorized(accuracy_per_epoch[-1], num_strings, string_length),
            'memorization_efficiency': self._calculate_memorization_efficiency(memorization_curve)
        }
        
        self.memorization_results.append(results)
        return results
    
    def _generate_random_bitstrings(self, num_strings: int, length: int) -> List[str]:
        """Generate random bitstrings for memorization experiments"""
        bitstrings = []
        for _ in range(num_strings):
            bitstring = ''.join(random.choice('01') for _ in range(length))
            bitstrings.append(bitstring)
        return bitstrings
    
    def _calculate_bits_memorized(self, accuracy: float, num_strings: int, string_length: int) -> float:
        """Calculate effective bits memorized based on accuracy"""
        # Information-theoretic calculation
        if accuracy == 0:
            return 0
        elif accuracy == 1:
            return num_strings * string_length
        else:
            # Use entropy-based calculation
            entropy = -accuracy * np.log2(accuracy + 1e-10) - (1-accuracy) * np.log2(1-accuracy + 1e-10)
            effective_bits = num_strings * string_length * (1 - entropy)
            return effective_bits
            
    def _calculate_memorization_efficiency(self, memorization_curve: List[Dict]) -> float:
        """Calculate how efficiently the model memorizes"""
        if not memorization_curve:
            return 0.0
            
        # Area under the memorization curve
        epochs = [m['epoch'] for m in memorization_curve]
        accuracies = [m['accuracy'] for m in memorization_curve]
        
        # Trapezoidal integration
        efficiency = np.trapz(accuracies, epochs) / len(epochs)
        return efficiency
    
    def detect_grokking(self, 
                       train_data: torch.Tensor,
                       test_data: torch.Tensor,
                       max_steps: int = 10000) -> Dict[str, Any]:
        """
        Detect grokking phenomenon: sudden generalization after memorization
        """
        
        print("Detecting grokking behavior...")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        
        grokking_detected = False
        grokking_step = None
        
        for step in tqdm(range(max_steps)):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            
            train_output = self.model(train_data[:, :-1])
            train_logits = train_output.logits if hasattr(train_output, 'logits') else train_output
            train_loss = F.cross_entropy(
                train_logits.view(-1, train_logits.shape[-1]),
                train_data[:, 1:].view(-1)
            )
            
            train_loss.backward()
            optimizer.step()
            
            # Evaluation
            if step % 100 == 0:
                self.model.eval()
                with torch.no_grad():
                    # Train metrics
                    train_pred = torch.argmax(train_logits, dim=-1)
                    train_acc = (train_pred == train_data[:, 1:]).float().mean().item()
                    
                    # Test metrics
                    test_output = self.model(test_data[:, :-1])
                    test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output
                    test_loss = F.cross_entropy(
                        test_logits.view(-1, test_logits.shape[-1]),
                        test_data[:, 1:].view(-1)
                    )
                    test_pred = torch.argmax(test_logits, dim=-1)
                    test_acc = (test_pred == test_data[:, 1:]).float().mean().item()
                    
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                
                # Detect grokking: high train accuracy but sudden jump in test accuracy
                if len(test_accuracies) > 10:
                    recent_test_acc = test_accuracies[-5:]
                    old_test_acc = test_accuracies[-10:-5]
                    
                    if (np.mean(recent_test_acc) > 0.9 and 
                        np.mean(old_test_acc) < 0.5 and
                        train_acc > 0.95):
                        grokking_detected = True
                        grokking_step = step
                        print(f"Grokking detected at step {step}!")
                        
        return {
            'grokking_detected': grokking_detected,
            'grokking_step': grokking_step,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies
        }
    
    def measure_double_descent(self,
                              dataset_sizes: List[int],
                              model_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Measure double descent phenomenon
        Test loss first decreases, then increases, then decreases again
        """
        
        print("Measuring double descent...")
        
        results = {
            'dataset_sizes': dataset_sizes,
            'test_losses': [],
            'train_losses': [],
            'interpolation_threshold': None,
            'double_descent_detected': False
        }
        
        for dataset_size in tqdm(dataset_sizes, desc="Dataset sizes"):
            # Generate synthetic dataset
            data = torch.randint(0, 1000, (dataset_size, 100), device=self.device)
            
            # Split train/test
            train_size = int(0.8 * dataset_size)
            train_data = data[:train_size]
            test_data = data[train_size:]
            
            # Train model
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
            
            for _ in range(100):  # Fixed number of steps
                self.model.train()
                output = self.model(train_data[:, :-1])
                logits = output.logits if hasattr(output, 'logits') else output
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    train_data[:, 1:].view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Evaluate
            self.model.eval()
            with torch.no_grad():
                # Train loss
                train_output = self.model(train_data[:, :-1])
                train_logits = train_output.logits if hasattr(train_output, 'logits') else train_output
                train_loss = F.cross_entropy(
                    train_logits.view(-1, train_logits.shape[-1]),
                    train_data[:, 1:].view(-1)
                ).item()
                
                # Test loss
                test_output = self.model(test_data[:, :-1])
                test_logits = test_output.logits if hasattr(test_output, 'logits') else test_output
                test_loss = F.cross_entropy(
                    test_logits.view(-1, test_logits.shape[-1]),
                    test_data[:, 1:].view(-1)
                ).item()
                
            results['train_losses'].append(train_loss)
            results['test_losses'].append(test_loss)
            
        # Detect double descent
        test_losses = results['test_losses']
        if len(test_losses) > 5:
            # Find local minimum, then maximum, then decreasing trend
            for i in range(2, len(test_losses) - 2):
                if (test_losses[i-1] > test_losses[i] < test_losses[i+1] and  # Local min
                    max(test_losses[i+1:]) > test_losses[i] * 1.1):  # Subsequent increase
                    
                    # Check for final decrease
                    max_idx = i + 1 + test_losses[i+1:].index(max(test_losses[i+1:]))
                    if max_idx < len(test_losses) - 1:
                        if test_losses[-1] < test_losses[max_idx] * 0.9:
                            results['double_descent_detected'] = True
                            results['interpolation_threshold'] = dataset_sizes[max_idx]
                            break
                            
        return results
    
    def analyze_layer_capacity(self) -> Dict[str, float]:
        """
        Analyze capacity distribution across layers
        """
        
        layer_capacities = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                # Calculate layer capacity
                params = sum(p.numel() for p in module.parameters())
                capacity_bits = params * 3.6  # Theoretical capacity
                
                # Adjust based on layer type
                if 'attention' in name.lower():
                    # Attention layers have higher effective capacity
                    capacity_bits *= 1.2
                elif 'mlp' in name.lower() or 'ffn' in name.lower():
                    # MLP layers have slightly lower capacity
                    capacity_bits *= 0.9
                elif 'embed' in name.lower():
                    # Embedding layers are mainly lookup
                    capacity_bits *= 0.7
                    
                layer_capacities[name] = capacity_bits
                
        return layer_capacities
    
    def visualize_capacity_analysis(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of capacity analysis
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Memorization curve
        if self.memorization_results:
            latest_result = self.memorization_results[-1]
            curve = latest_result['memorization_curve']
            
            epochs = [m['epoch'] for m in curve]
            accuracies = [m['accuracy'] for m in curve]
            bits = [m['bits_memorized'] for m in curve]
            
            axes[0, 0].plot(epochs, accuracies, 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Memorization Accuracy')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(epochs, bits, 'g-', linewidth=2)
            axes[0, 1].axhline(y=self.theoretical_capacity, color='r', linestyle='--', label='Theoretical Capacity')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Bits Memorized')
            axes[0, 1].set_title('Effective Memorization')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
        # 2. Grokking visualization
        if hasattr(self, 'grokking_results'):
            steps = range(len(self.grokking_results['train_accuracies']))
            axes[0, 2].plot(steps, self.grokking_results['train_accuracies'], label='Train', linewidth=2)
            axes[0, 2].plot(steps, self.grokking_results['test_accuracies'], label='Test', linewidth=2)
            
            if self.grokking_results['grokking_detected']:
                grok_step = self.grokking_results['grokking_step'] // 100
                axes[0, 2].axvline(x=grok_step, color='r', linestyle='--', label='Grokking Point')
                
            axes[0, 2].set_xlabel('Steps (x100)')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Grokking Detection')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
        # 3. Double descent
        if hasattr(self, 'double_descent_results'):
            results = self.double_descent_results
            axes[1, 0].plot(results['dataset_sizes'], results['train_losses'], label='Train Loss', linewidth=2)
            axes[1, 0].plot(results['dataset_sizes'], results['test_losses'], label='Test Loss', linewidth=2)
            
            if results['interpolation_threshold']:
                axes[1, 0].axvline(x=results['interpolation_threshold'], color='r', linestyle='--', 
                                  label='Interpolation Threshold')
                                  
            axes[1, 0].set_xlabel('Dataset Size')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Double Descent Phenomenon')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
        # 4. Layer capacity distribution
        layer_capacities = self.analyze_layer_capacity()
        if layer_capacities:
            layers = list(layer_capacities.keys())[:20]  # Top 20 layers
            capacities = [layer_capacities[l] for l in layers]
            
            axes[1, 1].barh(range(len(layers)), capacities, color='teal')
            axes[1, 1].set_yticks(range(len(layers)))
            axes[1, 1].set_yticklabels([l.split('.')[-1][:15] for l in layers], fontsize=8)
            axes[1, 1].set_xlabel('Capacity (bits)')
            axes[1, 1].set_title('Layer-wise Capacity')
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
        # 5. Capacity utilization
        axes[1, 2].bar(['Theoretical', 'Measured'], 
                      [self.theoretical_capacity, 
                       self.memorization_results[-1]['bits_memorized'] if self.memorization_results else 0],
                      color=['blue', 'green'])
        axes[1, 2].set_ylabel('Bits')
        axes[1, 2].set_title('Capacity Utilization')
        axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Model Capacity Analysis: {self.total_params/1e6:.1f}M Parameters', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
        
    def generate_capacity_report(self) -> CapacityMeasurement:
        """
        Generate comprehensive capacity measurement report
        """
        
        # Compile all measurements
        measured_capacity = 0
        if self.memorization_results:
            measured_capacity = max(r['bits_memorized'] for r in self.memorization_results)
            
        # Find memorization threshold
        memorization_threshold = None
        generalization_onset = None
        
        if hasattr(self, 'grokking_results') and self.grokking_results['grokking_detected']:
            memorization_threshold = self.grokking_results['grokking_step']
            generalization_onset = memorization_threshold + 1000  # Estimate
            
        # Find double descent point
        double_descent_point = None
        if hasattr(self, 'double_descent_results'):
            double_descent_point = self.double_descent_results.get('interpolation_threshold')
            
        return CapacityMeasurement(
            model_name=self.model.__class__.__name__,
            total_parameters=self.total_params,
            theoretical_capacity_bits=self.theoretical_capacity,
            measured_capacity_bits=measured_capacity,
            memorization_threshold=memorization_threshold,
            generalization_onset=generalization_onset,
            double_descent_point=double_descent_point,
            layer_capacities=self.analyze_layer_capacity(),
            bitstring_results=self.memorization_results
        )