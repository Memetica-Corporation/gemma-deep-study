"""
Gemma-3 Architecture Deep Analysis Framework
Frontier-level tools for understanding transformer architectures at scale
"""

from .core import GemmaArchitecture, LayerAnalyzer, AttentionMechanism
from .probing import ActivationProbe, GradientTracker, RepresentationAnalyzer
from .visualization import InteractiveVisualizer, LayerDynamicsPlotter
from .optimization import MetalOptimizer, QuantizationAnalyzer

__version__ = "0.1.0"
__author__ = "Fidelic AI Research"

__all__ = [
    "GemmaArchitecture",
    "LayerAnalyzer", 
    "AttentionMechanism",
    "ActivationProbe",
    "GradientTracker",
    "RepresentationAnalyzer",
    "InteractiveVisualizer",
    "LayerDynamicsPlotter",
    "MetalOptimizer",
    "QuantizationAnalyzer"
]