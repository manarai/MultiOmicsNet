"""
MultiOmicsNet: Cross-Layer Network Integration for Multi-Omics Data

A comprehensive algorithm for integrating single-cell RNA-seq, single-cell ATAC-seq, 
bulk metabolomics, and 16S rRNA sequencing data using cross-layer network integration approaches.
"""

__version__ = "0.1.0"
__author__ = "Manus AI"
__email__ = "contact@manus.ai"

from .core.integrator import MultiOmicsIntegrator
from .core.data_loader import DataLoader
from .networks.network_builder import NetworkBuilder
from .quantification.metrics import NetworkMetrics
from .differential.analysis import DifferentialAnalysis
from .visualization.plots import NetworkPlotter

__all__ = [
    "MultiOmicsIntegrator",
    "DataLoader", 
    "NetworkBuilder",
    "NetworkMetrics",
    "DifferentialAnalysis",
    "NetworkPlotter",
]

