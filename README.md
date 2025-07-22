# MultiOmicsNet: Cross-Layer Network Integration for Multi-Omics Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive algorithm for integrating single-cell RNA-seq, single-cell ATAC-seq, bulk metabolomics, and 16S rRNA sequencing data using cross-layer network integration approaches.

## Overview

MultiOmicsNet implements a hybrid approach combining inference-based (data-driven) and knowledge-based (curated interactions) integration methods to construct robust multi-omics networks. The algorithm enables quantification of network differences across biological conditions such as aging, disease states, and microbiome variations.

### Key Features

- **Multi-modal Integration**: Seamlessly integrates scRNA-seq, scATAC-seq, metabolomics, and microbiome data
- **Cross-layer Networks**: Constructs networks connecting molecular features across different omics layers
- **SCENIC+ Integration**: Leverages SCENIC+ for gene regulatory network inference
- **scVI Framework**: Uses scVI/MultiVI for robust single-cell data integration
- **Network Quantification**: Comprehensive metrics for network comparison and analysis
- **Differential Analysis**: Statistical methods to identify network differences across conditions
- **Uncertainty Quantification**: Confidence intervals and significance testing for network predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MultiOmicsNet.git
cd MultiOmicsNet

# Create conda environment
conda create -n multiomicsnet python=3.8
conda activate multiomicsnet

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
import multiomicsnet as mon
import pandas as pd
import numpy as np

# Load your multi-omics data
rna_data = pd.read_csv('data/rna_counts.csv', index_col=0)
atac_data = pd.read_csv('data/atac_peaks.csv', index_col=0)
metabolomics_data = pd.read_csv('data/metabolites.csv', index_col=0)
microbiome_data = pd.read_csv('data/microbiome_counts.csv', index_col=0)

# Initialize MultiOmicsNet
integrator = mon.MultiOmicsIntegrator()

# Add data layers
integrator.add_rna_data(rna_data)
integrator.add_atac_data(atac_data)
integrator.add_metabolomics_data(metabolomics_data)
integrator.add_microbiome_data(microbiome_data)

# Preprocess data
integrator.preprocess_data()

# Construct cross-layer networks
networks = integrator.build_networks(method='hybrid')

# Quantify network properties
metrics = integrator.compute_network_metrics(networks)

# Perform differential analysis
diff_results = integrator.differential_analysis(
    condition_column='treatment',
    control='control',
    treatment='treated'
)
```

## Repository Structure

```
MultiOmicsNet/
├── src/multiomicsnet/          # Main package source code
│   ├── __init__.py
│   ├── core/                   # Core integration algorithms
│   ├── preprocessing/          # Data preprocessing modules
│   ├── networks/              # Network construction methods
│   ├── quantification/        # Network quantification metrics
│   ├── differential/          # Differential analysis methods
│   ├── visualization/         # Plotting and visualization
│   └── utils/                 # Utility functions
├── notebooks/                 # Example Jupyter notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_network_construction.ipynb
│   ├── 03_differential_analysis.ipynb
│   └── 04_visualization_examples.ipynb
├── data/                      # Example datasets
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── examples/                  # Example scripts
└── requirements.txt           # Package dependencies
```

## Example Notebooks

1. **[Data Preprocessing](notebooks/01_data_preprocessing.ipynb)**: Load and preprocess multi-omics data
2. **[Network Construction](notebooks/02_network_construction.ipynb)**: Build cross-layer networks
3. **[Differential Analysis](notebooks/03_differential_analysis.ipynb)**: Compare networks across conditions
4. **[Visualization](notebooks/04_visualization_examples.ipynb)**: Create publication-ready plots

## Methods

### Integration Approaches

- **scVI/MultiVI**: Variational inference for single-cell multi-omics integration
- **SCENIC+**: Gene regulatory network inference from paired scRNA-seq and scATAC-seq
- **Knowledge-based**: KEGG, Reactome, and MetaCyc pathway integration
- **Inference-based**: Correlation, mutual information, and distance-based associations

### Network Quantification

- **Node-level metrics**: Centrality measures, clustering coefficients
- **Network-level metrics**: Modularity, path length, small-worldness
- **Multi-layer metrics**: Inter-layer connectivity, multiplex participation
- **Statistical testing**: Permutation tests, bootstrap confidence intervals

### Differential Analysis

- **Network comparison**: Edge-wise and global network differences
- **Statistical testing**: Multiple testing correction, effect size estimation
- **Uncertainty quantification**: Confidence intervals, significance testing

## Citation

If you use MultiOmicsNet in your research, please cite:

```bibtex
@software{multiomicsnet2025,
  title={MultiOmicsNet: Cross-Layer Network Integration for Multi-Omics Data},
  author={Manarai},
  year={2025},
  url={https://github.com/your-username/MultiOmicsNet}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Support

- **Documentation**: [https://multiomicsnet.readthedocs.io](https://multiomicsnet.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/MultiOmicsNet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/MultiOmicsNet/discussions)

