# MultiOmicsNet User Guide

**Author**: Manus AI  
**Version**: 1.0  
**Date**: July 21, 2025

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Data Preparation](#data-preparation)
5. [Algorithm Overview](#algorithm-overview)
6. [Detailed Usage](#detailed-usage)
7. [Network Quantification](#network-quantification)
8. [Differential Analysis](#differential-analysis)
9. [Visualization](#visualization)
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)
12. [API Reference](#api-reference)
13. [Examples and Case Studies](#examples-and-case-studies)
14. [References](#references)

## Introduction

MultiOmicsNet is a comprehensive algorithm for integrating single-cell RNA-seq, single-cell ATAC-seq, bulk metabolomics, and 16S rRNA sequencing data using cross-layer network integration approaches. The algorithm implements a hybrid approach that combines inference-based (data-driven) and knowledge-based (curated interactions) integration methods to construct robust multi-omics networks that can be quantified and compared across different biological conditions.

### Key Features

MultiOmicsNet provides several innovative features that distinguish it from existing multi-omics integration tools. The algorithm leverages state-of-the-art methods including SCENIC+ for gene regulatory network inference and scVI/MultiVI for single-cell data integration, while providing comprehensive network quantification metrics and statistical methods for differential analysis.

The primary strength of MultiOmicsNet lies in its ability to construct cross-layer networks that connect molecular features across different omics layers. This is achieved through a combination of mechanistic relationships (such as gene-peak linkages from paired scRNA-seq and scATAC-seq data), pathway-based connections (using curated databases like KEGG and Reactome), and data-driven associations (computed from correlation and mutual information analyses).

The algorithm provides robust statistical frameworks for comparing networks across different biological conditions, enabling researchers to identify condition-specific network changes that may be associated with disease, aging, treatment responses, or other phenotypic differences. The implementation includes comprehensive uncertainty quantification through bootstrap sampling and permutation testing, ensuring that network predictions and differential analysis results are statistically sound.

### Scientific Background

The integration of multi-omics data represents one of the most challenging and important problems in modern systems biology. Traditional approaches often analyze each omics layer independently, missing crucial interactions between different molecular levels that drive biological processes. MultiOmicsNet addresses this limitation by constructing integrated networks that capture both intra-layer and inter-layer relationships.

The theoretical foundation of MultiOmicsNet is based on network biology principles, which recognize that biological systems are best understood as networks of interacting components rather than collections of individual molecules. The algorithm implements several key concepts from network theory, including centrality measures for identifying important nodes, modularity analysis for detecting functional communities, and differential network analysis for comparing network structures across conditions.

The cross-layer integration approach implemented in MultiOmicsNet is particularly innovative because it combines multiple types of molecular interactions within a unified framework. Gene-peak linkages capture regulatory relationships between transcription and chromatin accessibility, metabolite-gene associations reveal metabolic regulation of gene expression, and host-microbe interactions provide insights into the role of the microbiome in modulating host molecular profiles.

## Installation

### System Requirements

MultiOmicsNet requires Python 3.8 or higher and has been tested on Linux, macOS, and Windows systems. The algorithm is computationally intensive and benefits from systems with adequate memory (minimum 8GB RAM recommended, 16GB or more for large datasets) and multiple CPU cores for parallel processing.

### Installation Methods

#### Method 1: Installation from Source

The recommended installation method is to clone the repository and install in development mode, which allows for easy updates and customization:

```bash
# Clone the repository
git clone https://github.com/your-username/MultiOmicsNet.git
cd MultiOmicsNet

# Create and activate conda environment
conda create -n multiomicsnet python=3.8
conda activate multiomicsnet

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

#### Method 2: Direct Installation

For users who prefer a simpler installation process:

```bash
pip install multiomicsnet
```

### Dependency Management

MultiOmicsNet has several key dependencies that are automatically installed during the setup process. The most important dependencies include scanpy for single-cell analysis, scvi-tools for variational inference, networkx for network analysis, and various statistical and visualization libraries.

Some dependencies may require additional system-level packages. On Ubuntu/Debian systems, you may need to install:

```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev
```

On macOS, ensure you have Xcode command line tools installed:

```bash
xcode-select --install
```

### Verification

After installation, verify that MultiOmicsNet is working correctly:

```python
import multiomicsnet as mon
print(f"MultiOmicsNet version: {mon.__version__}")

# Run basic functionality test
integrator = mon.MultiOmicsIntegrator()
print("Installation successful!")
```

## Quick Start

This section provides a minimal example to get you started with MultiOmicsNet quickly. For more detailed explanations, see the subsequent sections.

### Basic Workflow

```python
import multiomicsnet as mon
import pandas as pd
import numpy as np

# Initialize the integrator
integrator = mon.MultiOmicsIntegrator(
    integration_method='hybrid',
    scenic_plus=True,
    scvi_integration=True,
    verbose=True
)

# Load your data (replace with your actual data files)
rna_data = pd.read_csv('rna_counts.csv', index_col=0)
atac_data = pd.read_csv('atac_peaks.csv', index_col=0)
metabolomics_data = pd.read_csv('metabolites.csv', index_col=0)
microbiome_data = pd.read_csv('microbiome_counts.csv', index_col=0)

# Add data to integrator
integrator.add_rna_data(rna_data, sample_metadata=rna_metadata)
integrator.add_atac_data(atac_data, sample_metadata=atac_metadata)
integrator.add_metabolomics_data(metabolomics_data, sample_metadata=metabolomics_metadata)
integrator.add_microbiome_data(microbiome_data, sample_metadata=microbiome_metadata)

# Preprocess data
integrator.preprocess_data()

# Build networks
networks = integrator.build_networks()

# Compute metrics
metrics = integrator.compute_network_metrics()

# Perform differential analysis
diff_results = integrator.differential_analysis(
    condition_column='treatment',
    control='control',
    treatment='treated'
)

# Save results
integrator.save_results('results/')
```

This basic workflow demonstrates the core functionality of MultiOmicsNet. The algorithm handles the complex integration process internally, allowing users to focus on their biological questions rather than technical implementation details.


## Data Preparation

Proper data preparation is crucial for successful multi-omics integration. MultiOmicsNet accepts data in standard formats and provides comprehensive preprocessing capabilities, but understanding the expected data structure and quality requirements will help ensure optimal results.

### Single-Cell RNA-seq Data

Single-cell RNA-seq data should be provided as a count matrix with genes as rows and cells as columns. The algorithm accepts both pandas DataFrames and AnnData objects, with AnnData being preferred for large datasets due to its efficient storage and metadata handling capabilities.

The count matrix should contain raw or minimally processed counts rather than heavily normalized data, as MultiOmicsNet implements its own normalization procedures optimized for network construction. Quality control metrics should be included in the sample metadata, including the number of detected genes per cell, total UMI counts, and mitochondrial gene percentage.

Cell type annotations, while not required, significantly improve the integration process by allowing the algorithm to account for cell type-specific expression patterns. Batch information is also important for correcting technical variation that could confound biological signals in the network analysis.

Sample metadata should include at minimum a condition column that specifies the biological conditions being compared (e.g., 'young' vs 'aged', 'control' vs 'treatment'). Additional covariates such as sex, age, or technical batch can be included and will be considered during preprocessing and analysis.

Gene metadata is optional but recommended, particularly for datasets where gene symbols may be ambiguous or where additional gene annotations (such as pathway membership or functional categories) are available. This information can enhance the knowledge-based integration components of the algorithm.

### Single-Cell ATAC-seq Data

ATAC-seq data should be provided as a peak-by-cell matrix, where peaks represent accessible chromatin regions and values indicate accessibility counts or binary accessibility calls. The algorithm can handle both count-based and binary representations, with appropriate preprocessing applied based on the data type specified.

Peak metadata is particularly important for ATAC-seq data and should include genomic coordinates (chromosome, start, end positions) for each peak. This information is essential for linking peaks to genes and for integrating with other genomic datasets. Additional annotations such as peak type (promoter, enhancer, etc.) or transcription factor binding site predictions can enhance the analysis.

The algorithm implements specialized preprocessing for ATAC-seq data, including peak filtering based on accessibility frequency, cell quality filtering based on the number of accessible peaks, and normalization procedures that account for the sparse and binary nature of chromatin accessibility data.

For optimal integration with RNA-seq data, ATAC-seq samples should be matched or highly similar to the RNA-seq samples in terms of cell types and experimental conditions. While perfect matching is not required, significant differences in cell composition between the two modalities can complicate the integration process.

### Bulk Metabolomics Data

Metabolomics data should be provided as a feature table with metabolites as rows and samples as columns. Values typically represent metabolite abundances or concentrations, and the algorithm can handle various scales of measurement including raw abundances, log-transformed values, or normalized concentrations.

Metabolite identification and annotation are crucial for the knowledge-based integration components of MultiOmicsNet. The metabolite metadata should include identifiers that can be mapped to pathway databases such as KEGG, HMDB, or MetaCyc. Common identifiers include KEGG compound IDs, HMDB accession numbers, or chemical names that can be matched to database entries.

Quality control for metabolomics data should address missing values, which are common in metabolomics datasets due to detection limits and technical variability. MultiOmicsNet provides several imputation strategies, but the choice of method should be informed by the nature of the missing data (missing at random vs. missing not at random).

Sample metadata for metabolomics should match the experimental design of the other omics layers, with consistent condition labels and covariate information. Technical replicates should be identified and can be averaged during preprocessing, while biological replicates should be maintained as separate samples for statistical analysis.

### 16S rRNA Microbiome Data

Microbiome data should be provided as an operational taxonomic unit (OTU) or amplicon sequence variant (ASV) table with taxa as rows and samples as columns. Values represent read counts or relative abundances for each taxon in each sample.

Taxonomic classification information is essential and should be provided in the taxonomy metadata. This should include classifications at multiple taxonomic levels (phylum, class, order, family, genus, species) to enable analysis at different levels of resolution. The algorithm can work with partial classifications where species-level identification is not available.

Microbiome data requires specialized preprocessing to address the compositional nature of the data and the high variability in sequencing depth across samples. MultiOmicsNet implements centered log-ratio (CLR) transformation and other compositional data analysis methods to handle these challenges appropriately.

Sample metadata should include information about sample collection, storage, and processing procedures, as these can significantly impact microbiome composition. Matching of microbiome samples to host samples (for RNA-seq, ATAC-seq, and metabolomics) should be clearly indicated in the metadata.

### Data Quality Assessment

Before proceeding with integration, it is important to assess the quality of each omics dataset individually. MultiOmicsNet provides built-in quality control functions, but users should also perform independent quality assessment using specialized tools for each data type.

For single-cell data, key quality metrics include the number of detected features per cell, total counts per cell, and the percentage of counts from mitochondrial genes (for RNA-seq) or the TSS enrichment score (for ATAC-seq). Cells with extremely low or high values for these metrics may represent low-quality cells or doublets that should be filtered.

For bulk data (metabolomics and microbiome), important considerations include the number of detected features, the distribution of feature abundances, and the presence of batch effects or other technical confounders. Principal component analysis and other dimensionality reduction techniques can help identify potential issues with data quality or experimental design.

## Algorithm Overview

MultiOmicsNet implements a sophisticated multi-stage algorithm that integrates diverse omics data types through a combination of data-driven and knowledge-based approaches. Understanding the algorithmic framework will help users make informed decisions about parameter settings and interpret results appropriately.

### Theoretical Framework

The algorithm is based on the principle that biological systems are best understood as networks of interacting molecular components. Rather than analyzing each omics layer in isolation, MultiOmicsNet constructs integrated networks that capture relationships both within and between different molecular levels.

The theoretical foundation draws from several areas of computational biology and network science. Graph theory provides the mathematical framework for representing and analyzing network structures. Information theory contributes methods for quantifying associations between variables and detecting significant relationships. Statistical inference provides the tools for assessing the significance of network features and comparing networks across conditions.

The cross-layer integration approach is based on the recognition that different omics layers are not independent but are connected through well-understood biological mechanisms. Gene expression is regulated by chromatin accessibility, metabolic processes are controlled by enzyme expression levels, and the microbiome influences host metabolism through the production of bioactive compounds.

### Multi-Stage Processing Pipeline

The MultiOmicsNet algorithm consists of five main stages, each designed to address specific challenges in multi-omics integration while building toward the final integrated network representation.

**Stage 1: Data Preprocessing and Quality Control** addresses the technical challenges associated with integrating heterogeneous data types. Each omics layer undergoes specialized preprocessing appropriate for its data characteristics. Single-cell RNA-seq data is normalized using size factor normalization and log transformation, followed by highly variable gene selection. Single-cell ATAC-seq data undergoes peak filtering and binary transformation. Metabolomics data is log-transformed and scaled, with missing value imputation as needed. Microbiome data is transformed using centered log-ratio transformation to address compositional constraints.

Batch effect correction is applied using appropriate methods for each data type. For single-cell data, the algorithm leverages the scVI framework, which uses variational autoencoders to learn batch-corrected latent representations. For bulk data, traditional batch correction methods such as ComBat or linear model-based approaches are employed.

**Stage 2: Intra-Layer Network Construction** builds networks within each omics layer using methods appropriate for the data characteristics and biological interpretation. Gene co-expression networks are constructed using correlation-based methods, with statistical significance testing to identify robust associations. Chromatin accessibility networks capture co-accessibility patterns that may reflect shared regulatory mechanisms.

Metabolite correlation networks identify metabolites that show coordinated abundance patterns, potentially reflecting shared metabolic pathways or regulatory mechanisms. Microbial co-abundance networks capture ecological relationships between different microbial taxa, including potential competitive or cooperative interactions.

The choice of network construction method for each layer is informed by the data characteristics and biological interpretation. Correlation-based methods are appropriate for continuous data with approximately normal distributions, while mutual information-based methods can capture non-linear relationships. For sparse data such as single-cell measurements, specialized methods that account for zero-inflation and technical noise are employed.

**Stage 3: Cross-Layer Network Integration** represents the core innovation of MultiOmicsNet, connecting networks across different omics layers through multiple types of molecular relationships. Gene-peak linkages are established using both distance-based methods (linking peaks to nearby genes) and correlation-based methods (identifying peaks and genes with correlated activity patterns).

Gene-metabolite associations are established through pathway mapping using curated databases such as KEGG and Reactome, supplemented by correlation analysis to identify novel associations not captured in existing databases. Host-microbe interactions are inferred through correlation analysis between microbial abundances and host molecular profiles, with additional consideration of known metabolic capabilities of different microbial taxa.

The integration process uses a hybrid approach that combines knowledge-based and inference-based methods. Knowledge-based integration leverages curated databases and prior biological knowledge to establish connections with high confidence. Inference-based integration uses statistical methods to identify novel associations directly from the data, providing opportunities to discover previously unknown relationships.

**Stage 4: Network Quantification and Analysis** computes comprehensive metrics to characterize the integrated network structure. Node-level metrics include various centrality measures that identify important nodes in the network, clustering coefficients that measure local connectivity patterns, and participation coefficients that quantify how nodes connect across different omics layers.

Network-level metrics characterize global properties of the integrated network, including density, modularity, small-worldness, and efficiency measures. These metrics provide insights into the overall organization of the multi-omics system and can be compared across different conditions or experimental settings.

Multi-layer network metrics specifically address the multi-omics nature of the integrated network, quantifying properties such as inter-layer connectivity, layer similarity, and the extent to which different omics layers contribute to the overall network structure.

**Stage 5: Differential Network Analysis** implements statistical methods to identify and quantify differences in network structure between different biological conditions. The algorithm constructs condition-specific networks and compares them using multiple approaches, including edge-wise comparisons, node-level comparisons, and global network property comparisons.

Statistical significance is assessed using permutation testing, which provides robust p-values without making strong distributional assumptions. Multiple testing correction is applied to control the false discovery rate across the many comparisons involved in network analysis. Effect sizes are computed to quantify the magnitude of differences and help prioritize the most biologically relevant changes.

### Integration Methods

MultiOmicsNet implements three primary integration strategies that can be used individually or in combination depending on the research question and data availability.

**Inference-Based Integration** relies entirely on statistical associations computed directly from the data. This approach is data-driven and can identify novel relationships not captured in existing biological databases. Correlation analysis, mutual information, and other association measures are used to identify significant relationships between features across different omics layers.

The advantage of inference-based integration is its ability to discover new biological relationships and its applicability to any combination of omics data types. However, it may be sensitive to technical noise and confounding factors, and the biological interpretation of inferred associations may not always be clear.

**Knowledge-Based Integration** leverages curated biological databases and prior knowledge to establish connections between omics layers. Pathway databases such as KEGG, Reactome, and MetaCyc provide information about known metabolic and regulatory relationships. Protein-protein interaction databases and transcription factor binding site databases contribute additional layers of biological knowledge.

The advantage of knowledge-based integration is its biological interpretability and robustness to technical noise. However, it is limited by the completeness and accuracy of existing databases and may miss novel biological relationships not yet captured in the literature.

**Hybrid Integration** combines both inference-based and knowledge-based approaches to leverage the strengths of each method while mitigating their individual limitations. Known biological relationships provide a foundation of high-confidence connections, while data-driven associations can identify novel relationships and validate or refine existing knowledge.

The hybrid approach implemented in MultiOmicsNet uses a weighted combination strategy where knowledge-based connections receive higher confidence scores, but data-driven associations that are not contradicted by existing knowledge are also included in the integrated network. This approach provides the most comprehensive and robust integration while maintaining biological interpretability.

### Statistical Framework

The statistical framework underlying MultiOmicsNet is designed to provide robust inference while accounting for the multiple testing challenges inherent in network analysis. The algorithm implements several key statistical concepts to ensure that results are reliable and interpretable.

**Significance Testing** is performed at multiple levels, including individual edge significance, node-level significance, and global network significance. For individual edges, the algorithm computes p-values based on the null hypothesis of no association, using appropriate test statistics for different data types and association measures.

**Multiple Testing Correction** is essential given the large number of potential edges in multi-omics networks. The algorithm implements several correction methods, including the Benjamini-Hochberg false discovery rate (FDR) control and more conservative family-wise error rate (FWER) control methods. The choice of correction method depends on the research question and the desired balance between sensitivity and specificity.

**Effect Size Estimation** provides information about the magnitude of associations and differences, complementing significance testing with practical significance assessment. Effect sizes are particularly important in network analysis because statistical significance can be achieved with very small effect sizes in large datasets, but such associations may not be biologically meaningful.

**Uncertainty Quantification** is implemented through bootstrap sampling and permutation testing to provide confidence intervals for network metrics and differential analysis results. This allows users to assess the reliability of their findings and make informed decisions about which results to pursue in follow-up studies.

The statistical framework is designed to be conservative, prioritizing the reliability of results over the discovery of marginal associations. This approach helps ensure that the networks and differential analysis results represent robust biological signals rather than technical artifacts or statistical noise.

