#!/usr/bin/env python3
"""
Basic example of MultiOmicsNet usage.

This script demonstrates the basic workflow for multi-omics integration
using simulated data.
"""

import numpy as np
import pandas as pd
import multiomicsnet as mon

def generate_example_data():
    """Generate example multi-omics data for demonstration."""
    np.random.seed(42)
    
    # Generate RNA-seq data
    n_genes, n_cells = 1000, 500
    rna_data = pd.DataFrame(
        np.random.negative_binomial(10, 0.3, (n_genes, n_cells)),
        index=[f"Gene_{i}" for i in range(n_genes)],
        columns=[f"Cell_{i}" for i in range(n_cells)]
    )
    
    rna_metadata = pd.DataFrame({
        'cell_id': [f"Cell_{i}" for i in range(n_cells)],
        'condition': ['control'] * 250 + ['treatment'] * 250,
        'cell_type': np.random.choice(['T_cell', 'B_cell', 'Monocyte'], n_cells)
    })
    
    # Generate ATAC-seq data
    n_peaks, n_cells_atac = 2000, 400
    atac_data = pd.DataFrame(
        np.random.binomial(1, 0.1, (n_peaks, n_cells_atac)),
        index=[f"Peak_{i}" for i in range(n_peaks)],
        columns=[f"Cell_{i}" for i in range(n_cells_atac)]
    )
    
    atac_metadata = pd.DataFrame({
        'cell_id': [f"Cell_{i}" for i in range(n_cells_atac)],
        'condition': ['control'] * 200 + ['treatment'] * 200,
        'cell_type': np.random.choice(['T_cell', 'B_cell', 'Monocyte'], n_cells_atac)
    })
    
    # Generate metabolomics data
    n_metabolites, n_samples = 100, 20
    metabolomics_data = pd.DataFrame(
        np.random.lognormal(0, 1, (n_metabolites, n_samples)),
        index=[f"Metabolite_{i}" for i in range(n_metabolites)],
        columns=[f"Sample_{i}" for i in range(n_samples)]
    )
    
    metabolomics_metadata = pd.DataFrame({
        'sample_id': [f"Sample_{i}" for i in range(n_samples)],
        'condition': ['control'] * 10 + ['treatment'] * 10
    })
    
    # Generate microbiome data
    n_taxa, n_samples_micro = 50, 20
    microbiome_data = pd.DataFrame(
        np.random.negative_binomial(50, 0.1, (n_taxa, n_samples_micro)),
        index=[f"Taxa_{i}" for i in range(n_taxa)],
        columns=[f"Sample_{i}" for i in range(n_samples_micro)]
    )
    
    microbiome_metadata = metabolomics_metadata.copy()
    
    return {
        'rna': (rna_data, rna_metadata),
        'atac': (atac_data, atac_metadata),
        'metabolomics': (metabolomics_data, metabolomics_metadata),
        'microbiome': (microbiome_data, microbiome_metadata)
    }

def main():
    """Run the basic MultiOmicsNet example."""
    print("MultiOmicsNet Basic Example")
    print("=" * 50)
    
    # Generate example data
    print("Generating example data...")
    data = generate_example_data()
    
    # Initialize integrator
    print("Initializing MultiOmicsNet integrator...")
    integrator = mon.MultiOmicsIntegrator(
        integration_method='hybrid',
        scenic_plus=True,
        scvi_integration=True,
        verbose=True
    )
    
    # Add data
    print("Adding multi-omics data...")
    integrator.add_rna_data(data['rna'][0], sample_metadata=data['rna'][1])
    integrator.add_atac_data(data['atac'][0], sample_metadata=data['atac'][1])
    integrator.add_metabolomics_data(data['metabolomics'][0], sample_metadata=data['metabolomics'][1])
    integrator.add_microbiome_data(data['microbiome'][0], sample_metadata=data['microbiome'][1])
    
    # Preprocess data
    print("Preprocessing data...")
    integrator.preprocess_data()
    
    # Build networks
    print("Building cross-layer networks...")
    networks = integrator.build_networks()
    
    # Compute metrics
    print("Computing network metrics...")
    metrics = integrator.compute_network_metrics()
    
    # Perform differential analysis
    print("Performing differential analysis...")
    diff_results = integrator.differential_analysis(
        condition_column='condition',
        control='control',
        treatment='treatment'
    )
    
    # Save results
    print("Saving results...")
    integrator.save_results('example_results/')
    
    print("Analysis completed successfully!")
    print(f"Networks built: {list(networks.keys())}")
    print(f"Metrics computed: {list(metrics.keys())}")
    print(f"Differential results: {list(diff_results.keys())}")

if __name__ == "__main__":
    main()

