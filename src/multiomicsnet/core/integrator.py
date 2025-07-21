"""
Main MultiOmicsIntegrator class for cross-layer network integration.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path

from ..preprocessing.rna_processor import RNAProcessor
from ..preprocessing.atac_processor import ATACProcessor
from ..preprocessing.metabolomics_processor import MetabolomicsProcessor
from ..preprocessing.microbiome_processor import MicrobiomeProcessor
from ..networks.network_builder import NetworkBuilder
from ..quantification.metrics import NetworkMetrics
from ..differential.analysis import DifferentialAnalysis
from ..visualization.plots import NetworkPlotter
from ..utils.validation import DataValidator


class MultiOmicsIntegrator:
    """
    Main class for multi-omics cross-layer network integration.
    
    This class provides a unified interface for integrating single-cell RNA-seq,
    single-cell ATAC-seq, bulk metabolomics, and 16S rRNA sequencing data using
    cross-layer network integration approaches.
    
    Parameters
    ----------
    integration_method : str, default='hybrid'
        Method for network integration. Options: 'hybrid', 'inference', 'knowledge'
    scenic_plus : bool, default=True
        Whether to use SCENIC+ for gene regulatory network inference
    scvi_integration : bool, default=True
        Whether to use scVI/MultiVI for single-cell integration
    random_state : int, default=42
        Random state for reproducibility
    n_jobs : int, default=-1
        Number of parallel jobs to run
    verbose : bool, default=True
        Whether to print progress messages
    """
    
    def __init__(
        self,
        integration_method: str = 'hybrid',
        scenic_plus: bool = True,
        scvi_integration: bool = True,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        self.integration_method = integration_method
        self.scenic_plus = scenic_plus
        self.scvi_integration = scvi_integration
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Initialize data containers
        self.data = {}
        self.processed_data = {}
        self.networks = {}
        self.metrics = {}
        self.differential_results = {}
        
        # Initialize processors
        self.rna_processor = RNAProcessor(random_state=random_state, verbose=verbose)
        self.atac_processor = ATACProcessor(random_state=random_state, verbose=verbose)
        self.metabolomics_processor = MetabolomicsProcessor(random_state=random_state, verbose=verbose)
        self.microbiome_processor = MicrobiomeProcessor(random_state=random_state, verbose=verbose)
        
        # Initialize analysis modules
        self.network_builder = NetworkBuilder(
            integration_method=integration_method,
            scenic_plus=scenic_plus,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.network_metrics = NetworkMetrics(verbose=verbose)
        self.differential_analysis = DifferentialAnalysis(random_state=random_state, verbose=verbose)
        self.plotter = NetworkPlotter()
        
        # Initialize validator
        self.validator = DataValidator()
        
        if self.verbose:
            print(f"MultiOmicsIntegrator initialized with {integration_method} integration method")
    
    def add_rna_data(
        self,
        data: Union[pd.DataFrame, sc.AnnData],
        sample_metadata: Optional[pd.DataFrame] = None,
        gene_metadata: Optional[pd.DataFrame] = None,
        data_type: str = 'counts'
    ) -> None:
        """
        Add single-cell RNA-seq data.
        
        Parameters
        ----------
        data : pd.DataFrame or AnnData
            Gene expression count matrix (genes x cells) or AnnData object
        sample_metadata : pd.DataFrame, optional
            Sample metadata with cell information
        gene_metadata : pd.DataFrame, optional
            Gene metadata with gene annotations
        data_type : str, default='counts'
            Type of data: 'counts', 'normalized', 'log_normalized'
        """
        if self.verbose:
            print("Adding RNA-seq data...")
        
        # Validate data
        self.validator.validate_rna_data(data, sample_metadata, gene_metadata)
        
        self.data['rna'] = {
            'data': data,
            'sample_metadata': sample_metadata,
            'gene_metadata': gene_metadata,
            'data_type': data_type
        }
        
        if self.verbose:
            if isinstance(data, pd.DataFrame):
                print(f"RNA data added: {data.shape[0]} genes x {data.shape[1]} cells")
            else:
                print(f"RNA data added: {data.n_vars} genes x {data.n_obs} cells")
    
    def add_atac_data(
        self,
        data: Union[pd.DataFrame, sc.AnnData],
        sample_metadata: Optional[pd.DataFrame] = None,
        peak_metadata: Optional[pd.DataFrame] = None,
        data_type: str = 'counts'
    ) -> None:
        """
        Add single-cell ATAC-seq data.
        
        Parameters
        ----------
        data : pd.DataFrame or AnnData
            Peak accessibility count matrix (peaks x cells) or AnnData object
        sample_metadata : pd.DataFrame, optional
            Sample metadata with cell information
        peak_metadata : pd.DataFrame, optional
            Peak metadata with genomic coordinates
        data_type : str, default='counts'
            Type of data: 'counts', 'normalized', 'binary'
        """
        if self.verbose:
            print("Adding ATAC-seq data...")
        
        # Validate data
        self.validator.validate_atac_data(data, sample_metadata, peak_metadata)
        
        self.data['atac'] = {
            'data': data,
            'sample_metadata': sample_metadata,
            'peak_metadata': peak_metadata,
            'data_type': data_type
        }
        
        if self.verbose:
            if isinstance(data, pd.DataFrame):
                print(f"ATAC data added: {data.shape[0]} peaks x {data.shape[1]} cells")
            else:
                print(f"ATAC data added: {data.n_vars} peaks x {data.n_obs} cells")
    
    def add_metabolomics_data(
        self,
        data: pd.DataFrame,
        sample_metadata: Optional[pd.DataFrame] = None,
        metabolite_metadata: Optional[pd.DataFrame] = None,
        data_type: str = 'abundance'
    ) -> None:
        """
        Add bulk metabolomics data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Metabolite abundance matrix (metabolites x samples)
        sample_metadata : pd.DataFrame, optional
            Sample metadata
        metabolite_metadata : pd.DataFrame, optional
            Metabolite annotations and pathway information
        data_type : str, default='abundance'
            Type of data: 'abundance', 'log_abundance', 'normalized'
        """
        if self.verbose:
            print("Adding metabolomics data...")
        
        # Validate data
        self.validator.validate_metabolomics_data(data, sample_metadata, metabolite_metadata)
        
        self.data['metabolomics'] = {
            'data': data,
            'sample_metadata': sample_metadata,
            'metabolite_metadata': metabolite_metadata,
            'data_type': data_type
        }
        
        if self.verbose:
            print(f"Metabolomics data added: {data.shape[0]} metabolites x {data.shape[1]} samples")
    
    def add_microbiome_data(
        self,
        data: pd.DataFrame,
        sample_metadata: Optional[pd.DataFrame] = None,
        taxonomy_metadata: Optional[pd.DataFrame] = None,
        data_type: str = 'counts'
    ) -> None:
        """
        Add 16S rRNA microbiome data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Microbial abundance matrix (taxa x samples)
        sample_metadata : pd.DataFrame, optional
            Sample metadata
        taxonomy_metadata : pd.DataFrame, optional
            Taxonomic classifications and annotations
        data_type : str, default='counts'
            Type of data: 'counts', 'relative_abundance', 'clr_transformed'
        """
        if self.verbose:
            print("Adding microbiome data...")
        
        # Validate data
        self.validator.validate_microbiome_data(data, sample_metadata, taxonomy_metadata)
        
        self.data['microbiome'] = {
            'data': data,
            'sample_metadata': sample_metadata,
            'taxonomy_metadata': taxonomy_metadata,
            'data_type': data_type
        }
        
        if self.verbose:
            print(f"Microbiome data added: {data.shape[0]} taxa x {data.shape[1]} samples")
    
    def preprocess_data(
        self,
        rna_params: Optional[Dict] = None,
        atac_params: Optional[Dict] = None,
        metabolomics_params: Optional[Dict] = None,
        microbiome_params: Optional[Dict] = None
    ) -> None:
        """
        Preprocess all added data types.
        
        Parameters
        ----------
        rna_params : dict, optional
            Parameters for RNA-seq preprocessing
        atac_params : dict, optional
            Parameters for ATAC-seq preprocessing
        metabolomics_params : dict, optional
            Parameters for metabolomics preprocessing
        microbiome_params : dict, optional
            Parameters for microbiome preprocessing
        """
        if self.verbose:
            print("Preprocessing multi-omics data...")
        
        # Set default parameters
        rna_params = rna_params or {}
        atac_params = atac_params or {}
        metabolomics_params = metabolomics_params or {}
        microbiome_params = microbiome_params or {}
        
        # Preprocess RNA-seq data
        if 'rna' in self.data:
            if self.verbose:
                print("Preprocessing RNA-seq data...")
            self.processed_data['rna'] = self.rna_processor.preprocess(
                self.data['rna'], **rna_params
            )
        
        # Preprocess ATAC-seq data
        if 'atac' in self.data:
            if self.verbose:
                print("Preprocessing ATAC-seq data...")
            self.processed_data['atac'] = self.atac_processor.preprocess(
                self.data['atac'], **atac_params
            )
        
        # Preprocess metabolomics data
        if 'metabolomics' in self.data:
            if self.verbose:
                print("Preprocessing metabolomics data...")
            self.processed_data['metabolomics'] = self.metabolomics_processor.preprocess(
                self.data['metabolomics'], **metabolomics_params
            )
        
        # Preprocess microbiome data
        if 'microbiome' in self.data:
            if self.verbose:
                print("Preprocessing microbiome data...")
            self.processed_data['microbiome'] = self.microbiome_processor.preprocess(
                self.data['microbiome'], **microbiome_params
            )
        
        if self.verbose:
            print("Data preprocessing completed!")
    
    def build_networks(
        self,
        method: Optional[str] = None,
        network_params: Optional[Dict] = None
    ) -> Dict:
        """
        Build cross-layer networks from preprocessed data.
        
        Parameters
        ----------
        method : str, optional
            Integration method to use. If None, uses the method specified during initialization
        network_params : dict, optional
            Additional parameters for network construction
        
        Returns
        -------
        dict
            Dictionary containing constructed networks
        """
        if method is None:
            method = self.integration_method
        
        if self.verbose:
            print(f"Building cross-layer networks using {method} method...")
        
        # Check if data has been preprocessed
        if not self.processed_data:
            raise ValueError("Data must be preprocessed before building networks. Call preprocess_data() first.")
        
        network_params = network_params or {}
        
        # Build networks
        self.networks = self.network_builder.build_networks(
            self.processed_data,
            method=method,
            **network_params
        )
        
        if self.verbose:
            print(f"Networks built successfully! Available networks: {list(self.networks.keys())}")
        
        return self.networks
    
    def compute_network_metrics(
        self,
        networks: Optional[Dict] = None,
        metrics_list: Optional[List[str]] = None
    ) -> Dict:
        """
        Compute network quantification metrics.
        
        Parameters
        ----------
        networks : dict, optional
            Networks to analyze. If None, uses self.networks
        metrics_list : list, optional
            List of metrics to compute. If None, computes all available metrics
        
        Returns
        -------
        dict
            Dictionary containing computed metrics
        """
        if networks is None:
            networks = self.networks
        
        if not networks:
            raise ValueError("No networks available. Build networks first using build_networks().")
        
        if self.verbose:
            print("Computing network metrics...")
        
        self.metrics = self.network_metrics.compute_metrics(
            networks,
            metrics_list=metrics_list
        )
        
        if self.verbose:
            print("Network metrics computed successfully!")
        
        return self.metrics
    
    def differential_analysis(
        self,
        condition_column: str,
        control: str,
        treatment: str,
        networks: Optional[Dict] = None,
        method: str = 'permutation',
        n_permutations: int = 1000,
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform differential network analysis between conditions.
        
        Parameters
        ----------
        condition_column : str
            Column name in sample metadata indicating conditions
        control : str
            Control condition label
        treatment : str
            Treatment condition label
        networks : dict, optional
            Networks to analyze. If None, uses self.networks
        method : str, default='permutation'
            Statistical method for differential analysis
        n_permutations : int, default=1000
            Number of permutations for statistical testing
        alpha : float, default=0.05
            Significance level for statistical testing
        
        Returns
        -------
        dict
            Dictionary containing differential analysis results
        """
        if networks is None:
            networks = self.networks
        
        if not networks:
            raise ValueError("No networks available. Build networks first using build_networks().")
        
        if self.verbose:
            print(f"Performing differential analysis: {control} vs {treatment}")
        
        # Extract condition information from metadata
        condition_info = self._extract_condition_info(condition_column, control, treatment)
        
        self.differential_results = self.differential_analysis.analyze(
            networks=networks,
            condition_info=condition_info,
            method=method,
            n_permutations=n_permutations,
            alpha=alpha
        )
        
        if self.verbose:
            print("Differential analysis completed!")
        
        return self.differential_results
    
    def plot_networks(
        self,
        network_type: str = 'integrated',
        layout: str = 'spring',
        save_path: Optional[str] = None,
        **plot_kwargs
    ):
        """
        Plot network visualization.
        
        Parameters
        ----------
        network_type : str, default='integrated'
            Type of network to plot
        layout : str, default='spring'
            Layout algorithm for network visualization
        save_path : str, optional
            Path to save the plot
        **plot_kwargs
            Additional plotting parameters
        """
        if not self.networks:
            raise ValueError("No networks available. Build networks first using build_networks().")
        
        return self.plotter.plot_network(
            self.networks[network_type],
            layout=layout,
            save_path=save_path,
            **plot_kwargs
        )
    
    def save_results(
        self,
        output_dir: str,
        save_networks: bool = True,
        save_metrics: bool = True,
        save_differential: bool = True
    ) -> None:
        """
        Save analysis results to files.
        
        Parameters
        ----------
        output_dir : str
            Directory to save results
        save_networks : bool, default=True
            Whether to save network objects
        save_metrics : bool, default=True
            Whether to save network metrics
        save_differential : bool, default=True
            Whether to save differential analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Saving results to {output_dir}")
        
        # Save networks
        if save_networks and self.networks:
            import pickle
            with open(output_path / 'networks.pkl', 'wb') as f:
                pickle.dump(self.networks, f)
        
        # Save metrics
        if save_metrics and self.metrics:
            for metric_type, metrics_data in self.metrics.items():
                if isinstance(metrics_data, pd.DataFrame):
                    metrics_data.to_csv(output_path / f'metrics_{metric_type}.csv')
        
        # Save differential results
        if save_differential and self.differential_results:
            for result_type, results_data in self.differential_results.items():
                if isinstance(results_data, pd.DataFrame):
                    results_data.to_csv(output_path / f'differential_{result_type}.csv')
        
        if self.verbose:
            print("Results saved successfully!")
    
    def _extract_condition_info(self, condition_column: str, control: str, treatment: str) -> Dict:
        """Extract condition information from sample metadata."""
        condition_info = {}
        
        # Extract from each data type's metadata
        for data_type in ['rna', 'atac', 'metabolomics', 'microbiome']:
            if data_type in self.data and self.data[data_type]['sample_metadata'] is not None:
                metadata = self.data[data_type]['sample_metadata']
                if condition_column in metadata.columns:
                    condition_info[data_type] = {
                        'conditions': metadata[condition_column],
                        'control': control,
                        'treatment': treatment
                    }
        
        return condition_info

