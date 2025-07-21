"""
Visualization tools for multi-omics networks and analysis results.

This module provides comprehensive plotting functions for network visualization,
differential analysis results, and multi-omics integration outcomes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Union
import warnings
from pathlib import Path


class NetworkPlotter:
    """
    Comprehensive visualization tools for multi-omics networks.
    
    This class provides methods for plotting networks, differential analysis results,
    and creating publication-ready figures for multi-omics integration studies.
    """
    
    def __init__(self):
        # Set default plotting parameters
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Color schemes for different omics types
        self.omics_colors = {
            'rna': '#E74C3C',      # Red
            'atac': '#3498DB',     # Blue  
            'metabolomics': '#2ECC71',  # Green
            'microbiome': '#F39C12',    # Orange
            'integrated': '#9B59B6'     # Purple
        }
        
        # Node shapes for different omics types
        self.omics_shapes = {
            'rna': 'circle',
            'atac': 'square', 
            'metabolomics': 'triangle',
            'microbiome': 'diamond',
            'integrated': 'hexagon'
        }
    
    def plot_network(
        self,
        network: Union[nx.Graph, np.ndarray, pd.DataFrame],
        layout: str = 'spring',
        node_color_by: Optional[str] = None,
        node_size_by: Optional[str] = None,
        edge_width_by: Optional[str] = None,
        node_size: Union[int, float] = 50,
        edge_width: Union[int, float] = 1.0,
        alpha: float = 0.7,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        interactive: bool = False,
        **kwargs
    ):
        """
        Plot network visualization.
        
        Parameters
        ----------
        network : nx.Graph, np.ndarray, or pd.DataFrame
            Network to visualize
        layout : str, default='spring'
            Layout algorithm: 'spring', 'circular', 'kamada_kawai', 'spectral'
        node_color_by : str, optional
            Node attribute to color by
        node_size_by : str, optional
            Node attribute to size by
        edge_width_by : str, optional
            Edge attribute to determine width
        node_size : int or float, default=50
            Base node size
        edge_width : int or float, default=1.0
            Base edge width
        alpha : float, default=0.7
            Transparency level
        figsize : tuple, default=(12, 10)
            Figure size
        save_path : str, optional
            Path to save the plot
        interactive : bool, default=False
            Whether to create interactive plot using plotly
        **kwargs
            Additional plotting parameters
        
        Returns
        -------
        matplotlib.figure.Figure or plotly.graph_objects.Figure
            The created plot
        """
        # Convert to NetworkX if needed
        G = self._to_networkx(network)
        
        if interactive:
            return self._plot_network_interactive(
                G, layout, node_color_by, node_size_by, edge_width_by,
                node_size, edge_width, save_path, **kwargs
            )
        else:
            return self._plot_network_static(
                G, layout, node_color_by, node_size_by, edge_width_by,
                node_size, edge_width, alpha, figsize, save_path, **kwargs
            )
    
    def _to_networkx(self, network: Union[nx.Graph, np.ndarray, pd.DataFrame]) -> nx.Graph:
        """Convert various network formats to NetworkX graph."""
        if isinstance(network, nx.Graph):
            return network
        elif isinstance(network, (np.ndarray, pd.DataFrame)):
            if isinstance(network, pd.DataFrame):
                network = network.values
            G = nx.from_numpy_array(network)
            return G
        else:
            raise ValueError(f"Unsupported network type: {type(network)}")
    
    def _plot_network_static(
        self,
        G: nx.Graph,
        layout: str,
        node_color_by: Optional[str],
        node_size_by: Optional[str],
        edge_width_by: Optional[str],
        node_size: Union[int, float],
        edge_width: Union[int, float],
        alpha: float,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Create static network plot using matplotlib."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Determine node colors
        if node_color_by and node_color_by in G.nodes[list(G.nodes())[0]]:
            node_colors = [G.nodes[node][node_color_by] for node in G.nodes()]
        else:
            node_colors = self.omics_colors.get('integrated', '#9B59B6')
        
        # Determine node sizes
        if node_size_by and node_size_by in G.nodes[list(G.nodes())[0]]:
            node_sizes = [G.nodes[node][node_size_by] * node_size for node in G.nodes()]
        else:
            node_sizes = node_size
        
        # Determine edge widths
        if edge_width_by and G.edges() and edge_width_by in G.edges[list(G.edges())[0]]:
            edge_widths = [G.edges[edge][edge_width_by] * edge_width for edge in G.edges()]
        else:
            edge_widths = edge_width
        
        # Draw network
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sizes,
            alpha=alpha, ax=ax
        )
        
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, alpha=alpha*0.5, ax=ax
        )
        
        # Add labels if requested
        if kwargs.get('show_labels', False):
            nx.draw_networkx_labels(G, pos, ax=ax)
        
        ax.set_title(kwargs.get('title', 'Multi-Omics Network'), fontsize=16)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_network_interactive(
        self,
        G: nx.Graph,
        layout: str,
        node_color_by: Optional[str],
        node_size_by: Optional[str],
        edge_width_by: Optional[str],
        node_size: Union[int, float],
        edge_width: Union[int, float],
        save_path: Optional[str],
        **kwargs
    ) -> go.Figure:
        """Create interactive network plot using plotly."""
        # Compute layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Extract edge positions
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=edge_width, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Determine node colors and sizes
        if node_color_by and node_color_by in G.nodes[list(G.nodes())[0]]:
            node_colors = [G.nodes[node][node_color_by] for node in G.nodes()]
        else:
            node_colors = self.omics_colors.get('integrated', '#9B59B6')
        
        if node_size_by and node_size_by in G.nodes[list(G.nodes())[0]]:
            node_sizes = [G.nodes[node][node_size_by] * node_size for node in G.nodes()]
        else:
            node_sizes = node_size
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=[str(node) for node in G.nodes()],
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=kwargs.get('title', 'Multi-Omics Network'),
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Interactive Multi-Omics Network",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_differential_results(
        self,
        differential_results: Dict,
        plot_type: str = 'volcano',
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot differential analysis results.
        
        Parameters
        ----------
        differential_results : dict
            Results from differential analysis
        plot_type : str, default='volcano'
            Type of plot: 'volcano', 'manhattan', 'heatmap'
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, optional
            Path to save the plot
        **kwargs
            Additional plotting parameters
        
        Returns
        -------
        matplotlib.figure.Figure
            The created plot
        """
        if plot_type == 'volcano':
            return self._plot_volcano(differential_results, figsize, save_path, **kwargs)
        elif plot_type == 'manhattan':
            return self._plot_manhattan(differential_results, figsize, save_path, **kwargs)
        elif plot_type == 'heatmap':
            return self._plot_differential_heatmap(differential_results, figsize, save_path, **kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
    
    def _plot_volcano(
        self,
        differential_results: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Create volcano plot for differential analysis."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'differential_edges' in differential_results:
            df = differential_results['differential_edges']
            
            # Calculate -log10(p-value)
            neg_log_p = -np.log10(df['p_value'].clip(lower=1e-300))
            effect_size = df.get('effect_size', df.get('weight_difference', np.zeros(len(df))))
            
            # Color points based on significance
            colors = ['red' if p < 0.05 else 'gray' for p in df['p_value']]
            
            # Create scatter plot
            scatter = ax.scatter(effect_size, neg_log_p, c=colors, alpha=0.6, s=20)
            
            # Add significance line
            ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
            
            # Add labels
            ax.set_xlabel('Effect Size', fontsize=12)
            ax.set_ylabel('-log10(p-value)', fontsize=12)
            ax.set_title('Volcano Plot: Differential Network Analysis', fontsize=14)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Significant (p < 0.05)'),
                Patch(facecolor='gray', label='Not significant')
            ]
            ax.legend(handles=legend_elements)
            
            # Add grid
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_manhattan(
        self,
        differential_results: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Create Manhattan plot for differential analysis."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'differential_edges' in differential_results:
            df = differential_results['differential_edges']
            
            # Calculate -log10(p-value)
            neg_log_p = -np.log10(df['p_value'].clip(lower=1e-300))
            
            # Create x-axis positions
            x_pos = np.arange(len(df))
            
            # Color by network type if available
            if 'network' in df.columns:
                networks = df['network'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(networks)))
                color_map = dict(zip(networks, colors))
                point_colors = [color_map[net] for net in df['network']]
            else:
                point_colors = 'blue'
            
            # Create scatter plot
            ax.scatter(x_pos, neg_log_p, c=point_colors, alpha=0.6, s=20)
            
            # Add significance line
            ax.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
            
            # Labels and title
            ax.set_xlabel('Edge Index', fontsize=12)
            ax.set_ylabel('-log10(p-value)', fontsize=12)
            ax.set_title('Manhattan Plot: Differential Network Analysis', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_differential_heatmap(
        self,
        differential_results: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Create heatmap of differential analysis results."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'global_differences' in differential_results:
            # Create matrix of global differences
            networks = list(differential_results['global_differences'].keys())
            metrics = []
            
            # Get all metrics
            for network in networks:
                metrics.extend(differential_results['global_differences'][network].keys())
            metrics = list(set(metrics))
            
            # Create data matrix
            data_matrix = np.zeros((len(networks), len(metrics)))
            
            for i, network in enumerate(networks):
                for j, metric in enumerate(metrics):
                    if metric in differential_results['global_differences'][network]:
                        data_matrix[i, j] = differential_results['global_differences'][network][metric]['difference']
            
            # Create heatmap
            sns.heatmap(
                data_matrix,
                xticklabels=metrics,
                yticklabels=networks,
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                ax=ax
            )
            
            ax.set_title('Heatmap: Global Network Differences', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_network_metrics(
        self,
        metrics: Dict,
        metric_type: str = 'centrality',
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot network metrics visualization.
        
        Parameters
        ----------
        metrics : dict
            Network metrics dictionary
        metric_type : str, default='centrality'
            Type of metrics to plot: 'centrality', 'global', 'distribution'
        figsize : tuple, default=(15, 10)
            Figure size
        save_path : str, optional
            Path to save the plot
        **kwargs
            Additional plotting parameters
        
        Returns
        -------
        matplotlib.figure.Figure
            The created plot
        """
        if metric_type == 'centrality':
            return self._plot_centrality_metrics(metrics, figsize, save_path, **kwargs)
        elif metric_type == 'global':
            return self._plot_global_metrics(metrics, figsize, save_path, **kwargs)
        elif metric_type == 'distribution':
            return self._plot_metric_distributions(metrics, figsize, save_path, **kwargs)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    def _plot_centrality_metrics(
        self,
        metrics: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Plot centrality metrics."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        centrality_metrics = ['degree_centrality', 'betweenness_centrality', 
                            'closeness_centrality', 'eigenvector_centrality']
        
        for i, metric in enumerate(centrality_metrics):
            if metric in metrics and isinstance(metrics[metric], dict):
                values = list(metrics[metric].values())
                
                axes[i].hist(values, bins=30, alpha=0.7, color=f'C{i}')
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Centrality Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Network Centrality Metrics', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_global_metrics(
        self,
        metrics: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Plot global network metrics."""
        fig, ax = plt.subplots(figsize=figsize)
        
        if 'network_summary' in metrics:
            summary_df = metrics['network_summary']
            
            # Select numeric columns
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                # Create bar plot
                summary_df[numeric_cols].plot(kind='bar', ax=ax)
                ax.set_title('Global Network Metrics', fontsize=14)
                ax.set_xlabel('Networks')
                ax.set_ylabel('Metric Value')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _plot_metric_distributions(
        self,
        metrics: Dict,
        figsize: Tuple[int, int],
        save_path: Optional[str],
        **kwargs
    ) -> plt.Figure:
        """Plot distributions of network metrics."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        metric_names = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality',
                       'clustering_coefficient', 'pagerank', 'eigenvector_centrality']
        
        for i, metric in enumerate(metric_names):
            if i < len(axes) and metric in metrics and isinstance(metrics[metric], dict):
                values = list(metrics[metric].values())
                
                # Create violin plot
                axes[i].violinplot([values], positions=[0], showmeans=True)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_ylabel('Value')
                axes[i].set_xticks([])
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(metric_names), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Network Metric Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multilayer_network(
        self,
        networks: Dict[str, nx.Graph],
        layer_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot multi-layer network visualization.
        
        Parameters
        ----------
        networks : dict
            Dictionary of networks for each layer
        layer_positions : dict, optional
            Positions for each layer in the plot
        figsize : tuple, default=(15, 12)
            Figure size
        save_path : str, optional
            Path to save the plot
        **kwargs
            Additional plotting parameters
        
        Returns
        -------
        matplotlib.figure.Figure
            The created plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Default layer positions
        if layer_positions is None:
            n_layers = len(networks)
            layer_positions = {}
            for i, layer_name in enumerate(networks.keys()):
                angle = 2 * np.pi * i / n_layers
                layer_positions[layer_name] = (np.cos(angle), np.sin(angle))
        
        # Plot each layer
        for layer_name, G in networks.items():
            if len(G.nodes()) == 0:
                continue
                
            # Get layer position
            layer_x, layer_y = layer_positions.get(layer_name, (0, 0))
            
            # Compute node positions within layer
            pos = nx.spring_layout(G, k=0.3, iterations=20)
            
            # Adjust positions to layer location
            adjusted_pos = {}
            for node, (x, y) in pos.items():
                adjusted_pos[node] = (layer_x + x * 0.3, layer_y + y * 0.3)
            
            # Get layer color
            layer_color = self.omics_colors.get(layer_name.lower(), '#888888')
            
            # Draw nodes
            nx.draw_networkx_nodes(
                G, adjusted_pos, 
                node_color=layer_color,
                node_size=100,
                alpha=0.7,
                ax=ax,
                label=layer_name
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, adjusted_pos,
                edge_color=layer_color,
                alpha=0.3,
                width=0.5,
                ax=ax
            )
        
        # Add layer labels
        for layer_name, (x, y) in layer_positions.items():
            ax.text(x, y + 0.6, layer_name, 
                   horizontalalignment='center',
                   fontsize=12, fontweight='bold')
        
        ax.set_title('Multi-Layer Network Visualization', fontsize=16)
        ax.legend()
        ax.axis('equal')
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(
        self,
        networks: Dict,
        metrics: Dict,
        differential_results: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create comprehensive summary dashboard.
        
        Parameters
        ----------
        networks : dict
            Dictionary of networks
        metrics : dict
            Network metrics
        differential_results : dict, optional
            Differential analysis results
        save_path : str, optional
            Path to save the dashboard
        
        Returns
        -------
        matplotlib.figure.Figure
            The dashboard figure
        """
        # Create subplot layout
        if differential_results:
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        else:
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Network overview
        ax1 = fig.add_subplot(gs[0, :2])
        if 'network_summary' in metrics:
            summary_df = metrics['network_summary']
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_df[numeric_cols[:5]].plot(kind='bar', ax=ax1)
                ax1.set_title('Network Overview', fontsize=14)
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Degree distribution
        ax2 = fig.add_subplot(gs[0, 2])
        if 'degree_centrality' in metrics:
            values = list(metrics['degree_centrality'].values())
            ax2.hist(values, bins=20, alpha=0.7)
            ax2.set_title('Degree Distribution')
            ax2.set_xlabel('Degree Centrality')
            ax2.set_ylabel('Frequency')
        
        # Clustering coefficient
        ax3 = fig.add_subplot(gs[1, 0])
        if 'clustering_coefficient' in metrics:
            values = list(metrics['clustering_coefficient'].values())
            ax3.hist(values, bins=20, alpha=0.7, color='orange')
            ax3.set_title('Clustering Coefficient')
            ax3.set_xlabel('Clustering Coefficient')
            ax3.set_ylabel('Frequency')
        
        # Betweenness centrality
        ax4 = fig.add_subplot(gs[1, 1])
        if 'betweenness_centrality' in metrics:
            values = list(metrics['betweenness_centrality'].values())
            ax4.hist(values, bins=20, alpha=0.7, color='green')
            ax4.set_title('Betweenness Centrality')
            ax4.set_xlabel('Betweenness Centrality')
            ax4.set_ylabel('Frequency')
        
        # Network metrics comparison
        ax5 = fig.add_subplot(gs[1, 2])
        if 'network_summary' in metrics:
            summary_df = metrics['network_summary']
            global_metrics = ['network_density', 'transitivity', 'modularity']
            available_metrics = [m for m in global_metrics if m in summary_df.columns]
            if available_metrics:
                summary_df[available_metrics].plot(kind='bar', ax=ax5)
                ax5.set_title('Global Metrics')
                ax5.legend()
        
        # Differential analysis results (if available)
        if differential_results:
            # Volcano plot
            ax6 = fig.add_subplot(gs[2, :2])
            if 'differential_edges' in differential_results:
                df = differential_results['differential_edges']
                neg_log_p = -np.log10(df['p_value'].clip(lower=1e-300))
                effect_size = df.get('effect_size', df.get('weight_difference', np.zeros(len(df))))
                colors = ['red' if p < 0.05 else 'gray' for p in df['p_value']]
                
                ax6.scatter(effect_size, neg_log_p, c=colors, alpha=0.6, s=10)
                ax6.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
                ax6.set_xlabel('Effect Size')
                ax6.set_ylabel('-log10(p-value)')
                ax6.set_title('Differential Analysis: Volcano Plot')
            
            # Significant edges summary
            ax7 = fig.add_subplot(gs[2, 2])
            if 'differential_edges' in differential_results:
                df = differential_results['differential_edges']
                sig_counts = df.groupby('network')['p_value'].apply(lambda x: (x < 0.05).sum())
                sig_counts.plot(kind='bar', ax=ax7)
                ax7.set_title('Significant Edges by Network')
                ax7.set_xlabel('Network')
                ax7.set_ylabel('Number of Significant Edges')
                plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Multi-Omics Network Analysis Dashboard', fontsize=18, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

