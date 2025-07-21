"""
Network quantification metrics for multi-omics networks.

This module provides comprehensive metrics for quantifying network properties
including node-level, network-level, and multi-layer network metrics.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
import warnings


class NetworkMetrics:
    """
    Comprehensive network metrics computation for multi-omics networks.
    
    This class provides methods to compute various network metrics including
    centrality measures, topological properties, and multi-layer network metrics.
    
    Parameters
    ----------
    verbose : bool, default=True
        Whether to print progress messages
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.available_metrics = [
            'degree_centrality',
            'betweenness_centrality', 
            'closeness_centrality',
            'eigenvector_centrality',
            'pagerank',
            'clustering_coefficient',
            'local_efficiency',
            'modularity',
            'small_worldness',
            'network_density',
            'assortativity',
            'average_path_length',
            'diameter',
            'transitivity',
            'rich_club_coefficient'
        ]
    
    def compute_metrics(
        self,
        networks: Dict[str, Union[nx.Graph, np.ndarray, pd.DataFrame]],
        metrics_list: Optional[List[str]] = None
    ) -> Dict[str, Union[Dict, pd.DataFrame, float]]:
        """
        Compute network metrics for all provided networks.
        
        Parameters
        ----------
        networks : dict
            Dictionary of networks to analyze
        metrics_list : list, optional
            List of metrics to compute. If None, computes all available metrics
        
        Returns
        -------
        dict
            Dictionary containing computed metrics for each network
        """
        if metrics_list is None:
            metrics_list = self.available_metrics
        
        if self.verbose:
            print(f"Computing {len(metrics_list)} metrics for {len(networks)} networks...")
        
        results = {}
        
        for network_name, network in networks.items():
            if self.verbose:
                print(f"Processing network: {network_name}")
            
            # Convert to NetworkX graph if needed
            G = self._to_networkx(network)
            
            network_results = {}
            
            # Compute node-level metrics
            node_metrics = self._compute_node_metrics(G, metrics_list)
            network_results.update(node_metrics)
            
            # Compute network-level metrics
            network_level_metrics = self._compute_network_metrics(G, metrics_list)
            network_results.update(network_level_metrics)
            
            results[network_name] = network_results
        
        # Create summary statistics
        results['network_summary'] = self._create_summary(results, networks.keys())
        
        if self.verbose:
            print("Network metrics computation completed!")
        
        return results
    
    def _to_networkx(self, network: Union[nx.Graph, np.ndarray, pd.DataFrame]) -> nx.Graph:
        """Convert various network formats to NetworkX graph."""
        if isinstance(network, nx.Graph):
            return network
        elif isinstance(network, (np.ndarray, pd.DataFrame)):
            if isinstance(network, pd.DataFrame):
                network = network.values
            
            # Create graph from adjacency matrix
            G = nx.from_numpy_array(network)
            return G
        else:
            raise ValueError(f"Unsupported network type: {type(network)}")
    
    def _compute_node_metrics(self, G: nx.Graph, metrics_list: List[str]) -> Dict:
        """Compute node-level metrics."""
        results = {}
        
        # Degree centrality
        if 'degree_centrality' in metrics_list:
            results['degree_centrality'] = nx.degree_centrality(G)
        
        # Betweenness centrality
        if 'betweenness_centrality' in metrics_list:
            results['betweenness_centrality'] = nx.betweenness_centrality(G)
        
        # Closeness centrality
        if 'closeness_centrality' in metrics_list:
            if nx.is_connected(G):
                results['closeness_centrality'] = nx.closeness_centrality(G)
            else:
                # For disconnected graphs, compute for each component
                results['closeness_centrality'] = {}
                for component in nx.connected_components(G):
                    subgraph = G.subgraph(component)
                    closeness = nx.closeness_centrality(subgraph)
                    results['closeness_centrality'].update(closeness)
        
        # Eigenvector centrality
        if 'eigenvector_centrality' in metrics_list:
            try:
                results['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                if self.verbose:
                    print("Warning: Eigenvector centrality failed to converge")
                results['eigenvector_centrality'] = {node: 0.0 for node in G.nodes()}
        
        # PageRank
        if 'pagerank' in metrics_list:
            results['pagerank'] = nx.pagerank(G)
        
        # Clustering coefficient
        if 'clustering_coefficient' in metrics_list:
            results['clustering_coefficient'] = nx.clustering(G)
        
        # Local efficiency
        if 'local_efficiency' in metrics_list:
            results['local_efficiency'] = {
                node: nx.local_efficiency(G, node) for node in G.nodes()
            }
        
        return results
    
    def _compute_network_metrics(self, G: nx.Graph, metrics_list: List[str]) -> Dict:
        """Compute network-level metrics."""
        results = {}
        
        # Network density
        if 'network_density' in metrics_list:
            results['network_density'] = nx.density(G)
        
        # Assortativity
        if 'assortativity' in metrics_list:
            try:
                results['assortativity'] = nx.degree_assortativity_coefficient(G)
            except:
                results['assortativity'] = np.nan
        
        # Average path length
        if 'average_path_length' in metrics_list:
            if nx.is_connected(G):
                results['average_path_length'] = nx.average_shortest_path_length(G)
            else:
                # For disconnected graphs, compute average over components
                path_lengths = []
                for component in nx.connected_components(G):
                    if len(component) > 1:
                        subgraph = G.subgraph(component)
                        path_lengths.append(nx.average_shortest_path_length(subgraph))
                results['average_path_length'] = np.mean(path_lengths) if path_lengths else np.inf
        
        # Diameter
        if 'diameter' in metrics_list:
            if nx.is_connected(G):
                results['diameter'] = nx.diameter(G)
            else:
                # For disconnected graphs, compute max diameter over components
                diameters = []
                for component in nx.connected_components(G):
                    if len(component) > 1:
                        subgraph = G.subgraph(component)
                        diameters.append(nx.diameter(subgraph))
                results['diameter'] = max(diameters) if diameters else np.inf
        
        # Transitivity (global clustering coefficient)
        if 'transitivity' in metrics_list:
            results['transitivity'] = nx.transitivity(G)
        
        # Modularity
        if 'modularity' in metrics_list:
            # Use Louvain community detection
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)
                results['modularity'] = community_louvain.modularity(partition, G)
                results['communities'] = partition
            except ImportError:
                # Fallback to greedy modularity communities
                communities = nx.community.greedy_modularity_communities(G)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
                results['modularity'] = nx.community.modularity(G, communities)
                results['communities'] = partition
        
        # Small-worldness
        if 'small_worldness' in metrics_list:
            results['small_worldness'] = self._compute_small_worldness(G)
        
        # Rich club coefficient
        if 'rich_club_coefficient' in metrics_list:
            try:
                rich_club = nx.rich_club_coefficient(G)
                results['rich_club_coefficient'] = rich_club
            except:
                results['rich_club_coefficient'] = {}
        
        return results
    
    def _compute_small_worldness(self, G: nx.Graph) -> float:
        """
        Compute small-worldness coefficient.
        
        Small-worldness = (C/C_random) / (L/L_random)
        where C is clustering coefficient and L is average path length.
        """
        if not nx.is_connected(G):
            return np.nan
        
        # Actual network metrics
        C = nx.transitivity(G)
        L = nx.average_shortest_path_length(G)
        
        # Generate random network with same degree sequence
        try:
            degree_sequence = [d for n, d in G.degree()]
            G_random = nx.configuration_model(degree_sequence)
            G_random = nx.Graph(G_random)  # Remove multi-edges and self-loops
            G_random.remove_edges_from(nx.selfloop_edges(G_random))
            
            if nx.is_connected(G_random):
                C_random = nx.transitivity(G_random)
                L_random = nx.average_shortest_path_length(G_random)
                
                if C_random > 0 and L_random > 0:
                    small_worldness = (C / C_random) / (L / L_random)
                    return small_worldness
        except:
            pass
        
        return np.nan
    
    def _create_summary(self, results: Dict, network_names: List[str]) -> pd.DataFrame:
        """Create summary statistics table."""
        summary_data = []
        
        for network_name in network_names:
            if network_name in results:
                network_results = results[network_name]
                
                summary_row = {'Network': network_name}
                
                # Extract scalar metrics
                scalar_metrics = [
                    'network_density', 'assortativity', 'average_path_length',
                    'diameter', 'transitivity', 'modularity', 'small_worldness'
                ]
                
                for metric in scalar_metrics:
                    if metric in network_results:
                        summary_row[metric] = network_results[metric]
                
                # Compute summary statistics for node-level metrics
                node_metrics = [
                    'degree_centrality', 'betweenness_centrality', 'closeness_centrality',
                    'eigenvector_centrality', 'pagerank', 'clustering_coefficient'
                ]
                
                for metric in node_metrics:
                    if metric in network_results and isinstance(network_results[metric], dict):
                        values = list(network_results[metric].values())
                        if values:
                            summary_row[f'{metric}_mean'] = np.mean(values)
                            summary_row[f'{metric}_std'] = np.std(values)
                            summary_row[f'{metric}_max'] = np.max(values)
                
                summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)
    
    def compute_multilayer_metrics(
        self,
        networks: Dict[str, nx.Graph],
        layer_mapping: Dict[str, str]
    ) -> Dict:
        """
        Compute multi-layer network metrics.
        
        Parameters
        ----------
        networks : dict
            Dictionary of networks for each layer
        layer_mapping : dict
            Mapping of network names to layer types
        
        Returns
        -------
        dict
            Multi-layer network metrics
        """
        results = {}
        
        # Inter-layer connectivity
        results['inter_layer_connectivity'] = self._compute_inter_layer_connectivity(
            networks, layer_mapping
        )
        
        # Layer similarity
        results['layer_similarity'] = self._compute_layer_similarity(networks)
        
        # Multiplex participation coefficient
        results['multiplex_participation'] = self._compute_multiplex_participation(
            networks, layer_mapping
        )
        
        return results
    
    def _compute_inter_layer_connectivity(
        self,
        networks: Dict[str, nx.Graph],
        layer_mapping: Dict[str, str]
    ) -> Dict:
        """Compute connectivity between different layers."""
        connectivity = {}
        
        layer_types = list(set(layer_mapping.values()))
        
        for i, layer1 in enumerate(layer_types):
            for layer2 in layer_types[i+1:]:
                # Find networks belonging to each layer
                networks1 = [name for name, layer in layer_mapping.items() if layer == layer1]
                networks2 = [name for name, layer in layer_mapping.items() if layer == layer2]
                
                # Compute connectivity between layers
                total_connections = 0
                total_possible = 0
                
                for net1 in networks1:
                    for net2 in networks2:
                        if net1 in networks and net2 in networks:
                            G1, G2 = networks[net1], networks[net2]
                            
                            # Find common nodes
                            common_nodes = set(G1.nodes()) & set(G2.nodes())
                            
                            if common_nodes:
                                # Count connections between layers
                                connections = 0
                                for node in common_nodes:
                                    if G1.has_node(node) and G2.has_node(node):
                                        connections += len(G1[node]) + len(G2[node])
                                
                                total_connections += connections
                                total_possible += len(common_nodes) * 2
                
                if total_possible > 0:
                    connectivity[f'{layer1}_{layer2}'] = total_connections / total_possible
        
        return connectivity
    
    def _compute_layer_similarity(self, networks: Dict[str, nx.Graph]) -> pd.DataFrame:
        """Compute similarity between network layers."""
        network_names = list(networks.keys())
        n_networks = len(network_names)
        
        similarity_matrix = np.zeros((n_networks, n_networks))
        
        for i, net1 in enumerate(network_names):
            for j, net2 in enumerate(network_names):
                if i <= j:
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Compute Jaccard similarity of edges
                        G1, G2 = networks[net1], networks[net2]
                        
                        edges1 = set(G1.edges())
                        edges2 = set(G2.edges())
                        
                        intersection = len(edges1 & edges2)
                        union = len(edges1 | edges2)
                        
                        if union > 0:
                            similarity = intersection / union
                        else:
                            similarity = 0.0
                        
                        similarity_matrix[i, j] = similarity
                        similarity_matrix[j, i] = similarity
        
        return pd.DataFrame(
            similarity_matrix,
            index=network_names,
            columns=network_names
        )
    
    def _compute_multiplex_participation(
        self,
        networks: Dict[str, nx.Graph],
        layer_mapping: Dict[str, str]
    ) -> Dict:
        """Compute multiplex participation coefficient for nodes."""
        # Find nodes present in multiple layers
        all_nodes = set()
        for G in networks.values():
            all_nodes.update(G.nodes())
        
        participation = {}
        
        for node in all_nodes:
            # Find which layers contain this node
            node_layers = []
            node_degrees = []
            
            for net_name, G in networks.items():
                if G.has_node(node):
                    layer = layer_mapping.get(net_name, 'unknown')
                    node_layers.append(layer)
                    node_degrees.append(G.degree(node))
            
            if len(node_layers) > 1:
                # Compute participation coefficient
                total_degree = sum(node_degrees)
                if total_degree > 0:
                    participation_coeff = 1 - sum(
                        (degree / total_degree) ** 2 for degree in node_degrees
                    )
                    participation[node] = participation_coeff
                else:
                    participation[node] = 0.0
            else:
                participation[node] = 0.0
        
        return participation
    
    def compute_network_robustness(
        self,
        G: nx.Graph,
        attack_strategy: str = 'random',
        n_simulations: int = 100
    ) -> Dict:
        """
        Compute network robustness under node removal.
        
        Parameters
        ----------
        G : nx.Graph
            Network to analyze
        attack_strategy : str, default='random'
            Strategy for node removal: 'random', 'degree', 'betweenness'
        n_simulations : int, default=100
            Number of simulation runs
        
        Returns
        -------
        dict
            Robustness metrics
        """
        results = {
            'attack_strategy': attack_strategy,
            'fraction_removed': [],
            'largest_component_size': [],
            'efficiency': []
        }
        
        original_size = len(G.nodes())
        original_efficiency = nx.global_efficiency(G)
        
        for sim in range(n_simulations):
            G_copy = G.copy()
            nodes_to_remove = list(G_copy.nodes())
            
            # Sort nodes based on attack strategy
            if attack_strategy == 'degree':
                nodes_to_remove.sort(key=lambda x: G_copy.degree(x), reverse=True)
            elif attack_strategy == 'betweenness':
                betweenness = nx.betweenness_centrality(G_copy)
                nodes_to_remove.sort(key=lambda x: betweenness[x], reverse=True)
            else:  # random
                np.random.shuffle(nodes_to_remove)
            
            # Remove nodes progressively
            for i, node in enumerate(nodes_to_remove):
                if G_copy.has_node(node):
                    G_copy.remove_node(node)
                
                # Record metrics every 10% of nodes removed
                if (i + 1) % max(1, original_size // 10) == 0:
                    fraction_removed = (i + 1) / original_size
                    
                    if G_copy.nodes():
                        # Largest connected component size
                        largest_cc = max(nx.connected_components(G_copy), key=len)
                        largest_cc_size = len(largest_cc) / original_size
                        
                        # Network efficiency
                        efficiency = nx.global_efficiency(G_copy) / original_efficiency
                    else:
                        largest_cc_size = 0
                        efficiency = 0
                    
                    results['fraction_removed'].append(fraction_removed)
                    results['largest_component_size'].append(largest_cc_size)
                    results['efficiency'].append(efficiency)
        
        # Compute average across simulations
        n_points = len(results['fraction_removed']) // n_simulations
        
        avg_results = {
            'attack_strategy': attack_strategy,
            'fraction_removed': [],
            'largest_component_size_mean': [],
            'largest_component_size_std': [],
            'efficiency_mean': [],
            'efficiency_std': []
        }
        
        for i in range(n_points):
            indices = [i + j * n_points for j in range(n_simulations)]
            
            avg_results['fraction_removed'].append(
                np.mean([results['fraction_removed'][idx] for idx in indices])
            )
            avg_results['largest_component_size_mean'].append(
                np.mean([results['largest_component_size'][idx] for idx in indices])
            )
            avg_results['largest_component_size_std'].append(
                np.std([results['largest_component_size'][idx] for idx in indices])
            )
            avg_results['efficiency_mean'].append(
                np.mean([results['efficiency'][idx] for idx in indices])
            )
            avg_results['efficiency_std'].append(
                np.std([results['efficiency'][idx] for idx in indices])
            )
        
        return avg_results

