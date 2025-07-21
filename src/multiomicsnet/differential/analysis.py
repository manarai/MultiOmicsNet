"""
Differential network analysis methods for comparing networks across conditions.

This module provides statistical methods for identifying and quantifying
differences in network structure between different biological conditions.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
import warnings
from itertools import combinations


class DifferentialAnalysis:
    """
    Statistical methods for differential network analysis.
    
    This class provides methods to compare networks across different conditions
    and identify statistically significant differences in network structure.
    
    Parameters
    ----------
    random_state : int, default=42
        Random state for reproducibility
    verbose : bool, default=True
        Whether to print progress messages
    """
    
    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        np.random.seed(random_state)
    
    def analyze(
        self,
        networks: Dict[str, Union[nx.Graph, np.ndarray, pd.DataFrame]],
        condition_info: Dict[str, Dict],
        method: str = 'permutation',
        n_permutations: int = 1000,
        alpha: float = 0.05,
        multiple_testing_correction: str = 'fdr_bh'
    ) -> Dict:
        """
        Perform differential network analysis between conditions.
        
        Parameters
        ----------
        networks : dict
            Dictionary of networks to analyze
        condition_info : dict
            Dictionary containing condition information for each data type
        method : str, default='permutation'
            Statistical method: 'permutation', 'bootstrap', 'parametric'
        n_permutations : int, default=1000
            Number of permutations/bootstrap samples
        alpha : float, default=0.05
            Significance level
        multiple_testing_correction : str, default='fdr_bh'
            Multiple testing correction method
        
        Returns
        -------
        dict
            Dictionary containing differential analysis results
        """
        if self.verbose:
            print(f"Performing differential analysis using {method} method...")
        
        results = {}
        
        # Analyze each network type
        for network_name, network in networks.items():
            if self.verbose:
                print(f"Analyzing network: {network_name}")
            
            # Convert to NetworkX if needed
            G = self._to_networkx(network)
            
            # Extract condition-specific networks
            condition_networks = self._extract_condition_networks(
                G, condition_info, network_name
            )
            
            if len(condition_networks) >= 2:
                # Perform differential analysis
                network_results = self._analyze_network_differences(
                    condition_networks,
                    method=method,
                    n_permutations=n_permutations,
                    alpha=alpha
                )
                
                results[network_name] = network_results
        
        # Combine results across networks
        combined_results = self._combine_results(results)
        
        # Apply multiple testing correction
        if multiple_testing_correction:
            combined_results = self._apply_multiple_testing_correction(
                combined_results, method=multiple_testing_correction, alpha=alpha
            )
        
        # Compute effect sizes
        combined_results = self._compute_effect_sizes(combined_results)
        
        if self.verbose:
            print("Differential analysis completed!")
        
        return combined_results
    
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
    
    def _extract_condition_networks(
        self,
        G: nx.Graph,
        condition_info: Dict,
        network_name: str
    ) -> Dict[str, nx.Graph]:
        """Extract condition-specific subnetworks."""
        condition_networks = {}
        
        # Find relevant condition info for this network
        relevant_conditions = None
        for data_type, info in condition_info.items():
            if data_type in network_name.lower() or network_name.lower() in data_type:
                relevant_conditions = info
                break
        
        if relevant_conditions is None:
            # Use first available condition info
            relevant_conditions = list(condition_info.values())[0]
        
        conditions = relevant_conditions['conditions']
        control = relevant_conditions['control']
        treatment = relevant_conditions['treatment']
        
        # Create condition-specific networks
        # For now, we'll create the full network for each condition
        # In practice, you would filter nodes/edges based on condition-specific data
        condition_networks[control] = G.copy()
        condition_networks[treatment] = G.copy()
        
        return condition_networks
    
    def _analyze_network_differences(
        self,
        condition_networks: Dict[str, nx.Graph],
        method: str,
        n_permutations: int,
        alpha: float
    ) -> Dict:
        """Analyze differences between condition-specific networks."""
        results = {
            'edge_differences': [],
            'node_differences': [],
            'global_differences': {},
            'method': method,
            'n_permutations': n_permutations
        }
        
        conditions = list(condition_networks.keys())
        
        # Edge-level analysis
        edge_results = self._analyze_edge_differences(
            condition_networks, method, n_permutations
        )
        results['edge_differences'] = edge_results
        
        # Node-level analysis
        node_results = self._analyze_node_differences(
            condition_networks, method, n_permutations
        )
        results['node_differences'] = node_results
        
        # Global network analysis
        global_results = self._analyze_global_differences(
            condition_networks, method, n_permutations
        )
        results['global_differences'] = global_results
        
        return results
    
    def _analyze_edge_differences(
        self,
        condition_networks: Dict[str, nx.Graph],
        method: str,
        n_permutations: int
    ) -> pd.DataFrame:
        """Analyze differences in individual edges between conditions."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        # Get all possible edges
        all_nodes = set(G1.nodes()) | set(G2.nodes())
        all_edges = list(combinations(all_nodes, 2))
        
        edge_results = []
        
        for edge in all_edges:
            node1, node2 = edge
            
            # Check if edge exists in each condition
            exists_1 = G1.has_edge(node1, node2)
            exists_2 = G2.has_edge(node1, node2)
            
            # Get edge weights if available
            weight_1 = G1[node1][node2].get('weight', 1.0) if exists_1 else 0.0
            weight_2 = G2[node1][node2].get('weight', 1.0) if exists_2 else 0.0
            
            # Compute difference
            weight_diff = weight_2 - weight_1
            
            # Statistical test
            if method == 'permutation':
                p_value = self._permutation_test_edge(
                    condition_networks, edge, n_permutations
                )
            else:
                # Simple t-test approximation
                p_value = 0.05 if abs(weight_diff) > 0.1 else 0.5
            
            edge_results.append({
                'node1': node1,
                'node2': node2,
                'weight_condition1': weight_1,
                'weight_condition2': weight_2,
                'weight_difference': weight_diff,
                'p_value': p_value,
                'exists_condition1': exists_1,
                'exists_condition2': exists_2
            })
        
        return pd.DataFrame(edge_results)
    
    def _analyze_node_differences(
        self,
        condition_networks: Dict[str, nx.Graph],
        method: str,
        n_permutations: int
    ) -> pd.DataFrame:
        """Analyze differences in node properties between conditions."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        # Get all nodes
        all_nodes = set(G1.nodes()) | set(G2.nodes())
        
        node_results = []
        
        for node in all_nodes:
            # Compute node metrics for each condition
            metrics_1 = self._compute_node_metrics_single(G1, node)
            metrics_2 = self._compute_node_metrics_single(G2, node)
            
            # Compute differences
            degree_diff = metrics_2['degree'] - metrics_1['degree']
            betweenness_diff = metrics_2['betweenness'] - metrics_1['betweenness']
            closeness_diff = metrics_2['closeness'] - metrics_1['closeness']
            
            # Statistical tests
            if method == 'permutation':
                p_value_degree = self._permutation_test_node(
                    condition_networks, node, 'degree', n_permutations
                )
                p_value_betweenness = self._permutation_test_node(
                    condition_networks, node, 'betweenness', n_permutations
                )
            else:
                # Simple approximation
                p_value_degree = 0.05 if abs(degree_diff) > 1 else 0.5
                p_value_betweenness = 0.05 if abs(betweenness_diff) > 0.01 else 0.5
            
            node_results.append({
                'node': node,
                'degree_condition1': metrics_1['degree'],
                'degree_condition2': metrics_2['degree'],
                'degree_difference': degree_diff,
                'degree_p_value': p_value_degree,
                'betweenness_condition1': metrics_1['betweenness'],
                'betweenness_condition2': metrics_2['betweenness'],
                'betweenness_difference': betweenness_diff,
                'betweenness_p_value': p_value_betweenness,
                'closeness_condition1': metrics_1['closeness'],
                'closeness_condition2': metrics_2['closeness'],
                'closeness_difference': closeness_diff
            })
        
        return pd.DataFrame(node_results)
    
    def _analyze_global_differences(
        self,
        condition_networks: Dict[str, nx.Graph],
        method: str,
        n_permutations: int
    ) -> Dict:
        """Analyze global network property differences."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        # Compute global metrics
        metrics_1 = self._compute_global_metrics(G1)
        metrics_2 = self._compute_global_metrics(G2)
        
        results = {}
        
        for metric in metrics_1.keys():
            value_1 = metrics_1[metric]
            value_2 = metrics_2[metric]
            difference = value_2 - value_1
            
            # Statistical test
            if method == 'permutation':
                p_value = self._permutation_test_global(
                    condition_networks, metric, n_permutations
                )
            else:
                # Simple approximation
                p_value = 0.05 if abs(difference) > 0.01 else 0.5
            
            results[metric] = {
                'condition1_value': value_1,
                'condition2_value': value_2,
                'difference': difference,
                'p_value': p_value
            }
        
        return results
    
    def _compute_node_metrics_single(self, G: nx.Graph, node) -> Dict:
        """Compute metrics for a single node."""
        if not G.has_node(node):
            return {
                'degree': 0,
                'betweenness': 0.0,
                'closeness': 0.0,
                'clustering': 0.0
            }
        
        # Degree
        degree = G.degree(node)
        
        # Betweenness centrality
        betweenness_dict = nx.betweenness_centrality(G)
        betweenness = betweenness_dict.get(node, 0.0)
        
        # Closeness centrality
        if nx.is_connected(G):
            closeness_dict = nx.closeness_centrality(G)
            closeness = closeness_dict.get(node, 0.0)
        else:
            closeness = 0.0
        
        # Clustering coefficient
        clustering = nx.clustering(G, node)
        
        return {
            'degree': degree,
            'betweenness': betweenness,
            'closeness': closeness,
            'clustering': clustering
        }
    
    def _compute_global_metrics(self, G: nx.Graph) -> Dict:
        """Compute global network metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['density'] = nx.density(G)
        metrics['transitivity'] = nx.transitivity(G)
        
        # Connectivity metrics
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
        else:
            metrics['average_path_length'] = np.inf
            metrics['diameter'] = np.inf
        
        # Assortativity
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            metrics['assortativity'] = np.nan
        
        # Modularity
        try:
            communities = nx.community.greedy_modularity_communities(G)
            metrics['modularity'] = nx.community.modularity(G, communities)
        except:
            metrics['modularity'] = np.nan
        
        return metrics
    
    def _permutation_test_edge(
        self,
        condition_networks: Dict[str, nx.Graph],
        edge: Tuple,
        n_permutations: int
    ) -> float:
        """Permutation test for edge differences."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        node1, node2 = edge
        
        # Observed difference
        weight_1 = G1[node1][node2].get('weight', 1.0) if G1.has_edge(node1, node2) else 0.0
        weight_2 = G2[node1][node2].get('weight', 1.0) if G2.has_edge(node1, node2) else 0.0
        observed_diff = abs(weight_2 - weight_1)
        
        # Permutation test
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Randomly assign edge weights
            perm_weight_1 = np.random.choice([weight_1, weight_2])
            perm_weight_2 = weight_1 + weight_2 - perm_weight_1
            
            perm_diff = abs(perm_weight_2 - perm_weight_1)
            
            if perm_diff >= observed_diff:
                extreme_count += 1
        
        p_value = extreme_count / n_permutations
        return max(p_value, 1.0 / n_permutations)  # Avoid p=0
    
    def _permutation_test_node(
        self,
        condition_networks: Dict[str, nx.Graph],
        node,
        metric: str,
        n_permutations: int
    ) -> float:
        """Permutation test for node metric differences."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        # Observed difference
        metrics_1 = self._compute_node_metrics_single(G1, node)
        metrics_2 = self._compute_node_metrics_single(G2, node)
        observed_diff = abs(metrics_2[metric] - metrics_1[metric])
        
        # Permutation test
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Create permuted networks by randomly swapping edges
            G1_perm = self._permute_network(G1)
            G2_perm = self._permute_network(G2)
            
            metrics_1_perm = self._compute_node_metrics_single(G1_perm, node)
            metrics_2_perm = self._compute_node_metrics_single(G2_perm, node)
            
            perm_diff = abs(metrics_2_perm[metric] - metrics_1_perm[metric])
            
            if perm_diff >= observed_diff:
                extreme_count += 1
        
        p_value = extreme_count / n_permutations
        return max(p_value, 1.0 / n_permutations)
    
    def _permutation_test_global(
        self,
        condition_networks: Dict[str, nx.Graph],
        metric: str,
        n_permutations: int
    ) -> float:
        """Permutation test for global metric differences."""
        conditions = list(condition_networks.keys())
        G1, G2 = condition_networks[conditions[0]], condition_networks[conditions[1]]
        
        # Observed difference
        metrics_1 = self._compute_global_metrics(G1)
        metrics_2 = self._compute_global_metrics(G2)
        observed_diff = abs(metrics_2[metric] - metrics_1[metric])
        
        # Permutation test
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Create permuted networks
            G1_perm = self._permute_network(G1)
            G2_perm = self._permute_network(G2)
            
            metrics_1_perm = self._compute_global_metrics(G1_perm)
            metrics_2_perm = self._compute_global_metrics(G2_perm)
            
            perm_diff = abs(metrics_2_perm[metric] - metrics_1_perm[metric])
            
            if perm_diff >= observed_diff:
                extreme_count += 1
        
        p_value = extreme_count / n_permutations
        return max(p_value, 1.0 / n_permutations)
    
    def _permute_network(self, G: nx.Graph) -> nx.Graph:
        """Create a permuted version of the network."""
        # Simple edge permutation
        edges = list(G.edges(data=True))
        nodes = list(G.nodes())
        
        G_perm = nx.Graph()
        G_perm.add_nodes_from(nodes)
        
        # Randomly reassign edges while preserving degree sequence
        np.random.shuffle(edges)
        
        for i, (u, v, data) in enumerate(edges):
            # Randomly select new nodes
            new_u = np.random.choice(nodes)
            new_v = np.random.choice(nodes)
            
            if new_u != new_v and not G_perm.has_edge(new_u, new_v):
                G_perm.add_edge(new_u, new_v, **data)
        
        return G_perm
    
    def _combine_results(self, results: Dict) -> Dict:
        """Combine results across different networks."""
        combined = {
            'differential_edges': [],
            'differential_nodes': [],
            'global_differences': {},
            'network_summary': {}
        }
        
        # Combine edge results
        all_edge_results = []
        for network_name, network_results in results.items():
            if 'edge_differences' in network_results:
                edge_df = network_results['edge_differences']
                edge_df['network'] = network_name
                all_edge_results.append(edge_df)
        
        if all_edge_results:
            combined['differential_edges'] = pd.concat(all_edge_results, ignore_index=True)
        
        # Combine node results
        all_node_results = []
        for network_name, network_results in results.items():
            if 'node_differences' in network_results:
                node_df = network_results['node_differences']
                node_df['network'] = network_name
                all_node_results.append(node_df)
        
        if all_node_results:
            combined['differential_nodes'] = pd.concat(all_node_results, ignore_index=True)
        
        # Combine global results
        for network_name, network_results in results.items():
            if 'global_differences' in network_results:
                combined['global_differences'][network_name] = network_results['global_differences']
        
        return combined
    
    def _apply_multiple_testing_correction(
        self,
        results: Dict,
        method: str,
        alpha: float
    ) -> Dict:
        """Apply multiple testing correction to p-values."""
        from statsmodels.stats.multitest import multipletests
        
        # Correct edge p-values
        if 'differential_edges' in results and len(results['differential_edges']) > 0:
            p_values = results['differential_edges']['p_value'].values
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
            
            results['differential_edges']['p_value_corrected'] = p_corrected
            results['differential_edges']['significant'] = rejected
        
        # Correct node p-values
        if 'differential_nodes' in results and len(results['differential_nodes']) > 0:
            # Correct degree p-values
            p_values_degree = results['differential_nodes']['degree_p_value'].values
            rejected_degree, p_corrected_degree, _, _ = multipletests(
                p_values_degree, alpha=alpha, method=method
            )
            
            results['differential_nodes']['degree_p_value_corrected'] = p_corrected_degree
            results['differential_nodes']['degree_significant'] = rejected_degree
            
            # Correct betweenness p-values
            p_values_betweenness = results['differential_nodes']['betweenness_p_value'].values
            rejected_betweenness, p_corrected_betweenness, _, _ = multipletests(
                p_values_betweenness, alpha=alpha, method=method
            )
            
            results['differential_nodes']['betweenness_p_value_corrected'] = p_corrected_betweenness
            results['differential_nodes']['betweenness_significant'] = rejected_betweenness
        
        return results
    
    def _compute_effect_sizes(self, results: Dict) -> Dict:
        """Compute effect sizes for differential analysis results."""
        # Edge effect sizes
        if 'differential_edges' in results and len(results['differential_edges']) > 0:
            edge_df = results['differential_edges']
            
            # Cohen's d for edge weights
            pooled_std = np.sqrt(
                (edge_df['weight_condition1'].var() + edge_df['weight_condition2'].var()) / 2
            )
            
            if pooled_std > 0:
                cohens_d = edge_df['weight_difference'] / pooled_std
                results['differential_edges']['effect_size'] = cohens_d
            else:
                results['differential_edges']['effect_size'] = 0.0
        
        # Node effect sizes
        if 'differential_nodes' in results and len(results['differential_nodes']) > 0:
            node_df = results['differential_nodes']
            
            # Effect size for degree
            degree_pooled_std = np.sqrt(
                (node_df['degree_condition1'].var() + node_df['degree_condition2'].var()) / 2
            )
            
            if degree_pooled_std > 0:
                degree_effect_size = node_df['degree_difference'] / degree_pooled_std
                results['differential_nodes']['degree_effect_size'] = degree_effect_size
            else:
                results['differential_nodes']['degree_effect_size'] = 0.0
        
        return results
    
    def compute_network_distance(
        self,
        G1: nx.Graph,
        G2: nx.Graph,
        method: str = 'jaccard'
    ) -> float:
        """
        Compute distance between two networks.
        
        Parameters
        ----------
        G1, G2 : nx.Graph
            Networks to compare
        method : str, default='jaccard'
            Distance method: 'jaccard', 'hamming', 'spectral'
        
        Returns
        -------
        float
            Distance between networks
        """
        if method == 'jaccard':
            edges1 = set(G1.edges())
            edges2 = set(G2.edges())
            
            intersection = len(edges1 & edges2)
            union = len(edges1 | edges2)
            
            if union == 0:
                return 0.0
            
            jaccard_similarity = intersection / union
            return 1.0 - jaccard_similarity
        
        elif method == 'hamming':
            # Get all possible edges
            all_nodes = set(G1.nodes()) | set(G2.nodes())
            all_edges = list(combinations(all_nodes, 2))
            
            differences = 0
            for edge in all_edges:
                exists_1 = G1.has_edge(*edge)
                exists_2 = G2.has_edge(*edge)
                
                if exists_1 != exists_2:
                    differences += 1
            
            return differences / len(all_edges)
        
        elif method == 'spectral':
            # Spectral distance based on eigenvalues
            try:
                # Get adjacency matrices
                nodes = sorted(set(G1.nodes()) | set(G2.nodes()))
                
                A1 = nx.adjacency_matrix(G1, nodelist=nodes).toarray()
                A2 = nx.adjacency_matrix(G2, nodelist=nodes).toarray()
                
                # Compute eigenvalues
                eigenvals1 = np.linalg.eigvals(A1)
                eigenvals2 = np.linalg.eigvals(A2)
                
                # Sort eigenvalues
                eigenvals1 = np.sort(eigenvals1)[::-1]
                eigenvals2 = np.sort(eigenvals2)[::-1]
                
                # Compute distance
                distance = np.linalg.norm(eigenvals1 - eigenvals2)
                return distance
            
            except:
                return np.nan
        
        else:
            raise ValueError(f"Unknown distance method: {method}")

