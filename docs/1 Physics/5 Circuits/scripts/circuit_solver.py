#!/usr/bin/env python3
"""
Circuit Solver: A graph-based approach to solving resistor networks

This script implements algorithms to calculate the equivalent resistance of any
resistor network using graph theory. It can:
1. Perform series reduction of resistors
2. Perform parallel reduction of resistors
3. Apply Y-Delta transformations when needed
4. Visualize the reduction process step-by-step
5. Handle arbitrary circuit configurations

Usage:
    python circuit_solver.py

Author: Don Ishmand
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import math

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

class CircuitSolver:
    """
    A class to solve resistor networks using graph theory.
    """
    def __init__(self, verbose=True, visualize=True):
        """
        Initialize the circuit solver.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed information during solving
        visualize : bool
            Whether to generate visualization of the reduction process
        """
        self.verbose = verbose
        self.visualize = visualize
        self.step_count = 0
        self.reduction_history = []
    
    def create_circuit(self, edges):
        """
        Create a circuit from a list of edges with resistance values.
        
        Parameters:
        -----------
        edges : list of tuples
            Each tuple is (node1, node2, resistance)
            
        Returns:
        --------
        G : NetworkX graph
            Graph representing the circuit
        """
        G = nx.Graph()
        for u, v, r in edges:
            # Handle parallel resistors by adding their conductances
            if G.has_edge(u, v):
                old_r = G[u][v]['resistance']
                # Calculate equivalent resistance in parallel
                new_r = 1.0 / (1.0/old_r + 1.0/r)
                G[u][v]['resistance'] = new_r
            else:
                G.add_edge(u, v, resistance=r)
        
        if self.verbose:
            print(f"Created circuit with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def solve(self, G, source, target):
        """
        Solve the circuit to find equivalent resistance between source and target.
        
        Parameters:
        -----------
        G : NetworkX graph
            Graph representing the circuit
        source : node
            Source node
        target : node
            Target node
            
        Returns:
        --------
        resistance : float
            Equivalent resistance between source and target
        """
        # Reset step count and history
        self.step_count = 0
        self.reduction_history = []
        
        # Make a deep copy to avoid modifying the original graph
        circuit = deepcopy(G)
        
        # Store initial state
        if self.visualize:
            self._save_state(circuit, "Initial Circuit")
        
        # Continue reduction until we can't simplify further
        while True:
            # Try series reduction
            series_reduced = self._apply_series_reductions(circuit, source, target)
            if series_reduced:
                continue
                
            # Try parallel reduction
            parallel_reduced = self._apply_parallel_reductions(circuit)
            if parallel_reduced:
                continue
                
            # Try Y-Delta transformation
            y_delta_reduced = self._apply_y_delta_transformations(circuit, source, target)
            if y_delta_reduced:
                continue
                
            # If we reach here, no further reduction is possible
            break
        
        # Calculate final resistance
        if circuit.has_edge(source, target):
            resistance = circuit[source][target]['resistance']
            if self.verbose:
                print(f"Equivalent resistance between nodes {source} and {target}: {resistance:.4f} Ω")
            return resistance
        else:
            if self.verbose:
                print(f"No connection between nodes {source} and {target}")
            return float('inf')
    
    def _apply_series_reductions(self, G, source, target):
        """
        Apply all possible series reductions.
        
        Parameters:
        -----------
        G : NetworkX graph
            Graph representing the circuit
        source : node
            Source node (to be preserved)
        target : node
            Target node (to be preserved)
            
        Returns:
        --------
        reduced : bool
            Whether any reduction was applied
        """
        for node in list(G.nodes()):
            # Skip source and target nodes
            if node == source or node == target:
                continue
                
            # Check if node has exactly two neighbors (degree 2)
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                n1, n2 = neighbors[0], neighbors[1]
                
                r1 = G[node][n1]['resistance']
                r2 = G[node][n2]['resistance']
                
                # Calculate combined resistance
                r_combined = r1 + r2
                
                # Remove node and add direct edge with combined resistance
                G.remove_node(node)
                
                # Add new edge (or update existing)
                if G.has_edge(n1, n2):
                    # Calculate parallel resistance with existing edge
                    r_existing = G[n1][n2]['resistance']
                    r_new = 1.0 / (1.0/r_existing + 1.0/(r1 + r2))
                    G[n1][n2]['resistance'] = r_new
                else:
                    G.add_edge(n1, n2, resistance=r_combined)
                
                self.step_count += 1
                if self.verbose:
                    print(f"Step {self.step_count}: Series reduction at node {node}. "
                          f"Combined {r1} Ω and {r2} Ω to {r_combined} Ω between nodes {n1} and {n2}")
                
                if self.visualize:
                    self._save_state(G, f"After Series Reduction (Node {node})")
                
                return True
        
        return False
    
    def _apply_parallel_reductions(self, G):
        """
        Apply all possible parallel reductions.
        
        Parameters:
        -----------
        G : NetworkX graph
            Graph representing the circuit
            
        Returns:
        --------
        reduced : bool
            Whether any reduction was applied
        """
        # Find all multi-edges (parallel resistors)
        parallel_edges = {}
        for u, v, data in G.edges(data=True):
            if u > v:  # Ensure consistent ordering
                u, v = v, u
            edge = (u, v)
            if edge not in parallel_edges:
                parallel_edges[edge] = []
            parallel_edges[edge].append(data['resistance'])
        
        # Apply parallel reduction where multiple edges exist
        for (u, v), resistances in parallel_edges.items():
            if len(resistances) > 1:
                # Calculate equivalent resistance
                conductance = sum(1.0/r for r in resistances)
                r_equivalent = 1.0 / conductance
                
                # Remove all edges and add a single equivalent edge
                for _ in range(len(resistances)):
                    if G.has_edge(u, v):
                        G.remove_edge(u, v)
                
                G.add_edge(u, v, resistance=r_equivalent)
                
                self.step_count += 1
                if self.verbose:
                    print(f"Step {self.step_count}: Parallel reduction between nodes {u} and {v}. "
                          f"Combined {resistances} to {r_equivalent:.4f} Ω")
                
                if self.visualize:
                    self._save_state(G, f"After Parallel Reduction ({u}-{v})")
                
                return True
        
        return False
    
    def _apply_y_delta_transformations(self, G, source, target):
        """
        Apply Y-Delta transformations where needed.
        
        Parameters:
        -----------
        G : NetworkX graph
            Graph representing the circuit
        source : node
            Source node (to be preserved)
        target : node
            Target node (to be preserved)
            
        Returns:
        --------
        transformed : bool
            Whether any transformation was applied
        """
        # Look for Y configurations (star nodes with degree 3)
        for node in list(G.nodes()):
            # Skip source and target nodes
            if node == source or node == target:
                continue
                
            # Check if node has exactly three neighbors (degree 3)
            if G.degree(node) == 3:
                neighbors = list(G.neighbors(node))
                a, b, c = neighbors
                
                # Get resistances in the Y configuration
                r1 = G[node][a]['resistance']
                r2 = G[node][b]['resistance']
                r3 = G[node][c]['resistance']
                
                # Calculate Delta (triangle) resistances
                r_sum = r1 + r2 + r3
                r_ab = (r1 * r2 + r2 * r3 + r3 * r1) / r3
                r_bc = (r1 * r2 + r2 * r3 + r3 * r1) / r1
                r_ca = (r1 * r2 + r2 * r3 + r3 * r1) / r2
                
                # Remove the Y node
                G.remove_node(node)
                
                # Add the Delta edges (or combine with existing edges)
                self._add_or_combine_edge(G, a, b, r_ab)
                self._add_or_combine_edge(G, b, c, r_bc)
                self._add_or_combine_edge(G, c, a, r_ca)
                
                self.step_count += 1
                if self.verbose:
                    print(f"Step {self.step_count}: Y-Delta transformation at node {node}. "
                          f"Transformed Y ({r1}, {r2}, {r3}) to Delta ({r_ab:.4f}, {r_bc:.4f}, {r_ca:.4f})")
                
                if self.visualize:
                    self._save_state(G, f"After Y-Delta Transformation (Node {node})")
                
                return True
        
        return False
    
    def _add_or_combine_edge(self, G, u, v, resistance):
        """
        Add a new edge or combine with existing edge in parallel.
        
        Parameters:
        -----------
        G : NetworkX graph
            Graph to modify
        u, v : nodes
            Nodes to connect
        resistance : float
            Resistance value for the new edge
        """
        if G.has_edge(u, v):
            # Calculate parallel resistance
            r_existing = G[u][v]['resistance']
            r_new = 1.0 / (1.0/r_existing + 1.0/resistance)
            G[u][v]['resistance'] = r_new
        else:
            G.add_edge(u, v, resistance=resistance)
    
    def _save_state(self, G, description):
        """
        Save the current state of the circuit for visualization.
        
        Parameters:
        -----------
        G : NetworkX graph
            Current state of the circuit
        description : str
            Description of the current state
        """
        self.reduction_history.append({
            'graph': deepcopy(G),
            'description': description
        })
    
    def visualize_reduction_process(self, source, target, filename='../figures/reduction_process.png'):
        """
        Visualize the reduction process step by step.
        
        Parameters:
        -----------
        source : node
            Source node
        target : node
            Target node
        filename : str
            Output filename for the visualization
        """
        if not self.reduction_history:
            print("No reduction history available. Run solve() first.")
            return
        
        n_steps = len(self.reduction_history)
        n_cols = min(3, n_steps)
        n_rows = math.ceil(n_steps / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Set consistent node positions across all steps
        pos = nx.spring_layout(self.reduction_history[0]['graph'], seed=42)
        
        for i, state in enumerate(self.reduction_history):
            ax = axes[i]
            G = state['graph']
            description = state['description']
            
            # Update positions for nodes that still exist
            current_pos = {node: pos[node] for node in G.nodes() if node in pos}
            
            # Draw the graph
            nx.draw(G, pos=current_pos, with_labels=True, node_color='skyblue', 
                    node_size=700, font_weight='bold', ax=ax)
            
            # Highlight source and target
            if source in G.nodes() and target in G.nodes():
                nx.draw_networkx_nodes(G, current_pos, nodelist=[source, target], 
                                      node_color='red', node_size=700, ax=ax)
            
            # Draw edge labels (resistance values)
            edge_labels = {(u, v): f"{d['resistance']:.2f} Ω" 
                          for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, current_pos, edge_labels=edge_labels, ax=ax)
            
            ax.set_title(description)
            ax.set_axis_off()
        
        # Hide any unused axes
        for i in range(len(self.reduction_history), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        
        return fig

def example_simple_circuit():
    """
    Example: Simple series-parallel circuit.
    """
    solver = CircuitSolver()
    
    # Create a simple circuit: two parallel branches, each with two resistors in series
    edges = [
        ('A', 'B', 10),  # 10 ohm resistor from A to B
        ('B', 'C', 20),  # 20 ohm resistor from B to C
        ('A', 'D', 30),  # 30 ohm resistor from A to D
        ('D', 'C', 40),  # 40 ohm resistor from D to C
    ]
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A', 'C')
    
    print(f"Equivalent resistance of simple circuit: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A', 'C', '../figures/simple_circuit_reduction.png')
    
    return resistance

def example_wheatstone_bridge():
    """
    Example: Wheatstone bridge circuit.
    """
    solver = CircuitSolver()
    
    # Create a Wheatstone bridge circuit
    edges = [
        ('A', 'B', 10),  # 10 ohm
        ('B', 'C', 20),  # 20 ohm
        ('A', 'D', 30),  # 30 ohm
        ('D', 'C', 40),  # 40 ohm
        ('B', 'D', 50),  # 50 ohm (bridge)
    ]
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A', 'C')
    
    print(f"Equivalent resistance of Wheatstone bridge: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A', 'C', '../figures/wheatstone_bridge_reduction.png')
    
    return resistance

def example_complex_network():
    """
    Example: Complex resistor network.
    """
    solver = CircuitSolver()
    
    # Create a more complex network
    edges = [
        ('A', 'B', 10),
        ('B', 'C', 20),
        ('C', 'D', 30),
        ('D', 'E', 40),
        ('E', 'A', 50),
        ('A', 'C', 60),
        ('A', 'D', 70),
        ('B', 'D', 80),
        ('B', 'E', 90),
        ('C', 'E', 100),
    ]
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A', 'D')
    
    print(f"Equivalent resistance of complex network: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A', 'D', '../figures/complex_network_reduction.png')
    
    return resistance

def example_user_input():
    """
    Example: Get circuit configuration from user input.
    """
    print("\n=== Custom Circuit Solver ===")
    print("Enter resistor connections in format 'node1 node2 resistance'")
    print("Enter 'done' when finished")
    
    edges = []
    while True:
        inp = input("> ")
        if inp.lower() == 'done':
            break
        
        try:
            parts = inp.split()
            if len(parts) != 3:
                print("Invalid format. Use 'node1 node2 resistance'")
                continue
                
            node1, node2, resistance = parts[0], parts[1], float(parts[2])
            edges.append((node1, node2, resistance))
            print(f"Added resistor: {node1} -- {resistance} Ω --> {node2}")
        except ValueError:
            print("Invalid resistance value. Must be a number.")
    
    if not edges:
        print("No resistors added.")
        return
    
    source = input("Enter source node: ")
    target = input("Enter target node: ")
    
    solver = CircuitSolver()
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, source, target)
    
    print(f"Equivalent resistance between {source} and {target}: {resistance:.4f} Ω")
    solver.visualize_reduction_process(source, target, '../figures/custom_circuit_reduction.png')
    
    return resistance

if __name__ == "__main__":
    print("=== Circuit Solver: Graph-Based Resistor Network Analysis ===")
    print("\n1. Simple Series-Parallel Circuit")
    example_simple_circuit()
    
    print("\n2. Wheatstone Bridge Circuit")
    example_wheatstone_bridge()
    
    print("\n3. Complex Resistor Network")
    example_complex_network()
    
    # Uncomment to use interactive mode
    # print("\n4. Custom Circuit")
    # example_user_input()
