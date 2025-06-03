#!/usr/bin/env python3
"""
Circuit Reducer: Automatically simplifies resistor networks using graph theory

This script implements the exact algorithm shown in the user's example for
reducing resistor networks by combining series and parallel resistors.
It focuses on the step-by-step reduction process and visualization of each step.

Usage:
    python circuit_reducer.py

Author: Don Ishmand
"""

import networkx as nx
import matplotlib.pyplot as plt
import os
from copy import deepcopy

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

def draw_graph(G, step_count, description):
    """
    Draw the circuit graph with resistor values.
    
    Parameters:
    -----------
    G : NetworkX graph
        The graph to draw
    step_count : int
        Current step number
    description : str
        Description of the current step
    """
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=700, font_weight='bold')
    
    # Draw edge labels (resistance values)
    edge_labels = {(u, v): f"{d['resistance']:.2f} Ω" 
                  for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(f"Step {step_count}: {description}")
    plt.tight_layout()
    plt.savefig(f"../figures/step_{step_count:02d}_{description.replace(' ', '_').lower()}.png", 
                dpi=300, bbox_inches='tight')
    plt.close()

def combine_series(G):
    """
    Combine resistors in series.
    
    This function implements the exact algorithm from the user's example:
    - Find nodes with degree 2 (not terminals)
    - Combine the two adjacent resistors
    - Remove the node and add a new edge with the combined resistance
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    int : Number of steps performed
    """
    changed = True
    step_count = 0
    
    while changed:
        changed = False
        
        for node in list(G.nodes()):
            # Skip terminal nodes (labeled "B+" or "B-")
            if node in ("B+", "B-"):
                continue
                
            # Check if node has exactly two connections (degree 2)
            if G.degree(node) == 2:
                neighbors = list(G.neighbors(node))
                u, v = neighbors
                
                # Get edge data (resistance values)
                edge1_data = G.get_edge_data(u, node)
                edge2_data = G.get_edge_data(node, v)
                
                # Handle MultiGraph edges by taking the first edge
                edge1_key = list(edge1_data.keys())[0]
                edge2_key = list(edge2_data.keys())[0]
                
                edge1 = edge1_data[edge1_key]
                edge2 = edge2_data[edge2_key]
                
                # Calculate combined resistance
                R_new = edge1['resistance'] + edge2['resistance']
                
                # Remove node and add direct edge with combined resistance
                G.remove_node(node)
                G.add_edge(u, v, resistance=R_new)
                
                changed = True
                step_count += 1
                
                # Draw the graph after this step
                draw_graph(G, step_count, f"Series {u} - {v} = {R_new:.2f}Ω")
                
                break
    
    return step_count

def combine_parallel(G):
    """
    Combine resistors in parallel.
    
    This function implements the exact algorithm from the user's example:
    - Find parallel edges between nodes
    - Calculate the equivalent resistance
    - Replace multiple edges with a single edge
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    int : Number of steps performed
    """
    changed = True
    step_count = 0
    
    while changed:
        changed = False
        seen = set()
        
        # First check for direct parallel edges
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u >= v:  # Skip to avoid processing edges twice
                    continue
                    
                # Skip if no edge exists
                if not G.has_edge(u, v):
                    continue
                    
                # Count edges between these nodes
                if G.number_of_edges(u, v) > 1:
                    # Get all resistances between these nodes
                    resistances = [d['resistance'] for d in G.get_edge_data(u, v).values()]
                    
                    # Calculate parallel resistance
                    R_parallel = 1 / sum(1/r for r in resistances)
                    
                    # Remove all edges between these nodes
                    G.remove_edges_from([(u, v, key) for key in G.get_edge_data(u, v).keys()])
                    
                    # Add a single edge with the equivalent resistance
                    G.add_edge(u, v, resistance=R_parallel)
                    
                    changed = True
                    step_count += 1
                    
                    # Draw the graph after this step
                    draw_graph(G, step_count, f"Parallel {u} - {v} = {R_parallel:.2f}Ω")
                    
                    break
            
            if changed:
                break
    
    return step_count

def reduce_circuit(G):
    """
    Reduce a circuit by applying series and parallel reductions until no more can be applied.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    int : Total number of reduction steps
    """
    total_steps = 0
    
    # Initial circuit
    draw_graph(G, 0, "Initial Circuit")
    
    # Continue applying reductions until no more changes
    while True:
        # Try series reduction
        series_steps = combine_series(G)
        total_steps += series_steps
        
        # Try parallel reduction
        parallel_steps = combine_parallel(G)
        total_steps += parallel_steps
        
        # If no reductions were made, we're done
        if series_steps == 0 and parallel_steps == 0:
            break
    
    return total_steps

def create_example_circuit():
    """
    Create an example circuit with both series and parallel resistors.
    
    Returns:
    --------
    G : NetworkX graph
        Example circuit graph
    """
    G = nx.MultiGraph()
    
    # Add resistors in the first path
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("A", "B", resistance=20)
    G.add_edge("B", "C", resistance=30)
    G.add_edge("C", "B-", resistance=40)
    
    # Add parallel path
    G.add_edge("B+", "D", resistance=50)
    G.add_edge("D", "E", resistance=60)
    G.add_edge("E", "B-", resistance=70)
    
    # Add cross connections
    G.add_edge("A", "D", resistance=80)
    G.add_edge("B", "E", resistance=90)
    
    return G

def create_wheatstone_bridge():
    """
    Create a Wheatstone bridge circuit.
    
    Returns:
    --------
    G : NetworkX graph
        Wheatstone bridge circuit graph
    """
    G = nx.MultiGraph()
    
    # Add the bridge structure
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("B+", "B", resistance=20)
    G.add_edge("A", "C", resistance=30)
    G.add_edge("B", "C", resistance=40)
    G.add_edge("A", "B", resistance=50)  # Bridge resistor
    G.add_edge("C", "B-", resistance=60)
    
    return G

def create_complex_circuit():
    """
    Create a complex circuit with multiple series and parallel paths.
    
    Returns:
    --------
    G : NetworkX graph
        Complex circuit graph
    """
    G = nx.MultiGraph()
    
    # Main path
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("A", "B", resistance=15)
    G.add_edge("B", "C", resistance=20)
    G.add_edge("C", "D", resistance=25)
    G.add_edge("D", "B-", resistance=30)
    
    # Parallel paths
    G.add_edge("B+", "E", resistance=35)
    G.add_edge("E", "F", resistance=40)
    G.add_edge("F", "B-", resistance=45)
    
    # Cross connections
    G.add_edge("A", "E", resistance=50)
    G.add_edge("B", "F", resistance=55)
    G.add_edge("C", "E", resistance=60)
    
    # Additional parallel paths
    G.add_edge("A", "C", resistance=65)
    G.add_edge("B", "D", resistance=70)
    
    # Multiple parallel resistors
    G.add_edge("C", "D", resistance=75)
    G.add_edge("C", "D", resistance=80)  # Second resistor between C and D
    
    return G

def analyze_circuit(G, name):
    """
    Analyze a circuit by reducing it and calculating the equivalent resistance.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit to analyze
    name : str
        Name of the circuit for reporting
    """
    print(f"\n=== Analyzing {name} ===")
    print(f"Initial circuit: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Make a copy to avoid modifying the original
    G_copy = deepcopy(G)
    
    # Reduce the circuit
    steps = reduce_circuit(G_copy)
    
    # Calculate final resistance
    if G_copy.has_edge("B+", "B-"):
        # Get resistance value from the first edge (there should be only one after reduction)
        edge_data = G_copy.get_edge_data("B+", "B-")
        key = list(edge_data.keys())[0]
        resistance = edge_data[key]["resistance"]
        print(f"Equivalent resistance: {resistance:.2f} Ω")
    else:
        print("No path between terminals after reduction!")
        
        # Debug information
        print("Remaining nodes:", list(G_copy.nodes()))
        print("Remaining edges:", [(u, v) for u, v in G_copy.edges()])
    
    print(f"Reduction completed in {steps} steps")
    print(f"Final circuit: {G_copy.number_of_nodes()} nodes, {G_copy.number_of_edges()} edges")

if __name__ == "__main__":
    print("=== Circuit Reducer: Automatic Simplification of Resistor Networks ===")
    
    # Example 1: Simple circuit
    simple_circuit = create_example_circuit()
    analyze_circuit(simple_circuit, "Simple Circuit")
    
    # Example 2: Wheatstone bridge
    bridge = create_wheatstone_bridge()
    analyze_circuit(bridge, "Wheatstone Bridge")
    
    # Example 3: Complex circuit
    complex_circuit = create_complex_circuit()
    analyze_circuit(complex_circuit, "Complex Circuit")
    
    print("\nAll circuit analyses complete. Figures saved to '../figures/' directory.")
