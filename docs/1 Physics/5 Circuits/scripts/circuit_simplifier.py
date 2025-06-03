#!/usr/bin/env python3
"""
Circuit Simplifier: A graph-based approach to solving resistor networks

This script implements the algorithm described in the problem statement for
simplifying resistor networks by combining series and parallel resistors.
It follows the exact approach shown in the example code and can handle any
configuration of resistors.

Usage:
    python circuit_simplifier.py

Author: Don Ishmand
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

def draw_graph(G, step_count, title):
    """
    Draw the circuit graph with resistor values.
    
    Parameters:
    -----------
    G : NetworkX graph
        The graph to draw
    step_count : int
        Current step number
    title : str
        Title for the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Use spring layout with fixed seed for consistent positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=700, font_weight='bold')
    
    # Highlight terminal nodes
    terminal_nodes = [n for n in G.nodes() if n in ("B+", "B-")]
    if terminal_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=terminal_nodes, 
                              node_color='red', node_size=700)
    
    # Draw edge labels (resistance values)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        edge_labels[(u, v)] = f"{data['resistance']:.1f}Ω"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(f"Step {step_count}: {title}")
    plt.axis('off')
    
    plt.savefig(f"../figures/step_{step_count:02d}_{title.replace(' ', '_').lower()}.png", 
               dpi=300, bbox_inches='tight')
    
    plt.close()

def combine_series(G):
    """
    Combine resistors in series.
    
    This function exactly implements the algorithm from the problem statement:
    - Find nodes with degree 2 (not terminals "B+" or "B-")
    - Combine the two adjacent resistors
    - Remove the node and add a new edge with the combined resistance
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    step_count : int
        Number of steps performed
    """
    changed = True
    step_count = 0
    
    while changed:
        changed = False
        
        for node in list(G.nodes()):
            # Skip terminal nodes
            if node in ("B+", "B-"):
                continue
                
            # Check if node has exactly two connections (degree 2)
            if G.degree(node) == 2:
                # Get the two neighbors
                neighbors = list(G.neighbors(node))
                u, v = neighbors
                
                # Get resistance values
                edge1 = G[u][node]['resistance']
                edge2 = G[node][v]['resistance']
                
                # Calculate new resistance (series: R_new = R1 + R2)
                R_new = edge1 + edge2
                
                # Remove node and add direct edge with combined resistance
                G.remove_node(node)
                
                # Add new edge or combine with existing in parallel
                if G.has_edge(u, v):
                    # Calculate parallel resistance with existing edge
                    r_existing = G[u][v]['resistance']
                    r_combined = 1.0 / (1.0/r_existing + 1.0/R_new)
                    G[u][v]['resistance'] = r_combined
                else:
                    G.add_edge(u, v, resistance=R_new)
                
                changed = True
                step_count += 1
                
                # Draw the graph after this step
                draw_graph(G, step_count, f"Series: {u}-{node}-{v} → {u}-{v}={R_new:.1f}Ω")
                
                break
    
    return step_count

def combine_parallel(G):
    """
    Combine resistors in parallel.
    
    This function exactly implements the algorithm from the problem statement:
    - Find nodes with multiple edges between them
    - Calculate the equivalent resistance using the parallel formula
    - Replace with a single equivalent resistor
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    step_count : int
        Number of steps performed
    """
    changed = True
    step_count = 0
    
    while changed:
        changed = False
        seen = set()
        
        # Check all pairs of nodes for parallel connections
        for u in list(G.nodes()):
            for v in list(G.nodes()):
                if u >= v:  # Skip to avoid processing pairs twice
                    continue
                
                # Skip if no edge exists
                if not G.has_edge(u, v):
                    continue
                
                # Skip if we've already processed this pair
                if (u, v) in seen or (v, u) in seen:
                    continue
                
                seen.add((u, v))
                
                # Check if there are multiple parallel resistors
                if G.number_of_edges(u, v) > 1:
                    # Get all resistances between these nodes
                    resistances = [G[u][v]['resistance']]
                    
                    # Calculate parallel resistance (1/R_eq = 1/R1 + 1/R2 + ...)
                    total_conductance = sum(1.0/r for r in resistances)
                    R_parallel = 1.0 / total_conductance
                    
                    # Update the edge with the equivalent resistance
                    G[u][v]['resistance'] = R_parallel
                    
                    changed = True
                    step_count += 1
                    
                    # Draw the graph after this step
                    draw_graph(G, step_count, f"Parallel: {u}-{v}={R_parallel:.1f}Ω")
                    
                    break
            
            if changed:
                break
    
    return step_count

def simplify_circuit(G):
    """
    Simplify a circuit by applying series and parallel reductions until no more can be applied.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to simplify
        
    Returns:
    --------
    total_steps : int
        Total number of reduction steps
    """
    # Draw initial circuit
    draw_graph(G, 0, "Initial Circuit")
    
    total_steps = 0
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

def create_simple_series_circuit():
    """Create a simple series circuit"""
    G = nx.Graph()
    
    # Add resistors in series
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("A", "B", resistance=20)
    G.add_edge("B", "C", resistance=30)
    G.add_edge("C", "B-", resistance=40)
    
    return G

def create_simple_parallel_circuit():
    """Create a simple parallel circuit"""
    G = nx.Graph()
    
    # Add resistors in parallel (directly between terminals)
    G.add_edge("B+", "B-", resistance=10)
    G.add_edge("B+", "B-", resistance=20)  # This won't work in NetworkX
    
    # Workaround: use a MultiGraph
    G_multi = nx.MultiGraph()
    G_multi.add_edge("B+", "B-", resistance=10)
    G_multi.add_edge("B+", "B-", resistance=20)
    
    return G_multi

def create_series_parallel_circuit():
    """Create a series-parallel circuit"""
    G = nx.Graph()
    
    # First path: 10Ω + 20Ω
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("A", "B-", resistance=20)
    
    # Second path: 30Ω + 40Ω
    G.add_edge("B+", "C", resistance=30)
    G.add_edge("C", "B-", resistance=40)
    
    return G

def create_wheatstone_bridge():
    """Create a Wheatstone bridge circuit"""
    G = nx.Graph()
    
    # Add the bridge structure
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("B+", "B", resistance=20)
    G.add_edge("A", "C", resistance=30)
    G.add_edge("B", "C", resistance=40)
    G.add_edge("A", "B", resistance=50)  # Bridge resistor
    G.add_edge("C", "B-", resistance=60)
    
    return G

def create_complex_circuit():
    """Create a complex circuit with multiple series and parallel paths"""
    G = nx.Graph()
    
    # Main path
    G.add_edge("B+", "A", resistance=10)
    G.add_edge("A", "B", resistance=20)
    G.add_edge("B", "C", resistance=30)
    G.add_edge("C", "B-", resistance=40)
    
    # Parallel path
    G.add_edge("B+", "D", resistance=50)
    G.add_edge("D", "E", resistance=60)
    G.add_edge("E", "B-", resistance=70)
    
    # Cross connections
    G.add_edge("A", "D", resistance=80)
    G.add_edge("B", "E", resistance=90)
    
    return G

def analyze_circuit(G, name):
    """
    Analyze a circuit by simplifying it and calculating the equivalent resistance.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit to analyze
    name : str
        Name of the circuit for reporting
    
    Returns:
    --------
    resistance : float or None
        Equivalent resistance if found, None otherwise
    """
    print(f"\n=== Analyzing {name} ===")
    print(f"Initial circuit: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Make a copy to avoid modifying the original
    G_copy = deepcopy(G)
    
    # Simplify the circuit
    steps = simplify_circuit(G_copy)
    
    # Calculate final resistance
    resistance = None
    if G_copy.has_edge("B+", "B-"):
        resistance = G_copy["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {resistance:.2f} Ω")
    else:
        print("No direct path between terminals after simplification.")
        print("Remaining nodes:", list(G_copy.nodes()))
        print("Remaining edges:", [(u, v) for u, v in G_copy.edges()])
    
    print(f"Simplification completed in {steps} steps")
    
    return resistance

def create_comparison_chart(results):
    """Create a comparison chart of equivalent resistances"""
    plt.figure(figsize=(10, 6))
    
    circuits = list(results.keys())
    resistances = list(results.values())
    
    plt.bar(circuits, resistances, color='skyblue')
    plt.ylabel('Equivalent Resistance (Ω)')
    plt.title('Comparison of Circuit Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('../figures/circuit_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGenerated comparison chart at '../figures/circuit_comparison.png'")

def run_all_examples():
    """Run all example circuits and collect results"""
    results = {}
    
    # Simple series circuit
    G_series = create_simple_series_circuit()
    r_series = analyze_circuit(G_series, "Simple Series Circuit")
    if r_series is not None:
        results["Series"] = r_series
    
    # Series-parallel circuit
    G_series_parallel = create_series_parallel_circuit()
    r_series_parallel = analyze_circuit(G_series_parallel, "Series-Parallel Circuit")
    if r_series_parallel is not None:
        results["Series-Parallel"] = r_series_parallel
    
    # Wheatstone bridge
    G_bridge = create_wheatstone_bridge()
    r_bridge = analyze_circuit(G_bridge, "Wheatstone Bridge Circuit")
    if r_bridge is not None:
        results["Wheatstone Bridge"] = r_bridge
    
    # Complex circuit
    G_complex = create_complex_circuit()
    r_complex = analyze_circuit(G_complex, "Complex Circuit")
    if r_complex is not None:
        results["Complex"] = r_complex
    
    # Create comparison chart
    if results:
        create_comparison_chart(results)

def main():
    """Main function to run the circuit simplifier"""
    print("=== Circuit Simplifier ===")
    print("Implementing the algorithm from the problem statement")
    
    run_all_examples()

if __name__ == "__main__":
    main()
