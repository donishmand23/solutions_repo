#!/usr/bin/env python3
"""
Resistor Network Solver: A graph-based approach to simplifying resistor networks

This script implements the exact algorithm shown in the problem statement for
reducing resistor networks by combining series and parallel resistors.
It can handle any configuration of resistors and automatically simplify the network
according to the laws of resistor simplification.

Usage:
    python resistor_network_solver.py

Author: Don Ishmand
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

def draw_graph(G, step_count, title, filename=None):
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
    filename : str, optional
        If provided, save figure to this path
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
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"../figures/step_{step_count:02d}_{title.replace(' ', '_').lower()}.png", 
                   dpi=300, bbox_inches='tight')
    
    plt.close()

def combine_series(G):
    """
    Combine resistors in series.
    
    Implements the algorithm from the problem statement:
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
                
                # Get resistance values
                r1 = G[node][u]['resistance']
                r2 = G[node][v]['resistance']
                
                # Calculate combined resistance
                r_new = r1 + r2
                
                # Remove node and add direct edge with combined resistance
                G.remove_node(node)
                
                # Add new edge or combine with existing
                if G.has_edge(u, v):
                    # Calculate parallel resistance with existing edge
                    r_existing = G[u][v]['resistance']
                    r_combined = 1.0 / (1.0/r_existing + 1.0/r_new)
                    G[u][v]['resistance'] = r_combined
                else:
                    G.add_edge(u, v, resistance=r_new)
                
                changed = True
                step_count += 1
                
                # Draw the graph after this step
                draw_graph(G, step_count, f"Series Reduction: {node} removed, {u}-{v}={r_new:.1f}Ω")
                
                break
    
    return step_count

def combine_parallel(G):
    """
    Combine resistors in parallel.
    
    Implements the algorithm from the problem statement:
    - Find nodes with multiple connections between them
    - Calculate the equivalent resistance using the parallel formula
    - Replace with a single equivalent resistor
    
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
    seen = set()
    
    while changed:
        changed = False
        seen.clear()
        
        for u, v in list(G.edges()):
            # Skip edges we've already processed
            if (u, v) in seen or (v, u) in seen:
                continue
            
            seen.add((u, v))
            
            # Check if there are multiple edges between these nodes
            if G.number_of_edges(u, v) > 1:
                # Get all resistances between these nodes
                resistances = []
                for _, _, k, data in G.edges(u, v, keys=True, data=True):
                    resistances.append(data['resistance'])
                
                # Calculate parallel resistance
                total_conductance = sum(1.0/r for r in resistances)
                r_parallel = 1.0 / total_conductance
                
                # Remove all edges between these nodes
                while G.has_edge(u, v):
                    G.remove_edge(u, v)
                
                # Add a single edge with the equivalent resistance
                G.add_edge(u, v, resistance=r_parallel)
                
                changed = True
                step_count += 1
                
                # Draw the graph after this step
                draw_graph(G, step_count, f"Parallel Reduction: {u}-{v}={r_parallel:.1f}Ω")
                
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

def create_circuit_from_edges(edges):
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
        G.add_edge(u, v, resistance=r)
    return G

def example_simple_series():
    """Example: Simple series circuit"""
    print("\n=== Simple Series Circuit ===")
    edges = [
        ("B+", "A", 10),
        ("A", "B", 20),
        ("B", "C", 30),
        ("C", "B-", 40)
    ]
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω (expected: 100Ω)")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

def example_simple_parallel():
    """Example: Simple parallel circuit"""
    print("\n=== Simple Parallel Circuit ===")
    edges = [
        ("B+", "B-", 10),
        ("B+", "B-", 20),
        ("B+", "B-", 30)
    ]
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω (expected: 5.45Ω)")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

def example_series_parallel():
    """Example: Series-parallel circuit"""
    print("\n=== Series-Parallel Circuit ===")
    edges = [
        # First path: 10Ω + 20Ω
        ("B+", "A", 10),
        ("A", "B-", 20),
        
        # Second path: 30Ω + 40Ω
        ("B+", "C", 30),
        ("C", "B-", 40)
    ]
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω (expected: 17.14Ω)")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

def example_wheatstone_bridge():
    """Example: Wheatstone bridge circuit"""
    print("\n=== Wheatstone Bridge Circuit ===")
    edges = [
        ("B+", "A", 10),
        ("B+", "B", 20),
        ("A", "C", 30),
        ("B", "C", 40),
        ("A", "B", 50),  # Bridge resistor
        ("C", "B-", 60)
    ]
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

def example_complex_circuit():
    """Example: Complex circuit with multiple series and parallel paths"""
    print("\n=== Complex Circuit ===")
    edges = [
        # Main path
        ("B+", "A", 10),
        ("A", "B", 20),
        ("B", "C", 30),
        ("C", "B-", 40),
        
        # Parallel path
        ("B+", "D", 50),
        ("D", "E", 60),
        ("E", "B-", 70),
        
        # Cross connections
        ("A", "D", 80),
        ("B", "E", 90)
    ]
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

def example_ladder_circuit(n=3):
    """Example: Ladder circuit with n rungs"""
    print(f"\n=== Ladder Circuit ({n} rungs) ===")
    edges = []
    
    # Create ladder structure
    for i in range(n+1):
        # Add horizontal rungs (except for the last node)
        if i < n:
            edges.append((f"A{i}", f"A{i+1}", 10))  # Top rail
            edges.append((f"B{i}", f"B{i+1}", 10))  # Bottom rail
        
        # Add vertical rungs
        edges.append((f"A{i}", f"B{i}", 20))  # Vertical connections
    
    # Connect to terminals
    edges.append(("B+", "A0", 5))
    edges.append((f"A{n}", "B-", 5))
    
    G = create_circuit_from_edges(edges)
    steps = reduce_circuit(G)
    
    # Calculate final resistance
    if G.has_edge("B+", "B-"):
        r_eq = G["B+"]["B-"]["resistance"]
        print(f"Equivalent resistance: {r_eq:.1f}Ω")
    else:
        print("Error: No path between terminals after reduction!")
    
    print(f"Reduction completed in {steps} steps")
    return G

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
    G_series = example_simple_series()
    if G_series.has_edge("B+", "B-"):
        results["Series"] = G_series["B+"]["B-"]["resistance"]
    
    # Simple parallel circuit
    G_parallel = example_simple_parallel()
    if G_parallel.has_edge("B+", "B-"):
        results["Parallel"] = G_parallel["B+"]["B-"]["resistance"]
    
    # Series-parallel circuit
    G_series_parallel = example_series_parallel()
    if G_series_parallel.has_edge("B+", "B-"):
        results["Series-Parallel"] = G_series_parallel["B+"]["B-"]["resistance"]
    
    # Wheatstone bridge
    G_bridge = example_wheatstone_bridge()
    if G_bridge.has_edge("B+", "B-"):
        results["Wheatstone Bridge"] = G_bridge["B+"]["B-"]["resistance"]
    
    # Complex circuit
    G_complex = example_complex_circuit()
    if G_complex.has_edge("B+", "B-"):
        results["Complex"] = G_complex["B+"]["B-"]["resistance"]
    
    # Ladder circuit
    G_ladder = example_ladder_circuit(3)
    if G_ladder.has_edge("B+", "B-"):
        results["Ladder (3 rungs)"] = G_ladder["B+"]["B-"]["resistance"]
    
    # Create comparison chart
    if results:
        create_comparison_chart(results)

if __name__ == "__main__":
    print("=== Resistor Network Solver ===")
    print("Implementing the algorithm from the problem statement")
    
    run_all_examples()
