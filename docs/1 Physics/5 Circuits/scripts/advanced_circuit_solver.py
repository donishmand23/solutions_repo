#!/usr/bin/env python3
"""
Advanced Circuit Solver: A comprehensive graph-based approach to solving resistor networks

This script implements a complete algorithm for simplifying any resistor network
by applying series and parallel reductions, as well as Y-Delta transformations
when needed. It can handle arbitrary circuit configurations and visualizes the
step-by-step reduction process.

Usage:
    python advanced_circuit_solver.py

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

def combine_series(G, source, target):
    """
    Combine resistors in series.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
    source : node
        Source terminal (to preserve)
    target : node
        Target terminal (to preserve)
        
    Returns:
    --------
    changed : bool
        Whether any reduction was applied
    step_count : int
        Number of steps performed
    """
    changed = False
    step_count = 0
    
    for node in list(G.nodes()):
        # Skip terminal nodes
        if node == source or node == target:
            continue
            
        # Check if node has exactly two connections (degree 2)
        if G.degree(node) == 2:
            # Get the two neighbors
            neighbors = list(G.neighbors(node))
            u, v = neighbors
            
            # Get resistance values
            edge1 = G[node][u]['resistance']
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
    
    return changed, step_count

def combine_parallel(G):
    """
    Combine resistors in parallel.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
        
    Returns:
    --------
    changed : bool
        Whether any reduction was applied
    step_count : int
        Number of steps performed
    """
    changed = False
    step_count = 0
    
    # Check all pairs of nodes for parallel connections
    for u in list(G.nodes()):
        for v in list(G.nodes()):
            if u >= v:  # Skip to avoid processing pairs twice
                continue
            
            # Skip if no edge exists
            if not G.has_edge(u, v):
                continue
            
            # Check if there are multiple edges between these nodes
            if isinstance(G, nx.MultiGraph) and G.number_of_edges(u, v) > 1:
                # Get all resistances between these nodes
                resistances = []
                for edge_key in G[u][v]:
                    resistances.append(G[u][v][edge_key]['resistance'])
                
                if not resistances:
                    continue
                
                # Calculate parallel resistance (1/R_eq = 1/R1 + 1/R2 + ...)
                total_conductance = sum(1.0/r for r in resistances)
                if total_conductance == 0:
                    continue
                    
                R_parallel = 1.0 / total_conductance
                
                # Remove all edges between these nodes
                while G.has_edge(u, v):
                    G.remove_edge(u, v)
                
                # Add a single edge with the equivalent resistance
                G.add_edge(u, v, resistance=R_parallel)
                
                changed = True
                step_count += 1
                
                # Draw the graph after this step
                draw_graph(G, step_count, f"Parallel: {u}-{v}={R_parallel:.1f}Ω")
                
                break
        
        if changed:
            break
    
    return changed, step_count

def apply_y_delta(G, source, target):
    """
    Apply Y-Delta transformation to reduce the circuit.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to reduce
    source : node
        Source terminal (to preserve)
    target : node
        Target terminal (to preserve)
        
    Returns:
    --------
    changed : bool
        Whether any transformation was applied
    step_count : int
        Number of steps performed
    """
    changed = False
    step_count = 0
    
    # Look for Y configurations (star nodes with degree 3)
    for node in list(G.nodes()):
        # Skip terminal nodes
        if node == source or node == target:
            continue
            
        # Check if node has exactly three neighbors (degree 3)
        if G.degree(node) == 3:
            # Get the three neighbors
            neighbors = list(G.neighbors(node))
            a, b, c = neighbors
            
            # Get resistances in the Y configuration
            r1 = G[node][a]['resistance']
            r2 = G[node][b]['resistance']
            r3 = G[node][c]['resistance']
            
            # Calculate Delta (triangle) resistances
            r_sum = r1 * r2 + r2 * r3 + r3 * r1
            r_ab = r_sum / r3
            r_bc = r_sum / r1
            r_ca = r_sum / r2
            
            # Remove the Y node
            G.remove_node(node)
            
            # Add the Delta edges (or combine with existing edges)
            if G.has_edge(a, b):
                # Calculate parallel resistance with existing edge
                r_existing = G[a][b]['resistance']
                r_combined = 1.0 / (1.0/r_existing + 1.0/r_ab)
                G[a][b]['resistance'] = r_combined
            else:
                G.add_edge(a, b, resistance=r_ab)
                
            if G.has_edge(b, c):
                # Calculate parallel resistance with existing edge
                r_existing = G[b][c]['resistance']
                r_combined = 1.0 / (1.0/r_existing + 1.0/r_bc)
                G[b][c]['resistance'] = r_combined
            else:
                G.add_edge(b, c, resistance=r_bc)
                
            if G.has_edge(c, a):
                # Calculate parallel resistance with existing edge
                r_existing = G[c][a]['resistance']
                r_combined = 1.0 / (1.0/r_existing + 1.0/r_ca)
                G[c][a]['resistance'] = r_combined
            else:
                G.add_edge(c, a, resistance=r_ca)
            
            changed = True
            step_count += 1
            
            # Draw the graph after this step
            draw_graph(G, step_count, f"Y-Delta: {node} → {a},{b},{c}")
            
            break
    
    return changed, step_count

def simplify_circuit(G, source="B+", target="B-"):
    """
    Simplify a circuit by applying series, parallel, and Y-Delta reductions.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit graph to simplify
    source : node
        Source terminal
    target : node
        Target terminal
        
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
        series_changed, series_steps = combine_series(G, source, target)
        total_steps += series_steps
        
        # Try parallel reduction
        parallel_changed, parallel_steps = combine_parallel(G)
        total_steps += parallel_steps
        
        # Try Y-Delta transformation
        y_delta_changed, y_delta_steps = apply_y_delta(G, source, target)
        total_steps += y_delta_steps
        
        # If no reductions were made, we're done
        if not (series_changed or parallel_changed or y_delta_changed):
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
    G = nx.MultiGraph()
    
    # Add resistors in parallel (directly between terminals)
    G.add_edge("B+", "B-", resistance=10)
    G.add_edge("B+", "B-", resistance=20)
    G.add_edge("B+", "B-", resistance=30)
    
    return G

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

def create_ladder_circuit(n=3):
    """Create a ladder circuit with n rungs"""
    G = nx.Graph()
    
    # Create ladder structure
    for i in range(n+1):
        # Add horizontal rungs (except for the last node)
        if i < n:
            G.add_edge(f"A{i}", f"A{i+1}", resistance=10)  # Top rail
            G.add_edge(f"B{i}", f"B{i+1}", resistance=10)  # Bottom rail
        
        # Add vertical rungs
        G.add_edge(f"A{i}", f"B{i}", resistance=20)  # Vertical connections
    
    # Connect to terminals
    G.add_edge("B+", "A0", resistance=5)
    G.add_edge(f"A{n}", "B-", resistance=5)
    
    return G

def analyze_circuit(G, name, source="B+", target="B-"):
    """
    Analyze a circuit by simplifying it and calculating the equivalent resistance.
    
    Parameters:
    -----------
    G : NetworkX graph
        Circuit to analyze
    name : str
        Name of the circuit for reporting
    source : node
        Source terminal
    target : node
        Target terminal
    
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
    steps = simplify_circuit(G_copy, source, target)
    
    # Calculate final resistance
    resistance = None
    if G_copy.has_edge(source, target):
        if isinstance(G_copy, nx.MultiGraph):
            # For MultiGraph, we need to get the first edge key
            edge_key = list(G_copy[source][target].keys())[0]
            resistance = G_copy[source][target][edge_key]["resistance"]
        else:
            resistance = G_copy[source][target]["resistance"]
        print(f"Equivalent resistance: {resistance:.2f} Ω")
    else:
        print("No direct path between terminals after simplification.")
        print("Remaining nodes:", list(G_copy.nodes()))
        print("Remaining edges:", [(u, v) for u, v in G_copy.edges()])
        
        # Try to find a path between source and target
        try:
            path = nx.shortest_path(G_copy, source=source, target=target)
            print(f"Found path between terminals: {path}")
            
            # Calculate resistance along the path
            total_resistance = 0
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                if isinstance(G_copy, nx.MultiGraph):
                    edge_key = list(G_copy[u][v].keys())[0]
                    total_resistance += G_copy[u][v][edge_key]["resistance"]
                else:
                    total_resistance += G_copy[u][v]["resistance"]
            
            print(f"Path resistance: {total_resistance:.2f} Ω")
            resistance = total_resistance
        except nx.NetworkXNoPath:
            print("No path exists between terminals.")
    
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
    
    # Simple parallel circuit
    G_parallel = create_simple_parallel_circuit()
    r_parallel = analyze_circuit(G_parallel, "Simple Parallel Circuit")
    if r_parallel is not None:
        results["Parallel"] = r_parallel
    
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
    
    # Ladder circuit
    G_ladder = create_ladder_circuit(3)
    r_ladder = analyze_circuit(G_ladder, "Ladder Circuit (3 rungs)")
    if r_ladder is not None:
        results["Ladder (3 rungs)"] = r_ladder
    
    # Create comparison chart
    if results:
        create_comparison_chart(results)

def main():
    """Main function to run the advanced circuit solver"""
    print("=== Advanced Circuit Solver ===")
    print("Implementing a comprehensive algorithm for resistor network simplification")
    
    run_all_examples()

if __name__ == "__main__":
    main()
