#!/usr/bin/env python3
"""
Generate figures for the Equivalent Resistance Using Graph Theory solution.
This script creates illustrations of:
1. Series reduction
2. Parallel reduction
3. Y-Delta transformation
4. Example circuits (simple, Wheatstone bridge, complex network)
5. Circuit reduction process
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch, Circle
import matplotlib.colors as mcolors
import os
from matplotlib.lines import Line2D

# Create the figures directory if it doesn't exist
os.makedirs('../figures', exist_ok=True)

# Set up consistent styling
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# Color scheme
NODE_COLOR = '#1f77b4'  # Blue
HIGHLIGHT_COLOR = '#ff7f0e'  # Orange
EDGE_COLOR = '#2ca02c'  # Green
HIGHLIGHT_EDGE = '#d62728'  # Red
BACKGROUND_COLOR = '#f8f9fa'  # Light gray
TEXT_COLOR = '#333333'  # Dark gray

def draw_resistor_network(G, pos=None, node_labels=None, edge_labels=None, highlight_nodes=None, highlight_edges=None, 
                          title="Resistor Network", filename=None, ax=None, show_edge_labels=True):
    """
    Draw a resistor network with proper styling
    
    Parameters:
    -----------
    G : NetworkX graph
        The graph to draw
    pos : dict, optional
        Node positions
    node_labels : dict, optional
        Labels for nodes
    edge_labels : dict, optional
        Labels for edges (resistance values)
    highlight_nodes : list, optional
        Nodes to highlight
    highlight_edges : list, optional
        Edges to highlight
    title : str, optional
        Figure title
    filename : str, optional
        If provided, save figure to this path
    ax : matplotlib axis, optional
        Axis to draw on
    show_edge_labels : bool, optional
        Whether to display edge labels
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Calculate positions if not provided
    if pos is None:
        pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    node_colors = [HIGHLIGHT_COLOR if n in (highlight_nodes or []) else NODE_COLOR for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    
    # Draw edges
    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if highlight_edges is not None and ((u, v) in highlight_edges or (v, u) in highlight_edges):
            edge_colors.append(HIGHLIGHT_EDGE)
            edge_widths.append(3.0)
        else:
            edge_colors.append(EDGE_COLOR)
            edge_widths.append(2.0)
    
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax)
    
    # Draw node labels
    if node_labels is None:
        node_labels = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='white', font_weight='bold', ax=ax)
    
    # Draw edge labels (resistance values)
    if show_edge_labels:
        if edge_labels is None:
            edge_labels = {(u, v): f"{G[u][v]['resistance']} Ω" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color=TEXT_COLOR, ax=ax)
    
    # Customize layout
    ax.set_title(title, fontsize=14, fontweight='bold', color=TEXT_COLOR)
    ax.set_axis_off()
    
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")
    
    return ax, pos

def generate_series_reduction_figure():
    """Generate an illustration of series reduction"""
    # Before reduction
    G_before = nx.Graph()
    G_before.add_edge(0, 1, resistance=10)
    G_before.add_edge(1, 2, resistance=20)
    
    # After reduction
    G_after = nx.Graph()
    G_after.add_edge(0, 2, resistance=30)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Consistent node positions
    pos_before = {0: (-1, 0), 1: (0, 0), 2: (1, 0)}
    pos_after = {0: (-1, 0), 2: (1, 0)}
    
    # Draw before reduction
    draw_resistor_network(G_before, pos=pos_before, 
                         highlight_nodes=[1], 
                         highlight_edges=[(0, 1), (1, 2)],
                         title="Before Series Reduction", 
                         ax=ax1)
    
    # Draw after reduction
    draw_resistor_network(G_after, pos=pos_after, 
                         highlight_edges=[(0, 2)],
                         title="After Series Reduction", 
                         ax=ax2)
    
    # Add arrow between diagrams
    fig.text(0.5, 0.5, "→", fontsize=30, ha='center', va='center', color=TEXT_COLOR)
    
    # Add explanation
    fig.text(0.5, 0.05, 
             "Series Reduction: $R_{eq} = R_1 + R_2 = 10 Ω + 20 Ω = 30 Ω$",
             fontsize=12, ha='center', color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('../figures/series_reduction.png', dpi=300, bbox_inches='tight')
    print("Generated series reduction figure")

def generate_parallel_reduction_figure():
    """Generate an illustration of parallel reduction"""
    # Before reduction
    G_before = nx.Graph()
    G_before.add_edge(0, 1, resistance=20)
    G_before.add_edge(0, 1, resistance=30)  # Multiple edges not supported in NetworkX, this is a workaround
    
    # After reduction
    G_after = nx.Graph()
    G_after.add_edge(0, 1, resistance=12)  # 1/(1/20 + 1/30) = 12
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Consistent node positions
    pos = {0: (-0.5, 0), 1: (0.5, 0)}
    
    # For 'before' figure, draw edges manually because NetworkX doesn't support parallel edges
    ax1.set_facecolor(BACKGROUND_COLOR)
    
    # Draw nodes
    nx.draw_networkx_nodes(G_before, pos, node_color=NODE_COLOR, node_size=700, ax=ax1)
    
    # Draw curved edges for parallel resistors
    ax1.plot([pos[0][0], pos[1][0]], [pos[0][1]+0.1, pos[1][1]+0.1], 
             color=HIGHLIGHT_EDGE, linewidth=3.0, alpha=0.8, label='20 Ω')
    ax1.plot([pos[0][0], pos[1][0]], [pos[0][1]-0.1, pos[1][1]-0.1], 
             color=HIGHLIGHT_EDGE, linewidth=3.0, alpha=0.8, label='30 Ω')
    
    # Add resistance labels
    ax1.text(0, 0.15, '20 Ω', color=TEXT_COLOR, ha='center', fontsize=10)
    ax1.text(0, -0.15, '30 Ω', color=TEXT_COLOR, ha='center', fontsize=10)
    
    # Draw node labels
    nx.draw_networkx_labels(G_before, pos, labels={0:'0', 1:'1'}, font_color='white', font_weight='bold', ax=ax1)
    
    ax1.set_title("Before Parallel Reduction", fontsize=14, fontweight='bold', color=TEXT_COLOR)
    ax1.set_axis_off()
    
    # Draw 'after' figure
    draw_resistor_network(G_after, pos=pos, highlight_edges=[(0, 1)],
                         title="After Parallel Reduction", ax=ax2)
    
    # Add arrow between diagrams
    fig.text(0.5, 0.5, "→", fontsize=30, ha='center', va='center', color=TEXT_COLOR)
    
    # Add explanation
    fig.text(0.5, 0.05, 
             "Parallel Reduction: $\\frac{1}{R_{eq}} = \\frac{1}{R_1} + \\frac{1}{R_2} = \\frac{1}{20 Ω} + \\frac{1}{30 Ω} \\approx \\frac{1}{12 Ω}$",
             fontsize=12, ha='center', color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('../figures/parallel_reduction.png', dpi=300, bbox_inches='tight')
    print("Generated parallel reduction figure")

def generate_y_delta_transformation_figure():
    """Generate an illustration of Y-Delta transformation"""
    # Y configuration
    G_y = nx.Graph()
    G_y.add_edge(0, 3, resistance=10)
    G_y.add_edge(1, 3, resistance=20)
    G_y.add_edge(2, 3, resistance=30)
    
    # Delta configuration
    G_delta = nx.Graph()
    G_delta.add_edge(0, 1, resistance=35)
    G_delta.add_edge(1, 2, resistance=25)
    G_delta.add_edge(2, 0, resistance=15)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Node positions for Y
    pos_y = {0: (0, 1), 1: (1, 0), 2: (-1, 0), 3: (0, 0)}
    
    # Node positions for Delta
    pos_delta = {0: (0, 1), 1: (1, 0), 2: (-1, 0)}
    
    # Draw Y configuration
    draw_resistor_network(G_y, pos=pos_y, 
                         highlight_nodes=[3],
                         highlight_edges=[(0, 3), (1, 3), (2, 3)],
                         title="Y (Star) Configuration", 
                         ax=ax1)
    
    # Draw Delta configuration
    draw_resistor_network(G_delta, pos=pos_delta, 
                         highlight_edges=[(0, 1), (1, 2), (2, 0)],
                         title="Δ (Delta) Configuration", 
                         ax=ax2)
    
    # Add bidirectional arrow between diagrams
    fig.text(0.5, 0.5, "⟷", fontsize=30, ha='center', va='center', color=TEXT_COLOR)
    
    # Add transformation equations
    fig.text(0.5, 0.1, 
             "Y to Δ: $R_{12} = \\frac{R_1 R_2 + R_2 R_3 + R_3 R_1}{R_3}$",
             fontsize=10, ha='center', color=TEXT_COLOR)
    fig.text(0.5, 0.05, 
             "Δ to Y: $R_1 = \\frac{R_{12} R_{13}}{R_{12} + R_{23} + R_{13}}$",
             fontsize=10, ha='center', color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('../figures/y_delta_transformation.png', dpi=300, bbox_inches='tight')
    print("Generated Y-Delta transformation figure")

def generate_simple_circuit_figure():
    """Generate a simple series-parallel circuit figure"""
    G = nx.Graph()
    G.add_edge(0, 1, resistance=10)
    G.add_edge(1, 2, resistance=20)
    G.add_edge(0, 3, resistance=30)
    G.add_edge(3, 2, resistance=40)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Custom positions for clearer layout
    pos = {0: (0, 0), 1: (1, 1), 2: (2, 0), 3: (1, -1)}
    
    # Draw the circuit
    draw_resistor_network(G, pos=pos, 
                         node_labels={0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                         title="Simple Series-Parallel Circuit", 
                         ax=ax)
    
    # Add equivalent resistance annotation
    ax.text(1, -2, 
            "Equivalent resistance: $R_{eq} = 26.67$ Ω\n" +
            "Path 1: $10$ Ω + $20$ Ω = $30$ Ω\n" +
            "Path 2: $30$ Ω + $40$ Ω = $70$ Ω\n" +
            "Combined: $\\frac{1}{R_{eq}} = \\frac{1}{30} + \\frac{1}{70}$",
            fontsize=12, ha='center', color=TEXT_COLOR, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/simple_circuit.png', dpi=300, bbox_inches='tight')
    print("Generated simple circuit figure")

def generate_wheatstone_bridge_figure():
    """Generate a Wheatstone bridge circuit figure"""
    G = nx.Graph()
    G.add_edge(0, 1, resistance=10)
    G.add_edge(0, 2, resistance=20)
    G.add_edge(1, 3, resistance=30)
    G.add_edge(2, 3, resistance=40)
    G.add_edge(1, 2, resistance=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Custom positions for clearer layout
    pos = {0: (0, 0), 1: (1, 1), 2: (1, -1), 3: (2, 0)}
    
    # Draw the circuit
    draw_resistor_network(G, pos=pos, 
                         node_labels={0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                         title="Wheatstone Bridge Circuit", 
                         ax=ax)
    
    # Add equivalent resistance annotation
    ax.text(1, -2, 
            "Equivalent resistance between A and D: $R_{eq} = 22.86$ Ω\n" +
            "This circuit requires Y-Δ transformation\n" +
            "since it contains a bridge configuration.",
            fontsize=12, ha='center', color=TEXT_COLOR, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/wheatstone_bridge.png', dpi=300, bbox_inches='tight')
    print("Generated Wheatstone bridge figure")

def generate_complex_network_figure():
    """Generate a complex resistor network figure"""
    G = nx.Graph()
    G.add_edge(0, 1, resistance=10)
    G.add_edge(1, 2, resistance=20)
    G.add_edge(2, 3, resistance=30)
    G.add_edge(3, 0, resistance=40)
    G.add_edge(0, 2, resistance=50)
    G.add_edge(1, 3, resistance=60)
    G.add_edge(4, 0, resistance=70)
    G.add_edge(4, 1, resistance=80)
    G.add_edge(4, 2, resistance=90)
    G.add_edge(4, 3, resistance=100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Custom positions for clearer layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the circuit
    draw_resistor_network(G, pos=pos, 
                         highlight_nodes=[0, 3],
                         title="Complex Resistor Network", 
                         ax=ax)
    
    # Add equivalent resistance annotation
    ax.text(0, -1, 
            "Equivalent resistance between nodes 0 and 3: $R_{eq} = 18.97$ Ω\n" +
            "This complex network would be difficult to solve with traditional methods.",
            fontsize=12, ha='center', color=TEXT_COLOR, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../figures/complex_network.png', dpi=300, bbox_inches='tight')
    print("Generated complex network figure")

def generate_reduction_process_figure():
    """Generate a figure showing the reduction process for a circuit"""
    # Initial circuit
    G_initial = nx.Graph()
    G_initial.add_edge(0, 1, resistance=10)
    G_initial.add_edge(1, 2, resistance=20)
    G_initial.add_edge(0, 3, resistance=30)
    G_initial.add_edge(3, 2, resistance=40)
    
    # After first reduction (series)
    G_step1 = nx.Graph()
    G_step1.add_edge(0, 2, resistance=30)  # 10 + 20
    G_step1.add_edge(0, 3, resistance=30)
    G_step1.add_edge(3, 2, resistance=40)
    
    # After second reduction (parallel)
    G_step2 = nx.Graph()
    G_step2.add_edge(0, 2, resistance=70/3)  # 1/(1/30 + 1/70) ≈ 21.0
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Consistent node positions
    pos_initial = {0: (0, 0), 1: (1, 1), 2: (2, 0), 3: (1, -1)}
    pos_step1 = {0: (0, 0), 2: (2, 0), 3: (1, -1)}
    pos_step2 = {0: (0, 0), 2: (2, 0)}
    
    # Draw initial circuit
    draw_resistor_network(G_initial, pos=pos_initial, 
                         highlight_nodes=[1],
                         title="Initial Circuit", 
                         ax=axes[0])
    
    # Draw after first reduction
    draw_resistor_network(G_step1, pos=pos_step1, 
                         highlight_edges=[(0, 2)],
                         title="After Series Reduction", 
                         ax=axes[1])
    
    # Draw after second reduction
    draw_resistor_network(G_step2, pos=pos_step2, 
                         title="After Parallel Reduction", 
                         ax=axes[2])
    
    # Add arrows between diagrams
    fig.text(0.33, 0.5, "→", fontsize=30, ha='center', va='center', color=TEXT_COLOR)
    fig.text(0.67, 0.5, "→", fontsize=30, ha='center', va='center', color=TEXT_COLOR)
    
    # Add explanation
    fig.text(0.5, 0.05, 
             "Complete Reduction Process: Series → Parallel → Final Equivalent Resistance",
             fontsize=12, ha='center', color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('../figures/reduction_process.png', dpi=300, bbox_inches='tight')
    print("Generated reduction process figure")

def generate_algorithm_flowchart():
    """Generate a flowchart of the reduction algorithm"""
    # Create a directed graph for the flowchart
    G = nx.DiGraph()
    
    # Add nodes for each step
    nodes = [
        "Start",
        "Input Circuit Graph",
        "Try Series\nReduction",
        "Try Parallel\nReduction",
        "Try Y-Δ\nTransformation",
        "No Reduction\nPossible?",
        "Output\nEquivalent\nResistance",
        "End"
    ]
    
    # Add connections between steps
    edges = [
        ("Start", "Input Circuit Graph"),
        ("Input Circuit Graph", "Try Series\nReduction"),
        ("Try Series\nReduction", "Try Parallel\nReduction"),
        ("Try Parallel\nReduction", "Try Y-Δ\nTransformation"),
        ("Try Y-Δ\nTransformation", "No Reduction\nPossible?"),
        ("No Reduction\nPossible?", "Output\nEquivalent\nResistance"),
        ("Output\nEquivalent\nResistance", "End"),
        # Feedback loops
        ("Try Series\nReduction", "Input Circuit Graph"),
        ("Try Parallel\nReduction", "Input Circuit Graph"),
        ("Try Y-Δ\nTransformation", "Input Circuit Graph"),
    ]
    
    # Add nodes and edges to the graph
    for node in nodes:
        G.add_node(node)
    
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    ax.set_facecolor(BACKGROUND_COLOR)
    
    # Set node positions for a top-down flowchart
    pos = {
        "Start": (0, 8),
        "Input Circuit Graph": (0, 7),
        "Try Series\nReduction": (0, 6),
        "Try Parallel\nReduction": (0, 4),
        "Try Y-Δ\nTransformation": (0, 2),
        "No Reduction\nPossible?": (0, 0),
        "Output\nEquivalent\nResistance": (0, -2),
        "End": (0, -4)
    }
    
    # Node colors based on function
    node_colors = {
        "Start": "#9CCC65",  # Light green
        "End": "#EF5350",    # Light red
        "Input Circuit Graph": "#42A5F5",  # Light blue
        "Output\nEquivalent\nResistance": "#42A5F5",  # Light blue
        "Try Series\nReduction": "#FF9800",  # Orange
        "Try Parallel\nReduction": "#FF9800",  # Orange
        "Try Y-Δ\nTransformation": "#FF9800",  # Orange
        "No Reduction\nPossible?": "#AB47BC",  # Purple
    }
    
    # Node shapes
    node_shapes = {
        "Start": "o",
        "End": "o",
        "No Reduction\nPossible?": "d",  # Diamond for decision
        "Input Circuit Graph": "s",  # Square for input/output
        "Output\nEquivalent\nResistance": "s",  # Square for input/output
        "Try Series\nReduction": "s",  # Square for process
        "Try Parallel\nReduction": "s",  # Square for process
        "Try Y-Δ\nTransformation": "s",  # Square for process
    }
    
    # Draw nodes with different shapes and colors
    for node_type, shape in node_shapes.items():
        nodes_of_type = [n for n in G.nodes() if n == node_type]
        if shape == "o":  # Circle
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, 
                                 node_color=[node_colors[n] for n in nodes_of_type],
                                 node_size=2000, node_shape=shape, ax=ax)
        elif shape == "d":  # Diamond
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, 
                                 node_color=[node_colors[n] for n in nodes_of_type],
                                 node_size=2000, node_shape="d", ax=ax)
        else:  # Rectangle/Square
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, 
                                 node_color=[node_colors[n] for n in nodes_of_type],
                                 node_size=2000, node_shape="s", ax=ax)
    
    # Draw all edges
    for u, v in G.edges():
        # Special handling for feedback loops
        if v == "Input Circuit Graph" and u != "Start":
            # Draw curved edge
            ax.annotate("", xy=pos[v], xytext=pos[u],
                      arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, 
                                      connectionstyle="arc3,rad=0.3", lw=1.5))
        else:
            # Draw regular edge
            ax.annotate("", xy=pos[v], xytext=pos[u],
                      arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=1.5))
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')
    
    # Add conditions to the decision diamond
    ax.text(1.5, 0, "Yes", fontsize=10, ha='center', color=TEXT_COLOR)
    ax.text(-1.5, 0, "No", fontsize=10, ha='center', color=TEXT_COLOR)
    
    # Add feedback labels
    ax.text(1.5, 6, "Success", fontsize=10, ha='center', color=TEXT_COLOR)
    ax.text(1.5, 4, "Success", fontsize=10, ha='center', color=TEXT_COLOR)
    ax.text(1.5, 2, "Success", fontsize=10, ha='center', color=TEXT_COLOR)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#9CCC65", markersize=15, label='Start/End'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#42A5F5", markersize=15, label='Input/Output'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#FF9800", markersize=15, label='Process'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor="#AB47BC", markersize=15, label='Decision')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Title
    ax.set_title("Graph Reduction Algorithm Flowchart", fontsize=16, fontweight='bold', color=TEXT_COLOR)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.savefig('../figures/algorithm_flowchart.png', dpi=300, bbox_inches='tight')
    print("Generated algorithm flowchart")

def generate_all_figures():
    """Generate all figures needed for the solution"""
    print("Generating all figures for Equivalent Resistance Using Graph Theory...")
    
    # Basic reduction illustrations
    generate_series_reduction_figure()
    generate_parallel_reduction_figure()
    generate_y_delta_transformation_figure()
    
    # Example circuits
    generate_simple_circuit_figure()
    generate_wheatstone_bridge_figure()
    generate_complex_network_figure()
    
    # Process illustrations
    generate_reduction_process_figure()
    generate_algorithm_flowchart()
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    generate_all_figures()
