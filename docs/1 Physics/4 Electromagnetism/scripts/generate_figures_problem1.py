import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

# Create output directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
figures_dir = os.path.join(parent_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

def create_graph_representation_figure():
    """Create a figure showing the graph representation of a circuit."""
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Draw circuit diagram in first subplot
    circuit_img = plt.imread(os.path.join(script_dir, 'circuit_template.png')) if os.path.exists(os.path.join(script_dir, 'circuit_template.png')) else None
    
    if circuit_img is not None:
        ax1.imshow(circuit_img)
        ax1.axis('off')
    else:
        # Draw a simple circuit if template is not available
        ax1.plot([0, 1, 1, 2, 2, 3], [1, 1, 0, 0, 1, 1], 'k-', linewidth=2)
        ax1.plot([0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 0, 0], 'k-', linewidth=2)
        ax1.plot([1, 1], [0, 1], 'k-', linewidth=2)
        ax1.plot([2, 2], [0, 1], 'k-', linewidth=2)
        
        # Add resistor symbols
        ax1.plot([0.4, 0.6], [1, 1], 'k-', linewidth=4)
        ax1.plot([0.4, 0.6], [0, 0], 'k-', linewidth=4)
        ax1.plot([1.4, 1.6], [0, 0], 'k-', linewidth=4)
        ax1.plot([1.4, 1.6], [1, 1], 'k-', linewidth=4)
        ax1.plot([2.4, 2.6], [1, 1], 'k-', linewidth=4)
        ax1.plot([2.4, 2.6], [0, 0], 'k-', linewidth=4)
        
        # Add resistor labels
        ax1.text(0.5, 1.1, 'R₁', fontsize=12)
        ax1.text(0.5, -0.1, 'R₂', fontsize=12)
        ax1.text(1.5, -0.1, 'R₃', fontsize=12)
        ax1.text(1.5, 1.1, 'R₄', fontsize=12)
        ax1.text(2.5, 1.1, 'R₅', fontsize=12)
        ax1.text(2.5, -0.1, 'R₆', fontsize=12)
        
        # Add node labels
        ax1.text(-0.1, 0.5, 'A', fontsize=12, fontweight='bold')
        ax1.text(1, 0.5, 'B', fontsize=12, fontweight='bold')
        ax1.text(2, 0.5, 'C', fontsize=12, fontweight='bold')
        ax1.text(3.1, 0.5, 'D', fontsize=12, fontweight='bold')
        
        ax1.set_xlim(-0.2, 3.2)
        ax1.set_ylim(-0.2, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
    
    ax1.set_title('Circuit Diagram')
    
    # Draw graph representation in second subplot
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C', 'D'])
    G.add_edge('A', 'B', weight=5, label='R₁')
    G.add_edge('A', 'B', weight=10, label='R₂')
    G.add_edge('B', 'C', weight=15, label='R₃')
    G.add_edge('B', 'C', weight=20, label='R₄')
    G.add_edge('C', 'D', weight=25, label='R₅')
    G.add_edge('C', 'D', weight=30, label='R₆')
    
    # Position nodes in a line
    pos = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0)}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax2)
    
    # Draw edges with different y-positions to avoid overlap
    edge_y_positions = {('A', 'B', 0): 0.1, ('A', 'B', 1): -0.1, 
                       ('B', 'C', 0): 0.1, ('B', 'C', 1): -0.1,
                       ('C', 'D', 0): 0.1, ('C', 'D', 1): -0.1}
    
    for u, v, key in G.edges(keys=True):
        y_pos = edge_y_positions.get((u, v, key), 0)
        # Create curved edge
        ax2.annotate("", 
                    xy=(pos[v][0], pos[v][1]), 
                    xytext=(pos[u][0], pos[u][1]),
                    arrowprops=dict(arrowstyle="-", color="black", 
                                   connectionstyle=f"arc3,rad={y_pos}"))
        
        # Add edge label (resistance value)
        label_pos = ((pos[u][0] + pos[v][0]) / 2, y_pos * 2)
        edge_data = G.get_edge_data(u, v, key)
        ax2.text(label_pos[0], label_pos[1], f"{edge_data['label']}\n{edge_data['weight']}Ω", 
                horizontalalignment='center', verticalalignment='center')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax2)
    
    ax2.set_title('Graph Representation')
    ax2.set_xlim(-0.5, 3.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'graph_representation.png'), dpi=300)
    plt.close()

def create_series_parallel_reduction_figure():
    """Create a figure showing series and parallel reduction operations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Series reduction
    G1 = nx.Graph()
    G1.add_nodes_from(['A', 'B', 'C'])
    G1.add_edge('A', 'B', weight=10)
    G1.add_edge('B', 'C', weight=20)
    
    pos1 = {'A': (0, 0), 'B': (1, 0), 'C': (2, 0)}
    
    # Draw original series circuit
    nx.draw_networkx_nodes(G1, pos1, node_size=500, node_color='lightblue', ax=ax1)
    nx.draw_networkx_edges(G1, pos1, width=2, ax=ax1)
    nx.draw_networkx_labels(G1, pos1, font_weight='bold', ax=ax1)
    
    # Add edge labels
    edge_labels = {('A', 'B'): '10Ω', ('B', 'C'): '20Ω'}
    nx.draw_networkx_edge_labels(G1, pos1, edge_labels=edge_labels, ax=ax1)
    
    # Draw arrow indicating reduction
    ax1.annotate("", xy=(1, -0.5), xytext=(1, -1), 
                arrowprops=dict(arrowstyle="->", lw=2))
    ax1.text(1, -0.75, "Series Reduction", ha='center')
    
    # Draw reduced circuit
    G1_reduced = nx.Graph()
    G1_reduced.add_nodes_from(['A', 'C'])
    G1_reduced.add_edge('A', 'C', weight=30)
    
    pos1_reduced = {'A': (0, -1.5), 'C': (2, -1.5)}
    
    nx.draw_networkx_nodes(G1_reduced, pos1_reduced, node_size=500, node_color='lightblue', ax=ax1)
    nx.draw_networkx_edges(G1_reduced, pos1_reduced, width=2, ax=ax1)
    nx.draw_networkx_labels(G1_reduced, pos1_reduced, font_weight='bold', ax=ax1)
    
    # Add edge label for reduced circuit
    edge_labels_reduced = {('A', 'C'): '30Ω\n(10Ω + 20Ω)'}
    nx.draw_networkx_edge_labels(G1_reduced, pos1_reduced, edge_labels=edge_labels_reduced, ax=ax1)
    
    ax1.set_title('Series Reduction')
    ax1.axis('off')
    
    # Parallel reduction
    G2 = nx.Graph()
    G2.add_nodes_from(['A', 'B'])
    G2.add_edge('A', 'B', weight=6)
    G2.add_edge('A', 'B', weight=12)
    
    pos2 = {'A': (0, 0), 'B': (2, 0)}
    
    # Draw original parallel circuit
    nx.draw_networkx_nodes(G2, pos2, node_size=500, node_color='lightblue', ax=ax2)
    
    # Draw curved edges for parallel resistors
    ax2.annotate("", xy=(pos2['B'][0], pos2['B'][1]), xytext=(pos2['A'][0], pos2['A'][1]),
                arrowprops=dict(arrowstyle="-", color="black", connectionstyle="arc3,rad=0.3"))
    ax2.annotate("", xy=(pos2['B'][0], pos2['B'][1]), xytext=(pos2['A'][0], pos2['A'][1]),
                arrowprops=dict(arrowstyle="-", color="black", connectionstyle="arc3,rad=-0.3"))
    
    # Add edge labels
    ax2.text(1, 0.3, "6Ω", ha='center')
    ax2.text(1, -0.3, "12Ω", ha='center')
    
    nx.draw_networkx_labels(G2, pos2, font_weight='bold', ax=ax2)
    
    # Draw arrow indicating reduction
    ax2.annotate("", xy=(1, -0.5), xytext=(1, -1), 
                arrowprops=dict(arrowstyle="->", lw=2))
    ax2.text(1, -0.75, "Parallel Reduction", ha='center')
    
    # Draw reduced circuit
    G2_reduced = nx.Graph()
    G2_reduced.add_nodes_from(['A', 'B'])
    G2_reduced.add_edge('A', 'B', weight=4)
    
    pos2_reduced = {'A': (0, -1.5), 'B': (2, -1.5)}
    
    nx.draw_networkx_nodes(G2_reduced, pos2_reduced, node_size=500, node_color='lightblue', ax=ax2)
    nx.draw_networkx_edges(G2_reduced, pos2_reduced, width=2, ax=ax2)
    nx.draw_networkx_labels(G2_reduced, pos2_reduced, font_weight='bold', ax=ax2)
    
    # Add edge label for reduced circuit
    edge_labels_reduced = {('A', 'B'): '4Ω\n(6Ω || 12Ω)'}
    nx.draw_networkx_edge_labels(G2_reduced, pos2_reduced, edge_labels=edge_labels_reduced, ax=ax2)
    
    ax2.set_title('Parallel Reduction')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'series_parallel_reduction.png'), dpi=300)
    plt.close()

def create_graph_reduction_steps_figure():
    """Create a figure showing the steps of graph reduction for a complex circuit."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # Original graph
    G_orig = nx.Graph()
    G_orig.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    G_orig.add_edge('A', 'B', weight=10, label='R₁')
    G_orig.add_edge('B', 'C', weight=20, label='R₂')
    G_orig.add_edge('C', 'D', weight=30, label='R₃')
    G_orig.add_edge('D', 'E', weight=40, label='R₄')
    G_orig.add_edge('A', 'C', weight=50, label='R₅')
    G_orig.add_edge('B', 'D', weight=60, label='R₆')
    G_orig.add_edge('C', 'E', weight=70, label='R₇')
    
    # Position nodes in a pentagon
    pos = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 0),
        'D': (1.5, -1),
        'E': (0.5, -1)
    }
    
    # Step 1: Original circuit
    nx.draw_networkx_nodes(G_orig, pos, node_size=500, node_color='lightblue', ax=axs[0])
    nx.draw_networkx_edges(G_orig, pos, width=2, ax=axs[0])
    nx.draw_networkx_labels(G_orig, pos, font_weight='bold', ax=axs[0])
    
    # Add edge labels
    edge_labels = {(u, v): f"{d['label']}\n{d['weight']}Ω" for u, v, d in G_orig.edges(data=True)}
    nx.draw_networkx_edge_labels(G_orig, pos, edge_labels=edge_labels, ax=axs[0])
    
    axs[0].set_title('Step 1: Original Circuit')
    axs[0].axis('off')
    
    # Step 2: First reduction (parallel edges)
    G_step2 = nx.Graph()
    G_step2.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
    G_step2.add_edge('A', 'B', weight=10, label='R₁')
    G_step2.add_edge('B', 'C', weight=20, label='R₂')
    G_step2.add_edge('C', 'D', weight=30, label='R₃')
    G_step2.add_edge('D', 'E', weight=40, label='R₄')
    G_step2.add_edge('A', 'C', weight=50, label='R₅')
    # Removed edge B-D
    G_step2.add_edge('C', 'E', weight=70, label='R₇')
    
    nx.draw_networkx_nodes(G_step2, pos, node_size=500, node_color='lightblue', ax=axs[1])
    nx.draw_networkx_edges(G_step2, pos, width=2, ax=axs[1])
    nx.draw_networkx_labels(G_step2, pos, font_weight='bold', ax=axs[1])
    
    # Add edge labels
    edge_labels = {(u, v): f"{d['label']}\n{d['weight']}Ω" for u, v, d in G_step2.edges(data=True)}
    nx.draw_networkx_edge_labels(G_step2, pos, edge_labels=edge_labels, ax=axs[1])
    
    # Highlight the change
    axs[1].plot([pos['B'][0], pos['D'][0]], [pos['B'][1], pos['D'][1]], 'r--', alpha=0.5)
    axs[1].text(1.25, 0, 'Removed', color='red', fontsize=10)
    
    axs[1].set_title('Step 2: Remove Edge B-D')
    axs[1].axis('off')
    
    # Step 3: Second reduction (series edges)
    G_step3 = nx.Graph()
    G_step3.add_nodes_from(['A', 'C', 'D', 'E'])
    G_step3.add_edge('A', 'C', weight=50, label='R₅')
    G_step3.add_edge('A', 'C', weight=30, label='R₁+R₂')
    G_step3.add_edge('C', 'D', weight=30, label='R₃')
    G_step3.add_edge('D', 'E', weight=40, label='R₄')
    G_step3.add_edge('C', 'E', weight=70, label='R₇')
    
    # Adjust positions for the reduced graph
    pos_step3 = {
        'A': (0, 0),
        'C': (2, 0),
        'D': (1.5, -1),
        'E': (0.5, -1)
    }
    
    nx.draw_networkx_nodes(G_step3, pos_step3, node_size=500, node_color='lightblue', ax=axs[2])
    
    # Draw edges with different y-positions to avoid overlap for parallel edges
    for u, v, d in G_step3.edges(data=True):
        if u == 'A' and v == 'C' and d['label'] == 'R₅':
            ax = axs[2]
            ax.annotate("", xy=(pos_step3[v][0], pos_step3[v][1]), 
                       xytext=(pos_step3[u][0], pos_step3[u][1]),
                       arrowprops=dict(arrowstyle="-", color="black", 
                                      connectionstyle="arc3,rad=0.3"))
            # Add edge label
            mid_x = (pos_step3[u][0] + pos_step3[v][0]) / 2
            mid_y = (pos_step3[u][1] + pos_step3[v][1]) / 2 + 0.3
            ax.text(mid_x, mid_y, f"{d['label']}\n{d['weight']}Ω", 
                   horizontalalignment='center', verticalalignment='center')
        elif u == 'A' and v == 'C' and d['label'] == 'R₁+R₂':
            ax = axs[2]
            ax.annotate("", xy=(pos_step3[v][0], pos_step3[v][1]), 
                       xytext=(pos_step3[u][0], pos_step3[u][1]),
                       arrowprops=dict(arrowstyle="-", color="black", 
                                      connectionstyle="arc3,rad=-0.3"))
            # Add edge label
            mid_x = (pos_step3[u][0] + pos_step3[v][0]) / 2
            mid_y = (pos_step3[u][1] + pos_step3[v][1]) / 2 - 0.3
            ax.text(mid_x, mid_y, f"{d['label']}\n{d['weight']}Ω", 
                   horizontalalignment='center', verticalalignment='center')
        else:
            axs[2].plot([pos_step3[u][0], pos_step3[v][0]], 
                       [pos_step3[u][1], pos_step3[v][1]], 'k-', linewidth=2)
            # Add edge label
            mid_x = (pos_step3[u][0] + pos_step3[v][0]) / 2
            mid_y = (pos_step3[u][1] + pos_step3[v][1]) / 2
            axs[2].text(mid_x, mid_y, f"{d['label']}\n{d['weight']}Ω", 
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(facecolor='white', alpha=0.7))
    
    nx.draw_networkx_labels(G_step3, pos_step3, font_weight='bold', ax=axs[2])
    
    # Highlight the change
    axs[2].plot([pos['A'][0], pos['B'][0], pos['C'][0]], 
               [pos['A'][1], pos['B'][1], pos['C'][1]], 'r--', alpha=0.5)
    axs[2].text(1, 0.5, 'Combined', color='red', fontsize=10)
    
    axs[2].set_title('Step 3: Combine Series Resistors')
    axs[2].axis('off')
    
    # Step 4: Final reduction
    G_step4 = nx.Graph()
    G_step4.add_nodes_from(['A', 'E'])
    G_step4.add_edge('A', 'E', weight=18.6, label='Req')
    
    # Adjust positions for the final graph
    pos_step4 = {
        'A': (0, 0),
        'E': (0.5, -1)
    }
    
    nx.draw_networkx_nodes(G_step4, pos_step4, node_size=500, node_color='lightblue', ax=axs[3])
    nx.draw_networkx_edges(G_step4, pos_step4, width=2, ax=axs[3])
    nx.draw_networkx_labels(G_step4, pos_step4, font_weight='bold', ax=axs[3])
    
    # Add edge label
    edge_labels = {('A', 'E'): f"{G_step4.edges[('A', 'E')]['label']}\n{G_step4.edges[('A', 'E')]['weight']}Ω"}
    nx.draw_networkx_edge_labels(G_step4, pos_step4, edge_labels=edge_labels, ax=axs[3])
    
    axs[3].set_title('Step 4: Final Equivalent Resistance')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'graph_reduction_steps.png'), dpi=300)
    plt.close()

def create_complex_circuit_figure():
    """Create a figure showing a more complex circuit and its graph representation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Create a complex graph
    G = nx.Graph()
    G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])
    G.add_edge('A', 'B', weight=5, label='R₁')
    G.add_edge('B', 'C', weight=10, label='R₂')
    G.add_edge('C', 'D', weight=15, label='R₃')
    G.add_edge('D', 'E', weight=20, label='R₄')
    G.add_edge('E', 'F', weight=25, label='R₅')
    G.add_edge('A', 'F', weight=30, label='R₆')
    G.add_edge('A', 'C', weight=35, label='R₇')
    G.add_edge('A', 'D', weight=40, label='R₈')
    G.add_edge('B', 'E', weight=45, label='R₉')
    G.add_edge('C', 'F', weight=50, label='R₁₀')
    
    # Position nodes in a hexagon
    pos = {
        'A': (0, 0),
        'B': (1, 1),
        'C': (2, 0),
        'D': (2, -1),
        'E': (1, -2),
        'F': (0, -1)
    }
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax1)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, ax=ax1)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_weight='bold', ax=ax1)
    
    # Add edge labels
    edge_labels = {(u, v): f"{d['label']}\n{d['weight']}Ω" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_size=8)
    
    ax1.set_title('Complex Circuit Graph Representation')
    ax1.axis('off')
    
    # Create a table showing the algorithm steps for this circuit
    steps = [
        ["Step", "Action", "Result"],
        ["1", "Identify parallel edges between A and C", "Replace R₇ with equivalent resistance"],
        ["2", "Identify series path B-C-D", "Replace with equivalent resistance"],
        ["3", "Identify parallel paths A-D", "Calculate equivalent resistance"],
        ["4", "Identify parallel paths A-F", "Calculate equivalent resistance"],
        ["5", "Reduce remaining edges", "Final equivalent resistance: 12.7Ω"]
    ]
    
    table = ax2.table(cellText=steps, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add equations for the steps
    equations = [
        "Step 1: No parallel edges to combine initially",
        "Step 2: R_{B-C-D} = R₂ + R₃ = 10Ω + 15Ω = 25Ω",
        "Step 3: R_{A-D} = \\frac{1}{\\frac{1}{40} + \\frac{1}{25}} = 15.4Ω",
        "Step 4: R_{A-F} = \\frac{1}{\\frac{1}{30} + \\frac{1}{50}} = 18.8Ω",
        "Step 5: R_{eq} = \\frac{1}{\\frac{1}{15.4} + \\frac{1}{18.8}} = 8.5Ω"
    ]
    
    for i, eq in enumerate(equations):
        ax2.text(0.5, -0.1 - i*0.1, eq, ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title('Algorithm Steps for Complex Circuit')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'complex_circuit.png'), dpi=300)
    plt.close()

def main():
    print("Generating figures for Equivalent Resistance Using Graph Theory...")
    create_graph_representation_figure()
    create_series_parallel_reduction_figure()
    create_graph_reduction_steps_figure()
    create_complex_circuit_figure()
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
