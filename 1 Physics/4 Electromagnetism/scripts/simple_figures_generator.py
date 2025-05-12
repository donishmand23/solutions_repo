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

def draw_resistor(ax, x1, y1, x2, y2, label=None, vertical=False):
    """Draw a resistor symbol between (x1,y1) and (x2,y2)"""
    if vertical:
        # Vertical resistor
        ax.plot([x1, x1], [y1, y1 + 0.3*(y2-y1)], 'k-', lw=2)
        ax.plot([x1, x1 + 0.1], [y1 + 0.3*(y2-y1), y1 + 0.35*(y2-y1)], 'k-', lw=2)
        ax.plot([x1 + 0.1, x1 - 0.1], [y1 + 0.35*(y2-y1), y1 + 0.45*(y2-y1)], 'k-', lw=2)
        ax.plot([x1 - 0.1, x1 + 0.1], [y1 + 0.45*(y2-y1), y1 + 0.55*(y2-y1)], 'k-', lw=2)
        ax.plot([x1 + 0.1, x1 - 0.1], [y1 + 0.55*(y2-y1), y1 + 0.65*(y2-y1)], 'k-', lw=2)
        ax.plot([x1 - 0.1, x1], [y1 + 0.65*(y2-y1), y1 + 0.7*(y2-y1)], 'k-', lw=2)
        ax.plot([x1, x1], [y1 + 0.7*(y2-y1), y2], 'k-', lw=2)
        
        if label:
            ax.text(x1 + 0.2, y1 + 0.5*(y2-y1), label, ha='left', va='center')
    else:
        # Horizontal resistor
        ax.plot([x1, x1 + 0.3*(x2-x1)], [y1, y1], 'k-', lw=2)
        ax.plot([x1 + 0.3*(x2-x1), x1 + 0.35*(x2-x1)], [y1, y1 + 0.1], 'k-', lw=2)
        ax.plot([x1 + 0.35*(x2-x1), x1 + 0.45*(x2-x1)], [y1 + 0.1, y1 - 0.1], 'k-', lw=2)
        ax.plot([x1 + 0.45*(x2-x1), x1 + 0.55*(x2-x1)], [y1 - 0.1, y1 + 0.1], 'k-', lw=2)
        ax.plot([x1 + 0.55*(x2-x1), x1 + 0.65*(x2-x1)], [y1 + 0.1, y1 - 0.1], 'k-', lw=2)
        ax.plot([x1 + 0.65*(x2-x1), x1 + 0.7*(x2-x1)], [y1 - 0.1, y1], 'k-', lw=2)
        ax.plot([x1 + 0.7*(x2-x1), x2], [y1, y1], 'k-', lw=2)
        
        if label:
            ax.text(x1 + 0.5*(x2-x1), y1 + 0.2, label, ha='center', va='bottom')

def create_graph_representation_figure():
    """Create a figure showing the graph representation of a circuit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Draw circuit diagram in first subplot
    # Draw wires
    ax1.plot([0, 1, 1, 2, 2, 3], [1, 1, 0, 0, 1, 1], 'k-', lw=2)
    ax1.plot([0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 0, 0], 'k-', lw=2)
    
    # Draw resistors
    draw_resistor(ax1, 0.2, 1, 0.8, 1, 'R₁')
    draw_resistor(ax1, 0.2, 0, 0.8, 0, 'R₂')
    draw_resistor(ax1, 1.2, 0, 1.8, 0, 'R₃')
    draw_resistor(ax1, 1.2, 1, 1.8, 1, 'R₄')
    draw_resistor(ax1, 2.2, 1, 2.8, 1, 'R₅')
    draw_resistor(ax1, 2.2, 0, 2.8, 0, 'R₆')
    
    # Add node labels
    ax1.text(-0.1, 0.5, 'A', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(1, 0.5, 'B', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(2, 0.5, 'C', fontsize=12, fontweight='bold', ha='center', va='center')
    ax1.text(3.1, 0.5, 'D', fontsize=12, fontweight='bold', ha='center', va='center')
    
    ax1.set_xlim(-0.2, 3.2)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Circuit Diagram')
    ax1.axis('off')
    
    # Draw graph representation in second subplot
    # Draw nodes
    node_positions = {'A': (0, 0), 'B': (2, 0), 'C': (4, 0), 'D': (6, 0)}
    
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax2.add_patch(circle)
        ax2.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges with labels
    # A to B, top edge (R₁)
    ax2.plot([0.3, 1.7], [0.1, 0.1], 'k-', lw=1.5)
    ax2.text(1, 0.3, 'R₁ = 5Ω', ha='center', va='bottom')
    
    # A to B, bottom edge (R₂)
    ax2.plot([0.3, 1.7], [-0.1, -0.1], 'k-', lw=1.5)
    ax2.text(1, -0.3, 'R₂ = 10Ω', ha='center', va='top')
    
    # B to C, top edge (R₄)
    ax2.plot([2.3, 3.7], [0.1, 0.1], 'k-', lw=1.5)
    ax2.text(3, 0.3, 'R₄ = 20Ω', ha='center', va='bottom')
    
    # B to C, bottom edge (R₃)
    ax2.plot([2.3, 3.7], [-0.1, -0.1], 'k-', lw=1.5)
    ax2.text(3, -0.3, 'R₃ = 15Ω', ha='center', va='top')
    
    # C to D, top edge (R₅)
    ax2.plot([4.3, 5.7], [0.1, 0.1], 'k-', lw=1.5)
    ax2.text(5, 0.3, 'R₅ = 25Ω', ha='center', va='bottom')
    
    # C to D, bottom edge (R₆)
    ax2.plot([4.3, 5.7], [-0.1, -0.1], 'k-', lw=1.5)
    ax2.text(5, -0.3, 'R₆ = 30Ω', ha='center', va='top')
    
    ax2.set_xlim(-1, 7)
    ax2.set_ylim(-1, 1)
    ax2.set_title('Graph Representation')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'graph_representation.png'), dpi=300)
    plt.close()

def create_series_parallel_reduction_figure():
    """Create a figure showing series and parallel reduction operations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Series reduction in first subplot
    # Draw original circuit with two resistors in series
    node_positions = {'A': (0, 0), 'B': (2, 0), 'C': (4, 0)}
    
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges with labels
    ax1.plot([0.3, 1.7], [0, 0], 'k-', lw=1.5)
    ax1.text(1, 0.3, 'R₁ = 10Ω', ha='center', va='bottom')
    
    ax1.plot([2.3, 3.7], [0, 0], 'k-', lw=1.5)
    ax1.text(3, 0.3, 'R₂ = 20Ω', ha='center', va='bottom')
    
    # Draw arrow indicating reduction
    ax1.arrow(2, -1, 0, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax1.text(2, -1.5, 'Series Reduction', ha='center', va='center')
    
    # Draw reduced circuit
    node_positions_reduced = {'A': (0, -3), 'C': (4, -3)}
    
    for node, pos in node_positions_reduced.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edge with label
    ax1.plot([0.3, 3.7], [-3, -3], 'k-', lw=1.5)
    ax1.text(2, -2.7, 'R₁ + R₂ = 30Ω', ha='center', va='bottom')
    
    # Add formula
    ax1.text(2, -4, 'Series: $R_{eq} = R_1 + R_2 + ... + R_n$', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    ax1.set_xlim(-1, 5)
    ax1.set_ylim(-4.5, 1)
    ax1.set_title('Series Reduction')
    ax1.axis('off')
    
    # Parallel reduction in second subplot
    # Draw original circuit with two resistors in parallel
    node_positions = {'A': (0, 0), 'B': (4, 0)}
    
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax2.add_patch(circle)
        ax2.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges with labels (curved to show parallel)
    ax2.plot([0.3, 3.7], [0.1, 0.1], 'k-', lw=1.5)
    ax2.text(2, 0.4, 'R₁ = 6Ω', ha='center', va='bottom')
    
    ax2.plot([0.3, 3.7], [-0.1, -0.1], 'k-', lw=1.5)
    ax2.text(2, -0.4, 'R₂ = 12Ω', ha='center', va='top')
    
    # Draw arrow indicating reduction
    ax2.arrow(2, -1, 0, -1, head_width=0.2, head_length=0.2, fc='black', ec='black')
    ax2.text(2, -1.5, 'Parallel Reduction', ha='center', va='center')
    
    # Draw reduced circuit
    node_positions_reduced = {'A': (0, -3), 'B': (4, -3)}
    
    for node, pos in node_positions_reduced.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax2.add_patch(circle)
        ax2.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edge with label
    ax2.plot([0.3, 3.7], [-3, -3], 'k-', lw=1.5)
    ax2.text(2, -2.7, 'R₁ || R₂ = 4Ω', ha='center', va='bottom')
    
    # Add formula
    ax2.text(2, -4, 'Parallel: $\\frac{1}{R_{eq}} = \\frac{1}{R_1} + \\frac{1}{R_2} + ... + \\frac{1}{R_n}$', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_xlim(-1, 5)
    ax2.set_ylim(-4.5, 1)
    ax2.set_title('Parallel Reduction')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'series_parallel_reduction.png'), dpi=300)
    plt.close()

def create_graph_reduction_steps_figure():
    """Create a figure showing the steps of graph reduction for a complex circuit."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    
    # Step 1: Original circuit (pentagon with 5 nodes)
    node_positions = {
        'A': (2, 4),
        'B': (0, 2),
        'C': (1, 0),
        'D': (3, 0),
        'E': (4, 2)
    }
    
    # Draw nodes
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        axs[0].add_patch(circle)
        axs[0].text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges with labels
    edges = [
        ('A', 'B', 'R₁ = 10Ω'),
        ('B', 'C', 'R₂ = 20Ω'),
        ('C', 'D', 'R₃ = 30Ω'),
        ('D', 'E', 'R₄ = 40Ω'),
        ('E', 'A', 'R₅ = 50Ω'),
        ('A', 'C', 'R₆ = 60Ω'),
        ('B', 'E', 'R₇ = 70Ω')
    ]
    
    for u, v, label in edges:
        pos_u = node_positions[u]
        pos_v = node_positions[v]
        axs[0].plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', lw=1.5)
        # Position label at midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2
        axs[0].text(mid_x, mid_y, label, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    axs[0].set_xlim(-1, 5)
    axs[0].set_ylim(-1, 5)
    axs[0].set_title('Step 1: Original Circuit')
    axs[0].axis('off')
    
    # Step 2: First reduction (remove edge A-C)
    # Same nodes as step 1
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        axs[1].add_patch(circle)
        axs[1].text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges except A-C
    edges_step2 = [edge for edge in edges if not (edge[0] == 'A' and edge[1] == 'C')]
    
    for u, v, label in edges_step2:
        pos_u = node_positions[u]
        pos_v = node_positions[v]
        axs[1].plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', lw=1.5)
        # Position label at midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2
        axs[1].text(mid_x, mid_y, label, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Show removed edge as dashed line
    pos_a = node_positions['A']
    pos_c = node_positions['C']
    axs[1].plot([pos_a[0], pos_c[0]], [pos_a[1], pos_c[1]], 'r--', lw=1.5, alpha=0.5)
    axs[1].text((pos_a[0] + pos_c[0])/2, (pos_a[1] + pos_c[1])/2, 'Removed', 
               color='red', ha='center', va='center')
    
    axs[1].set_xlim(-1, 5)
    axs[1].set_ylim(-1, 5)
    axs[1].set_title('Step 2: Remove Edge A-C')
    axs[1].axis('off')
    
    # Step 3: Combine series resistors
    # Nodes B and D are removed
    node_positions_step3 = {
        'A': (2, 4),
        'C': (1, 0),
        'E': (4, 2)
    }
    
    for node, pos in node_positions_step3.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        axs[2].add_patch(circle)
        axs[2].text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw new edges with combined resistors
    edges_step3 = [
        ('A', 'C', 'R₁ + R₂ = 30Ω'),
        ('C', 'E', 'R₃ + R₄ = 70Ω'),
        ('E', 'A', 'R₅ = 50Ω')
    ]
    
    for u, v, label in edges_step3:
        pos_u = node_positions_step3[u]
        pos_v = node_positions_step3[v]
        axs[2].plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', lw=1.5)
        # Position label at midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2
        axs[2].text(mid_x, mid_y, label, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7))
    
    # Show removed nodes
    axs[2].plot(node_positions['B'][0], node_positions['B'][1], 'rx', markersize=10)
    axs[2].plot(node_positions['D'][0], node_positions['D'][1], 'rx', markersize=10)
    axs[2].text(node_positions['B'][0], node_positions['B'][1] - 0.5, 'Removed', color='red')
    axs[2].text(node_positions['D'][0], node_positions['D'][1] - 0.5, 'Removed', color='red')
    
    axs[2].set_xlim(-1, 5)
    axs[2].set_ylim(-1, 5)
    axs[2].set_title('Step 3: Combine Series Resistors')
    axs[2].axis('off')
    
    # Step 4: Final reduction to equivalent resistance
    # Only nodes A and E remain
    node_positions_step4 = {
        'A': (1, 2),
        'E': (4, 2)
    }
    
    for node, pos in node_positions_step4.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        axs[3].add_patch(circle)
        axs[3].text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw final equivalent resistance
    pos_a = node_positions_step4['A']
    pos_e = node_positions_step4['E']
    axs[3].plot([pos_a[0], pos_e[0]], [pos_a[1], pos_e[1]], 'k-', lw=2)
    axs[3].text((pos_a[0] + pos_e[0])/2, (pos_a[1] + pos_e[1])/2 + 0.3, 
               'Req = 18.6Ω', ha='center', va='center', 
               bbox=dict(facecolor='white', alpha=0.7))
    
    # Add calculation
    axs[3].text(2.5, 1, 'Req = ((30Ω × 70Ω)/(30Ω + 70Ω)) + 50Ω = 18.6Ω', 
               ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8))
    
    axs[3].set_xlim(-1, 5)
    axs[3].set_ylim(-1, 5)
    axs[3].set_title('Step 4: Final Equivalent Resistance')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'graph_reduction_steps.png'), dpi=300)
    plt.close()

def create_complex_circuit_figure():
    """Create a figure showing a complex circuit and algorithm steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Draw a complex circuit in first subplot
    # Hexagon with 6 nodes
    node_positions = {
        'A': (3, 5),
        'B': (1, 4),
        'C': (1, 2),
        'D': (3, 1),
        'E': (5, 2),
        'F': (5, 4)
    }
    
    # Draw nodes
    for node, pos in node_positions.items():
        circle = Circle(pos, 0.3, color='lightblue', ec='black')
        ax1.add_patch(circle)
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontweight='bold')
    
    # Draw edges
    edges = [
        ('A', 'B', 'R₁ = 5Ω'),
        ('B', 'C', 'R₂ = 10Ω'),
        ('C', 'D', 'R₃ = 15Ω'),
        ('D', 'E', 'R₄ = 20Ω'),
        ('E', 'F', 'R₅ = 25Ω'),
        ('F', 'A', 'R₆ = 30Ω'),
        ('A', 'D', 'R₇ = 35Ω'),
        ('B', 'E', 'R₈ = 40Ω'),
        ('C', 'F', 'R₉ = 45Ω')
    ]
    
    for u, v, label in edges:
        pos_u = node_positions[u]
        pos_v = node_positions[v]
        ax1.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 'k-', lw=1.5)
        # Position label at midpoint
        mid_x = (pos_u[0] + pos_v[0]) / 2
        mid_y = (pos_u[1] + pos_v[1]) / 2
        ax1.text(mid_x, mid_y, label, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7))
    
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.set_title('Complex Circuit Graph Representation')
    ax1.axis('off')
    
    # Create a table showing the algorithm steps in second subplot
    table_data = [
        ['Step', 'Action', 'Result'],
        ['1', 'Identify parallel paths A-B-C-D and A-D', 'Calculate equivalent resistance'],
        ['2', 'Identify parallel paths B-C-F and B-E-F', 'Calculate equivalent resistance'],
        ['3', 'Identify series path A-F', 'Combine resistors'],
        ['4', 'Identify parallel paths A-D-E-F and A-F', 'Calculate equivalent resistance'],
        ['5', 'Final reduction', 'Equivalent resistance = 12.7Ω']
    ]
    
    # Create the table
    table = ax2.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Add algorithm description
    ax2.text(0.5, -0.1, 'Graph Theory Algorithm for Circuit Reduction:', 
            transform=ax2.transAxes, ha='center', va='center', fontweight='bold')
    
    ax2.text(0.5, -0.2, '1. Represent circuit as a graph with resistors as weighted edges', 
            transform=ax2.transAxes, ha='center', va='center')
    
    ax2.text(0.5, -0.3, '2. Identify series and parallel patterns', 
            transform=ax2.transAxes, ha='center', va='center')
    
    ax2.text(0.5, -0.4, '3. Apply reduction rules iteratively until only two nodes remain', 
            transform=ax2.transAxes, ha='center', va='center')
    
    ax2.text(0.5, -0.5, '4. The weight of the final edge is the equivalent resistance', 
            transform=ax2.transAxes, ha='center', va='center')
    
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
