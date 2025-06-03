#!/usr/bin/env python3
"""
Generate Circuit Examples: Creates example circuits and visualizes their reduction process

This script uses the CircuitSolver class to generate example circuits and visualize
the step-by-step reduction process. It creates figures showing how different circuit
configurations are simplified to calculate their equivalent resistance.

Usage:
    python generate_circuit_examples.py

Author: Don Ishmand
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from circuit_solver import CircuitSolver

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)

def generate_ladder_circuit(n_rungs=3):
    """
    Generate a ladder circuit with n rungs and solve it.
    
    Parameters:
    -----------
    n_rungs : int
        Number of rungs in the ladder
        
    Returns:
    --------
    resistance : float
        Equivalent resistance of the ladder circuit
    """
    solver = CircuitSolver(verbose=True)
    
    # Create ladder circuit
    edges = []
    
    # Add the vertical sides of the ladder (10 ohm each)
    for i in range(n_rungs + 1):
        if i < n_rungs:
            edges.append((f'A{i}', f'A{i+1}', 10))
            edges.append((f'B{i}', f'B{i+1}', 10))
        
        # Add the rungs (20 ohm each)
        if i < n_rungs + 1:
            edges.append((f'A{i}', f'B{i}', 20))
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A0', f'A{n_rungs}')
    
    print(f"Equivalent resistance of {n_rungs}-rung ladder: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A0', f'A{n_rungs}', f'../figures/ladder_{n_rungs}_reduction.png')
    
    return resistance

def generate_grid_circuit(size=3):
    """
    Generate a grid circuit of size x size and solve it.
    
    Parameters:
    -----------
    size : int
        Size of the grid (size x size)
        
    Returns:
    --------
    resistance : float
        Equivalent resistance of the grid circuit
    """
    solver = CircuitSolver(verbose=True)
    
    # Create grid circuit
    edges = []
    
    # Add horizontal connections (10 ohm each)
    for i in range(size):
        for j in range(size - 1):
            edges.append((f'N{i}_{j}', f'N{i}_{j+1}', 10))
    
    # Add vertical connections (20 ohm each)
    for i in range(size - 1):
        for j in range(size):
            edges.append((f'N{i}_{j}', f'N{i+1}_{j}', 20))
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'N0_0', f'N{size-1}_{size-1}')
    
    print(f"Equivalent resistance of {size}x{size} grid: {resistance:.4f} Ω")
    solver.visualize_reduction_process('N0_0', f'N{size-1}_{size-1}', f'../figures/grid_{size}x{size}_reduction.png')
    
    return resistance

def generate_bridge_circuit():
    """
    Generate a Wheatstone bridge circuit with unequal resistances and solve it.
    
    Returns:
    --------
    resistance : float
        Equivalent resistance of the bridge circuit
    """
    solver = CircuitSolver(verbose=True)
    
    # Create Wheatstone bridge with unequal resistances
    edges = [
        ('A', 'B', 10),  # 10 ohm
        ('B', 'C', 20),  # 20 ohm
        ('A', 'D', 30),  # 30 ohm
        ('D', 'C', 40),  # 40 ohm
        ('B', 'D', 25),  # 25 ohm (bridge)
    ]
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A', 'C')
    
    print(f"Equivalent resistance of unbalanced bridge: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A', 'C', '../figures/unbalanced_bridge_reduction.png')
    
    # Now create a balanced bridge
    edges_balanced = [
        ('A', 'B', 10),  # 10 ohm
        ('B', 'C', 20),  # 20 ohm
        ('A', 'D', 10),  # 10 ohm (same as A-B)
        ('D', 'C', 20),  # 20 ohm (same as B-C)
        ('B', 'D', 15),  # 15 ohm (bridge)
    ]
    
    circuit_balanced = solver.create_circuit(edges_balanced)
    resistance_balanced = solver.solve(circuit_balanced, 'A', 'C')
    
    print(f"Equivalent resistance of balanced bridge: {resistance_balanced:.4f} Ω")
    solver.visualize_reduction_process('A', 'C', '../figures/balanced_bridge_reduction.png')
    
    return resistance, resistance_balanced

def generate_complex_example():
    """
    Generate a complex circuit that requires all reduction techniques.
    
    Returns:
    --------
    resistance : float
        Equivalent resistance of the complex circuit
    """
    solver = CircuitSolver(verbose=True)
    
    # Create a complex circuit that requires series, parallel, and Y-Delta transformations
    edges = [
        # Outer pentagon
        ('A', 'B', 10),
        ('B', 'C', 15),
        ('C', 'D', 20),
        ('D', 'E', 25),
        ('E', 'A', 30),
        
        # Star configuration from center F
        ('F', 'A', 5),
        ('F', 'B', 10),
        ('F', 'C', 15),
        ('F', 'D', 20),
        ('F', 'E', 25),
        
        # Additional connections
        ('A', 'C', 35),
        ('B', 'D', 40),
        ('C', 'E', 45),
    ]
    
    circuit = solver.create_circuit(edges)
    resistance = solver.solve(circuit, 'A', 'D')
    
    print(f"Equivalent resistance of complex circuit: {resistance:.4f} Ω")
    solver.visualize_reduction_process('A', 'D', '../figures/complex_circuit_reduction.png')
    
    return resistance

def generate_all_examples():
    """Generate all example circuits and their visualizations."""
    print("=== Generating Circuit Examples ===")
    
    print("\n1. Ladder Circuit Examples")
    ladder_2 = generate_ladder_circuit(2)
    ladder_3 = generate_ladder_circuit(3)
    
    print("\n2. Grid Circuit Examples")
    grid_2 = generate_grid_circuit(2)
    grid_3 = generate_grid_circuit(3)
    
    print("\n3. Bridge Circuit Examples")
    unbalanced, balanced = generate_bridge_circuit()
    
    print("\n4. Complex Circuit Example")
    complex_resistance = generate_complex_example()
    
    # Create a summary figure
    plt.figure(figsize=(10, 6))
    
    # Plot results
    circuits = ['Ladder (2)', 'Ladder (3)', 'Grid (2x2)', 'Grid (3x3)', 
                'Unbalanced Bridge', 'Balanced Bridge', 'Complex']
    resistances = [ladder_2, ladder_3, grid_2, grid_3, unbalanced, balanced, complex_resistance]
    
    plt.bar(circuits, resistances, color='skyblue')
    plt.ylabel('Equivalent Resistance (Ω)')
    plt.title('Comparison of Circuit Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig('../figures/circuit_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGenerated comparison chart at '../figures/circuit_comparison.png'")

if __name__ == "__main__":
    generate_all_examples()
