"""
Expert-Parallelism Load Balancer (EPLB): A tool for balancing expert workloads across GPUs.
This module provides functions to distribute experts across multiple GPUs to optimize workload distribution.
"""
import os
import re
import json
from typing import Tuple, List, Optional

import torch
import pandas as pd
from eplb import rebalance_experts


# --- Helper functions ---

def natural_sort_key(s: str) -> List:
    """Generate a natural sort key for strings.
    
    Args:
        s: Input string to be sorted
        
    Returns:
        A list that can be used for natural sorting
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def load_csv_to_tensor(input_folder: str) -> Optional[torch.Tensor]:
    """Load CSV files from a folder and convert them to a tensor.
    
    Each CSV file represents one layer, and is converted to one row in the output tensor.
    
    Args:
        input_folder: Path to folder containing CSV files
        
    Returns:
        torch.Tensor of shape [num_layers, num_experts] or None if an error occurs
    """
    # Get all csv files and sort them naturally
    csv_files = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".csv")],
        key=natural_sort_key
    )
    num_layers = len(csv_files)

    if num_layers == 0:
        print(f"No CSV files found in {input_folder}")
        return None

    # Initialize a list to store rows
    rows = []

    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file, header=None)
            # Sum up all rows in the CSV file
            row_sums = df.sum(axis=0).values
            # Append to the list as a row
            rows.append(row_sums)
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            return None

    # Stack all rows into a single tensor of shape [num_layers, num_experts]
    return torch.tensor(rows, dtype=torch.float32)


def export_results_to_json(output_folder: str, phy2log: torch.Tensor, log2phy: torch.Tensor, 
                          logcnt: torch.Tensor, config: dict) -> None:
    """Export rebalance results to JSON files.
    
    Args:
        output_folder: Path to output folder
        phy2log: Physical to logical expert mapping
        log2phy: Logical to physical expert mapping
        logcnt: Expert count per logical expert
        config: Configuration parameters
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert tensors to lists for JSON serialization
    phy2log_json = phy2log.tolist()
    log2phy_json = log2phy.tolist()
    logcnt_json = logcnt.tolist()
    
    # Build filename suffix
    suffix = f"{config['num_replicas']}replicas_{config['num_groups']}groups_{config['num_nodes']}nodes_{config['num_gpus']}gpus"
    
    # Write JSON files
    with open(os.path.join(output_folder, f"phy2log_{suffix}.json"), "w") as f:
        json.dump(phy2log_json, f)
    with open(os.path.join(output_folder, f"log2phy_{suffix}.json"), "w") as f:
        json.dump(log2phy_json, f)
    with open(os.path.join(output_folder, f"logcnt_{suffix}.json"), "w") as f:
        json.dump(logcnt_json, f)
    
    print(f"Results exported to {output_folder}")

# --- Main execution function ---

def run_eplb(input_folder: str, config: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the Expert-Parallelism Load Balancer on input data.
    
    Args:
        input_folder: Path to folder containing input CSV files
        config: Dictionary containing configuration parameters:
               - num_replicas: Number of physical experts
               - num_groups: Number of expert groups
               - num_nodes: Number of server nodes
               - num_gpus: Number of GPUs
               
    Returns:
        Tuple of (phy2log, log2phy, logcnt) tensors
    """
    # Load weights from CSV files
    weights = load_csv_to_tensor(input_folder)
    if weights is None:
        raise ValueError(f"Failed to load weights from {input_folder}")
        
    print(f"Loaded {weights.size(0)} layers, each with {weights.size(1)} experts")
    
    # Run the rebalancing algorithm
    phy2log, log2phy, logcnt = rebalance_experts(
        weight=weights,
        num_replicas=config['num_replicas'],
        num_groups=config['num_groups'],
        num_nodes=config['num_nodes'],
        num_gpus=config['num_gpus']
    )
    
    print(f"Rebalancing complete:")
    print(f"  - Physical to logical map shape: {phy2log.shape}")
    print(f"  - Logical to physical map shape: {log2phy.shape}")
    print(f"  - Logical count shape: {logcnt.shape}")
    
    return phy2log, log2phy, logcnt


def main():
    """Main entry point for the application."""
    # Configuration parameters
    input_folder = r"/home/bingxche/log/mixtral8x7b_ep4_mixtral_dataset_15_prompts_vanilla/moe_token_dist"
    config = {
        'num_replicas': 12,  # Number of physical experts
        'num_groups': 1,    # Number of expert groups
        'num_nodes': 1,     # Number of server nodes
        'num_gpus': 4       # Number of GPUs
    }
    
    # Run the EPLB algorithm
    phy2log, log2phy, logcnt = run_eplb(input_folder, config)
    
    # Export results to JSON files
    output_folder = f"{input_folder}_eplb_{config['num_replicas']}replicas_{config['num_groups']}groups_{config['num_nodes']}nodes_{config['num_gpus']}gpus"
    export_results_to_json(output_folder, phy2log, log2phy, logcnt, config)


if __name__ == "__main__":
    main()