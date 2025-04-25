#!/usr/bin/env python3
import os
import subprocess
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

def parse_chamfer_file(filepath):
    """Parse a chamfer_distance.txt file and return epochs and distances."""
    epochs = []
    distances = []
    
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                # Extract epoch and distance using regex
                match = re.search(r'Epoch (\d+): Chamfer distance: ([\d\.]+)', line)
                if match:
                    epoch = int(match.group(1))
                    distance = float(match.group(2))
                    epochs.append(epoch)
                    distances.append(distance)
    
    return np.array(epochs), np.array(distances)

def create_comparison_plot(base_name, base_output_dir, model_configs):
    """Create comparison plots for the different methods."""
    # Create plots directory
    plots_dir = os.path.join(base_output_dir, 'comparison_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Dictionary to store data for each method
    method_data = {}
    
    # Collect chamfer distance data for each method
    for config in model_configs:
        method = config['method']
        model = config['model']
        
        output_dir = f"{base_output_dir}/{base_name}_{method}_{model.split('Cond')[0]}"
        chamfer_file = os.path.join(output_dir, "chamfer_distance.txt")
        
        if os.path.exists(chamfer_file):
            epochs, distances = parse_chamfer_file(chamfer_file)
            if len(epochs) > 0:
                method_data[method] = (epochs, distances)
    
    # If we have data for at least two methods, create the comparison plot
    if len(method_data) >= 2:
        plt.figure(figsize=(12, 6))
        
        # Define colors for consistent representation
        method_colors = {
            'FM': 'blue',
            'Geomdist': 'red'
        }
        
        for method, (epochs, distances) in method_data.items():
            color = method_colors.get(method, 'green')
            plt.plot(epochs, distances, label=f'{method}', color=color, marker='o', 
                     markersize=3, linestyle='-', alpha=0.8)
        
        plt.title(f'Chamfer Distance Comparison for {base_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Chamfer Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Set y-axis limits for better visualization
        plt.ylim(bottom=0)
        
        # Create log scale version for better visualization of improvements
        plt.yscale('log')
        output_file = os.path.join(plots_dir, f'{base_name}_chamfer_comparison_log.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        # Create linear scale version
        plt.yscale('linear')
        output_file = os.path.join(plots_dir, f'{base_name}_chamfer_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.close()
        
        print(f"Comparison plots created for {base_name}")
        
        # Create relative improvement plot if we have FM and Geomdist data
        if 'FM' in method_data and 'Geomdist' in method_data:
            plt.figure(figsize=(12, 6))
            
            fm_epochs, fm_distances = method_data['FM']
            gd_epochs, gd_distances = method_data['Geomdist']
            
            # Find common epochs
            min_epochs = max(fm_epochs[0], gd_epochs[0])
            max_epochs = min(fm_epochs[-1], gd_epochs[-1])
            
            # Filter data to common epoch range
            fm_mask = (fm_epochs >= min_epochs) & (fm_epochs <= max_epochs)
            gd_mask = (gd_epochs >= min_epochs) & (gd_epochs <= max_epochs)
            
            fm_filtered_epochs = fm_epochs[fm_mask]
            fm_filtered_dist = fm_distances[fm_mask]
            
            gd_filtered_epochs = gd_epochs[gd_mask]
            gd_filtered_dist = gd_distances[gd_mask]
            
            # Find common epochs
            common_epochs = sorted(set(fm_filtered_epochs).intersection(set(gd_filtered_epochs)))
            
            if common_epochs:
                relative_diff = []
                epochs_for_diff = []
                
                for epoch in common_epochs:
                    fm_idx = np.where(fm_filtered_epochs == epoch)[0][0]
                    gd_idx = np.where(gd_filtered_epochs == epoch)[0][0]
                    
                    fm_val = fm_filtered_dist[fm_idx]
                    gd_val = gd_filtered_dist[gd_idx]
                    
                    # Calculate relative difference (positive means FM is better)
                    rel_diff = (gd_val - fm_val) / max(fm_val, gd_val) * 100
                    
                    relative_diff.append(rel_diff)
                    epochs_for_diff.append(epoch)
                
                plt.plot(epochs_for_diff, relative_diff, color='purple', marker='o', 
                         markersize=3, linestyle='-')
                
                plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
                plt.title(f'Relative Improvement: FM vs Geomdist for {base_name}')
                plt.xlabel('Epochs')
                plt.ylabel('Relative Difference (%)') 
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Annotate better method
                plt.text(0.02, 0.95, 'FM better ↑', transform=plt.gca().transAxes, 
                         fontsize=10, verticalalignment='top')
                plt.text(0.02, 0.05, 'Geomdist better ↓', transform=plt.gca().transAxes, 
                         fontsize=10, verticalalignment='bottom')
                
                output_file = os.path.join(plots_dir, f'{base_name}_relative_improvement.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Relative improvement plot created for {base_name}")

def main():
    parser = argparse.ArgumentParser(description='Run training across multiple datasets and models')
    parser.add_argument('--base_output_dir', default='output', help='Base directory for outputs')
    parser.add_argument('--epochs', default=20000, type=int, help='Number of epochs')
    parser.add_argument('--num_points', default=10000, type=int, help='Number of points for inference')
    parser.add_argument('--num_steps', default=64, type=int, help='Number of steps')
    parser.add_argument('--mesh_dir', default='../datasets/meshes/SHREC20/off/', help='Directory containing mesh files')
    parser.add_argument('--plot_only', action='store_true', help='Only generate plots without training')
    args = parser.parse_args()

    # Find all mesh files in the specified directory
    mesh_dir = args.mesh_dir
    data_files = []
    
    # Look for common 3D file extensions
    for ext in ['*.obj', '*.off', '*.ply', '*.stl']:
        data_files.extend(glob.glob(os.path.join(mesh_dir, ext)))
    
    if not data_files:
        print(f"No mesh files found in {mesh_dir}")
        return
    
    print(f"Found {len(data_files)} mesh files to process")

    # Model configurations to try
    model_configs = [
        {'method': 'FM', 'model': 'FMCond'},
        {'method': 'Geomdist', 'model': 'EDMPrecond'},
    ]

    # Process each mesh file
    for data_file in data_files:
        # Extract base name without extension for directory naming
        base_name = os.path.splitext(os.path.basename(data_file))[0]
        
        if args.plot_only:
            # Only generate plots without training
            create_comparison_plot(base_name, args.base_output_dir, model_configs)
            continue
            
        # Run training for each model configuration
        for config in model_configs:
            method = config['method']
            model = config['model']
            
            # Create output directory name based on dataset and model config
            output_dir = f"{args.base_output_dir}/{base_name}_{method}_{model.split('Cond')[0]}"
            
            # Create the full command
            cmd = [
                "python", "main.py",
                "--output_dir", output_dir,
                "--log_dir", output_dir,
                "--data_path", data_file,
                "--train", "--inference",
                "--method", method,
                "--model", model,
                "--epochs", str(args.epochs),
                "--num_points_inference", str(args.num_points),
                "--num-steps", str(args.num_steps)
            ]
            
            # Print the command being executed
            print(f"\n\n{'='*80}")
            print(f"Running: {' '.join(cmd)}")
            print(f"{'='*80}\n")
            
            # Execute the command
            subprocess.run(cmd)
        
        # After all methods are trained for this dataset, create comparison plots
        create_comparison_plot(base_name, args.base_output_dir, model_configs)

if __name__ == "__main__":
    main()