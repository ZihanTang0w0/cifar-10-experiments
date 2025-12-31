#!/usr/bin/env python3
"""
Script to verify and list all generated datasets before evaluation
"""

import os
from pathlib import Path
import re

def find_generated_datasets(base_dir="./generated_data"):
    """Find all generated_batch files and parse their metadata"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory not found: {base_dir}")
        return []
    
    datasets = []
    
    # Find all generated_batch files
    for batch_file in base_path.rglob("generated_batch"):
        if batch_file.is_file():
            dir_name = batch_file.parent.name
            
            # Parse metadata from directory name
            # Example: generated_1000x10_mode_diffusion_steps_20_cfg_scale_1.0
            metadata = parse_metadata(dir_name)
            
            datasets.append({
                'path': str(batch_file),
                'dir_name': dir_name,
                'metadata': metadata,
                'size_mb': batch_file.stat().st_size / (1024 * 1024)
            })
    
    return sorted(datasets, key=lambda x: (
        x['metadata']['mode'],
        x['metadata']['cfg_scale'],
        x['metadata']['steps']
    ))

def parse_metadata(dir_name):
    """Parse metadata from directory name"""
    metadata = {
        'mode': None,
        'steps': None,
        'cfg_scale': None,
        'num_samples': None
    }
    
    # Extract mode
    if 'mode_diffusion' in dir_name:
        metadata['mode'] = 'diffusion'
    elif 'mode_flow' in dir_name:
        metadata['mode'] = 'flow'
    
    # Extract steps
    steps_match = re.search(r'steps_(\d+)', dir_name)
    if steps_match:
        metadata['steps'] = int(steps_match.group(1))
    
    # Extract cfg_scale
    cfg_match = re.search(r'cfg_scale_([\d.]+)', dir_name)
    if cfg_match:
        metadata['cfg_scale'] = float(cfg_match.group(1))
    
    # Extract num_samples
    samples_match = re.search(r'(\d+)x10', dir_name)
    if samples_match:
        metadata['num_samples'] = int(samples_match.group(1)) * 10
    
    return metadata

def main():
    print("=" * 70)
    print("  Generated Dataset Verification")
    print("=" * 70)
    print()
    
    datasets = find_generated_datasets()
    
    if not datasets:
        print("No datasets found in ./generated_data")
        return
    
    print(f"Found {len(datasets)} dataset(s):\n")
    
    # Group by mode
    flow_datasets = [d for d in datasets if d['metadata']['mode'] == 'flow']
    diffusion_datasets = [d for d in datasets if d['metadata']['mode'] == 'diffusion']
    
    # Display Flow datasets
    if flow_datasets:
        print("-" * 70)
        print("FLOW Model Datasets:")
        print("-" * 70)
        print(f"{'Steps':<8} {'CFG Scale':<12} {'Samples':<10} {'Size (MB)':<12} {'Log Name'}")
        print("-" * 70)
        
        for d in flow_datasets:
            m = d['metadata']
            log_name = f"eval_log_{d['dir_name'].replace('generated_', '')}.txt"
            print(f"{m['steps']:<8} {m['cfg_scale']:<12.1f} {m['num_samples']:<10} "
                  f"{d['size_mb']:<12.2f} {log_name}")
        
        print()
    
    # Display Diffusion datasets
    if diffusion_datasets:
        print("-" * 70)
        print("DIFFUSION Model Datasets:")
        print("-" * 70)
        print(f"{'Steps':<8} {'CFG Scale':<12} {'Samples':<10} {'Size (MB)':<12} {'Log Name'}")
        print("-" * 70)
        
        for d in diffusion_datasets:
            m = d['metadata']
            log_name = f"eval_log_{d['dir_name'].replace('generated_', '')}.txt"
            print(f"{m['steps']:<8} {m['cfg_scale']:<12.1f} {m['num_samples']:<10} "
                  f"{d['size_mb']:<12.2f} {log_name}")
        
        print()
    
    # Summary statistics
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total datasets: {len(datasets)}")
    print(f"  - Flow: {len(flow_datasets)}")
    print(f"  - Diffusion: {len(diffusion_datasets)}")
    
    if datasets:
        total_size = sum(d['size_mb'] for d in datasets)
        print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
        
        # Check for expected datasets based on your generation parameters
        print()
        print("Expected datasets:")
        print("  - Flow: CFG 1.0-2.8 (stride 0.2) × Steps 20-100 (stride 20)")
        print("    → 10 CFG values × 5 step values = 50 datasets")
        print("  - Diffusion: CFG 1.0-4.0 (stride 0.3) × Steps 20-100 (stride 20)")
        print("    → 11 CFG values × 5 step values = 55 datasets")
        print(f"  - Total expected: 105 datasets")
        print(f"  - Found: {len(datasets)} datasets")
        
        if len(datasets) < 105:
            print(f"  ⚠ Warning: {105 - len(datasets)} dataset(s) missing")
        elif len(datasets) == 105:
            print("  ✓ All expected datasets found!")
    
    print("=" * 70)
    print()
    print("To run evaluation:")
    print("  ./run_full_eval_auto.sh")
    print()
    print("Or with custom reference dataset:")
    print("  REFERENCE_DATASET=/path/to/test_batch ./run_full_eval_auto.sh")

if __name__ == "__main__":
    main()