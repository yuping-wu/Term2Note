import json
import numpy as np
import os
from collections import defaultdict

def aggregate_cv_results(num_folds=5, folder_path=""):
    """
    Aggregate cross-validation results from multiple JSON files.
    
    Args:
        num_folds (int): Number of folds in cross-validation
    
    Returns:
        dict: Dictionary containing mean and std for each metric
    """
    
    # Dictionary to store all metrics across folds
    all_metrics = defaultdict(list)
    
    # Read results from each fold
    for fold in range(1, num_folds+1):
        filename = os.path.join(folder_path, f"fold{fold}_metrics_epoch30.json") if folder_path else f"fold{fold}_metrics_epoch30.json"
        
        try:
            with open(filename, 'r') as f:
                fold_results = json.load(f)
            
            # Store each metric value
            for metric, value in fold_results.items():
                all_metrics[metric].append(value*100)
                
            print(f"Loaded results from {filename}")
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {filename}, skipping...")
    
    # Calculate mean and standard deviation for each metric
    aggregated_results = {}
    
    for metric, values in all_metrics.items():
        if values:  # Only process if we have values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # Sample std deviation
            
            aggregated_results[metric] = {
                'mean': round(mean_val, 2),
                'std': round(std_val, 2),
                'values': values  # Keep original values for reference
            }
    
    return aggregated_results

def print_results(results):
    """
    Pretty print the aggregated results.
    
    Args:
        results (dict): Aggregated results dictionary
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*60)
    # print(f"{'Metric':<20} {'Mean':<10} {'Std Dev':<10}")
    print(f"{'Metric':<20} {'Mean±Std':<20}")
    print("-"*40)
    
    for metric, stats in results.items():
        # print(f"{metric:<20} {stats['mean']:<10} {stats['std']:<10}")
        print(f"{metric:<20} {stats['mean']}±{stats['std']:<20}")
    
    print("-"*40)
    print(f"Results based on {len(next(iter(results.values()))['values'])} folds")

def save_aggregated_results(results, filename="cv_aggregated_results.json"):
    """
    Save aggregated results to a JSON file.
    
    Args:
        results (dict): Aggregated results dictionary
        filename (str): Output filename
    """
    # Create a clean version without the 'values' field for saving
    clean_results = {}
    for metric, stats in results.items():
        clean_results[metric] = {
            'mean': stats['mean'],
            'std': stats['std']
        }
    
    with open(filename, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nAggregated results saved to {filename}")

# Main execution
if __name__ == "__main__":
    subfolder = "EP8"
    results = aggregate_cv_results(num_folds=5, folder_path=f"output_icd/{subfolder}")
    
    if results:
        # Print summary
        print_results(results)
        
        # Save to file
        save_aggregated_results(results, filename=f"output_icd/ep8/{subfolder}/cv_aggregated_results.json")
        
        # Optional: Print detailed breakdown
        # print("\n" + "="*60)
        # print("DETAILED BREAKDOWN")
        # print("="*60)
        # for metric, stats in results.items():
        #     print(f"\n{metric.upper()}:")
        #     print(f"  Individual fold values: {stats['values']}")
        #     print(f"  Mean: {stats['mean']}")
        #     print(f"  Standard Deviation: {stats['std']}")
    else:
        print("No valid result files found!")