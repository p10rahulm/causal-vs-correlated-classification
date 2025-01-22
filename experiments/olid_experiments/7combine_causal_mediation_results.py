import pandas as pd
import glob
import os
from pathlib import Path
import sys
from datetime import datetime

# Set CUDA DEVICE
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))



import pandas as pd

causal_files = [
    'outputs/olid_offensive_causal_mediation/results_20250103_214027.csv',
    'outputs/olid_offensive_causal_mediation/results_20250103_214131.csv',
    'outputs/olid_offensive_causal_mediation/results_20250103_214216.csv',
    'outputs/olid_offensive_causal_mediation/results_20250103_214541.csv',
    'outputs/olid_offensive_causal_mediation/results_20250106_060204.csv'
]
causal_files.sort()

# Read and combine causal mediation files
causal_dfs = []
for file in causal_files:
    df = pd.read_csv(file)
    df_final = df[df['epoch'] == 'final_test']
    df_final = df_final.drop('epoch', axis=1)
    df_final = df_final.rename(columns={'regularized_epochs': 'epochs'})
    df_final['experiment_type'] = 'causal_mediation'
    causal_dfs.append(df_final)

# Combine causal mediation results
causal_combined = pd.concat(causal_dfs, ignore_index=True)

# Get all baseline files
baseline_files = [
    'outputs/olid_offensive_classifier_naive_baseline/results_20241230_200701.csv'
]
baseline_files.sort()

# Read and combine baseline files
baseline_dfs = []
for file in baseline_files:
    df = pd.read_csv(file)
    df['experiment_type'] = 'naive_baseline'
    df = df.rename(columns={
        'accuracy': 'test_accuracy',
        'precision': 'test_precision',
        'recall': 'test_recall',
        'f1': 'test_f1'
    })
    baseline_dfs.append(df)

# Combine baseline results
baseline_combined = pd.concat(baseline_dfs, ignore_index=True)
baseline_combined['lambda_reg'] = None

# Read causal subset results
causal_subset_file = 'outputs/olid_causal_only_precomputed/results_20241231_100417.csv'
causal_subset_df = pd.read_csv(causal_subset_file)
causal_subset_df['experiment_type'] = 'causal_subset'
causal_subset_df['lambda_reg'] = None
causal_subset_df = causal_subset_df.rename(columns={
    'final_loss': 'train_loss'
})

# Combine all datasets
all_results = pd.concat([causal_combined, baseline_combined, causal_subset_df], ignore_index=True)

# Reorder columns to put experiment_type second
cols = all_results.columns.tolist()
cols.remove('experiment_type')
cols.insert(1, 'experiment_type')

# Sort by model (ascending) and experiment_type (descending)
all_results = all_results[cols].sort_values(
    ['model', 'experiment_type'], 
    ascending=[True, False]
)

# Create timestamp for the output file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'outputs/olid_offensive_combined/results_all_{timestamp}.csv'

# Ensure directory exists
os.makedirs('outputs/olid_offensive_combined', exist_ok=True)

# Save combined results
all_results.to_csv(output_path, index=False)

print(f"Files processed:")
print(f"- Causal mediation files: {len(causal_files)}")
print(f"- Baseline files: {len(baseline_files)}")
print(f"- Causal subset files: 1")
print(f"Total rows in combined file: {len(all_results)}")
print(f"Column order:", cols)
print(f"Output saved to: {output_path}")