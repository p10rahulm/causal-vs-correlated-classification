import os
from pathlib import Path
import sys

# Add project root to system path
project_root = Path(__file__).resolve().parent
while not (project_root / '.git').exists() and project_root != project_root.parent:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import re
import unicodedata

def clean_text(text, is_first_column=False):
    # Preserve starting and ending quotes for the first column
    start_quote = '"' if is_first_column else ''
    end_quote = '"' if is_first_column else ''
    
    # Remove non-ASCII characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Replace double quotes with single quotes, except at the start and end of the first column
    if is_first_column:
        text = text.strip('"')
        text = text.replace('"', "'")
    # else:
    #     text = text.replace('"', "'")
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return start_quote + text + end_quote

# Read the CSV file
input_file = 'data/ood_genres.csv'
output_file = 'data/ood_genres_cleaned.csv'

# Read the CSV file without parsing it
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned_lines = []
for line in lines:
    parts = line.split('|')
    cleaned_parts = [
        clean_text(parts[0], is_first_column=True),
        clean_text(parts[1]) if len(parts) > 1 else '',
        parts[2].strip() if len(parts) > 2 else ''
    ]
    cleaned_lines.append('|'.join(cleaned_parts) + '\n')

# Write the cleaned data to a new CSV file
with open(output_file, 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print(f"Cleaned data has been written to {output_file}")