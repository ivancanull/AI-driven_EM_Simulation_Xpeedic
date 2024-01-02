import os
import pickle
import pandas as pd
import numpy as np
from .config import Config


def parse_generated_s_parameters(
    s: str,
    port: int,
) -> pd.DataFrame:
    """
    Parse the new version s-parameters files
    """
    # Initialize an empty list to store processed rows
    processed_rows = []

    # Read the CSV file line by line
    # Replace 'your_compact_data.csv' with your actual file name
    with open(s, 'r') as file:
        rows = file.readlines()

    # Process each row and compact rows
    compact_row = []
    for row in rows:
        # Remove leading and trailing whitespace
        row = row.strip('\n').strip()
        
        # Skip empty rows
        if not row or row.startswith(('!', '#')) or row == '\t':
            continue
        
        # Check if the row begins with a tab
        if row.startswith('\t'):
            # Split by tab and remove the empty first item
            items = row.split('\t')[1:]
        else:
            # Split by tab
            items = row.split('\t')
        
        # Append items to compact_row
        compact_row.append(items)
        
        # If compact_row has 4 items, add to processed_rows and reset compact_row
        row_change = int(np.ceil(port / 4) * port)

        if len(compact_row) == row_change:
            processed_rows.append([float(number) for sublist in compact_row for number in sublist])
            compact_row = []

    # Concatenate processed rows into a DataFrame
    final_data = pd.DataFrame(processed_rows)
    return final_data


