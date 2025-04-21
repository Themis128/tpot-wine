import os
import pandas as pd

# Define paths
input_dir = "/home/tbaltzakis/tpot-wine/data/processed_filled"
output_file = "/home/tbaltzakis/tpot-wine/AutoML_1/39_RandomForest/data/cleaned_combined_data.csv"

# Remove the old combined file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"Removed old file: {output_file}")

# List all CSV files in the directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# Merge and preprocess all CSV files
dataframes = []
for csv_file in csv_files:
    file_path = os.path.join(input_dir, csv_file)
    print(f"Reading file: {file_path}")
    df = pd.read_csv(file_path)

    # Preprocessing steps
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
    df = df.drop_duplicates()  # Remove duplicate rows
    df = df.fillna(method='ffill').fillna(method='bfill')  # Fill missing values (forward and backward fill)

    # Format timestamps to EU numerical format (if a 'date' column exists)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%d/%m/%Y')

    # Round numerical columns to 2 decimals
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].round(2)

    # Append processed dataframe
    dataframes.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Final preprocessing
combined_df = combined_df.drop_duplicates()  # Ensure no duplicates after merging
combined_df.columns = combined_df.columns.str.lower()  # Standardize column names to lowercase

# Save the combined dataframe to a new CSV file with EU-style formatting
os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the output directory exists
combined_df.to_csv(output_file, index=False, float_format="%.2f", decimal=",")
print(f"New combined file created: {output_file}")
