import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def create_small_dataset(input_path, output_path, num_rows=10000):
    """
    Create a smaller dataset by sampling from a larger dataset
    
    Parameters:
    -----------
    input_path : str
        Path to the original dataset
    output_path : str
        Path to save the smaller dataset
    num_rows : int
        Number of rows to sample
    """
    print(f"Reading original dataset from {input_path}...")
    
    # Read large CSV file with a chunksize to reduce memory usage
    chunks = pd.read_csv(input_path, chunksize=20000)
    
    # Create an empty list to store the samples
    sampled_chunks = []
    
    # Calculate how many rows to sample from each chunk
    # to get roughly 10000 rows total (estimating 10 chunks)
    rows_per_chunk = max(1, num_rows // 10)
    
    # Read chunks and sample from each
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}...")
        # Check if we already have enough rows
        if sum(len(df) for df in sampled_chunks) >= num_rows:
            break
            
        # Sample rows from this chunk
        sampled_chunk = chunk.sample(min(rows_per_chunk, len(chunk)))
        sampled_chunks.append(sampled_chunk)
        
        # Print progress
        total_rows = sum(len(df) for df in sampled_chunks)
        print(f"Collected {total_rows} rows so far")
    
    # Combine all sampled chunks
    small_df = pd.concat(sampled_chunks)
    
    # If we have more than num_rows, downsample to exactly num_rows
    if len(small_df) > num_rows:
        small_df = small_df.sample(num_rows)
    
    # Save the smaller dataset
    print(f"Saving {len(small_df)} rows to {output_path}...")
    small_df.to_csv(output_path, index=False)
    print("Done!")
    
    return small_df

if __name__ == "__main__":
    # Create small dataset
    input_path = "data.csv"
    output_path = "medium_data.csv"  # Changed name to reflect larger size
    df = create_small_dataset(input_path, output_path, num_rows=10000)
    
    # Provide some statistics about the data
    print("\nDataset information:")
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    # Check for target variable
    for col in df.columns:
        if 'risk' in col.lower() or 'grade' in col.lower() or 'rating' in col.lower():
            print(f"\nTarget column: {col}")
            print(f"Class distribution:")
            print(df[col].value_counts()) 