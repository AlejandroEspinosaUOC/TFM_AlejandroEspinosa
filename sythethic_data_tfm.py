import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_rows, num_agents, output_dir='synthetic_data'):
    np.random.seed(42)  # For reproducibility

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    for N in range(0, num_agents):
        data = {
            'Reward': (np.random.rand(num_rows) * 40) - 20, 
            'CPU_Usage': np.random.randint(0, 101, num_rows), 
            'RAM_Usage': np.random.randint(0, 101, num_rows),  
        }

        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(f'{output_dir}/synthetic_data_{N}_Inference.csv', index=False)


        
# Example usage
generate_synthetic_data(num_rows=10000, num_agents=10)