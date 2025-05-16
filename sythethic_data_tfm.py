import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_rows, num_agents, output_dir='synthetic_data'):
    """
    Generate synthetic data for multiple agents and save it to CSV files.

    Parameters
    ----------
    num_rows : int
        Number of rows of data to generate for each agent.
    num_agents : int
        Number of agents for which to generate data.
    output_dir : str, optional
        Directory where the CSV files will be saved (default is 'synthetic_data').

    Notes
    -----
    - The function seeds the random number generator for reproducibility.
    - Each agent's data includes 'Reward', 'CPU_Usage', and 'RAM_Usage'.
    - The 'Reward' values are generated as random floats between -20 and 20.
    - 'CPU_Usage' and 'RAM_Usage' are random integers between 0 and 100.
    - Data for each agent is saved in a separate CSV file in the specified directory.
    """

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