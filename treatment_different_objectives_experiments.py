import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_average_improvement(base_path='.'):
    mean_masked_list = []
    mean_marl_list = []
    std_masked_list = []
    std_marl_list = []

    for seed in range(1, 20):  # seeds from 1 to 19
        folder = os.path.join(base_path, f'node_info_tfm_seed{seed}')
        csv_path = os.path.join(folder, 'results_backup10agents_mix.csv')

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Compute per-folder mean of mean and std columns
            masked_avg = df['mean_masked'].mean()
            marl_avg = df['mean_marl'].mean()
            masked_std_avg = df['masked_rewards_std'].mean()
            marl_std_avg = df['marl_rewards_std'].mean()

            mean_masked_list.append(masked_avg)
            mean_marl_list.append(marl_avg)
            std_masked_list.append(masked_std_avg)
            std_marl_list.append(marl_std_avg)
        else:
            print(f"Warning: File not found - {csv_path}")

    # Final averages across all seeds
    overall_masked_avg = sum(mean_masked_list) / len(mean_masked_list)
    overall_marl_avg = sum(mean_marl_list) / len(mean_marl_list)
    overall_masked_std = sum(std_masked_list) / len(std_masked_list)
    overall_marl_std = sum(std_marl_list) / len(std_marl_list)

    improvement = overall_marl_avg - overall_masked_avg
    percent_improvement = (improvement / overall_masked_avg) * 100 if overall_masked_avg != 0 else 0

    print(f"Average mean_masked     : {overall_masked_avg:.6f}")
    print(f"Average mean_marl       : {overall_marl_avg:.6f}")
    print(f"Absolute improvement    : {improvement:.6f}")
    print(f"Percentage improvement  : {percent_improvement:.2f}%")
    print(f"Average Masked std dev  : {overall_masked_std:.6f}")
    print(f"Average Marl std dev    : {overall_marl_std:.6f}")

def plot_rewards(total_rewards_masked, total_rewards_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(total_rewards_masked))
    ax.plot(x_values, total_rewards_masked, label='Base MARL', marker='o', linestyle='-', color='b')
    ax.plot(x_values, total_rewards_marl, label='Complex MARL', marker='x', linestyle='-', color='orange')
    ax.set_xlabel('Number of Pods in Experiment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Rewards: Base MARL vs Complex MARL')
    ax.legend()
    ax.set_xticks(np.arange(0, len(x_values), 25))
    plt.tight_layout()

def plot_rewards_moving_average(total_rewards_masked, total_rewards_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(total_rewards_masked))
    window_size = 25
    moving_avg_masked = pd.Series(total_rewards_masked).rolling(window=window_size).mean()
    moving_avg_marl = pd.Series(total_rewards_marl).rolling(window=window_size).mean()
    ax.plot(x_values, moving_avg_masked, label='Base MARL (Moving Avg)', linestyle='-', color='b')
    ax.plot(x_values, moving_avg_marl, label='Complex MARL (Moving Avg)', linestyle='-', color='orange')
    ax.set_xlabel('Number of Pods in Experiment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Rewards: Base MARL vs Complex MARL')
    ax.legend()
    ax.set_xticks(np.arange(0, len(x_values), 25))
    plt.tight_layout()

def plot_rewards_with_std(mean_masked, std_masked, mean_marl, std_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(mean_masked))
    ax.plot(x_values, mean_masked, label='Base MARL (Mean)', color='blue')
    ax.plot(x_values, mean_marl, label='Complex MARL (Mean)', color='orange')
    ax.fill_between(x_values,
                    np.array(mean_masked) - np.array(std_masked),
                    np.array(mean_masked) + np.array(std_masked),
                    color='blue', alpha=0.2, label='Masked Model ± Std')
    ax.fill_between(x_values,
                    np.array(mean_marl) - np.array(std_marl),
                    np.array(mean_marl) + np.array(std_marl),
                    color='orange', alpha=0.2, label='MARL ± Std')
    ax.set_xlabel('Number of Pods in Experiment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Per-Experiment Rewards ± Std Dev: Base MARL vs Complex MARL')
    ax.legend()
    ax.set_xticks(np.arange(0, len(x_values), 25))
    plt.tight_layout()
    
def regenerate_plots_from_csv(csv_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Extract data columns
    total_rewards_masked = df["total_rewards_masked"].tolist()
    masked_rewards_std = df["masked_rewards_std"].tolist()
    mean_masked = df["mean_masked"].tolist()

    total_rewards_marl = df["total_rewards_marl"].tolist()
    marl_rewards_std = df["marl_rewards_std"].tolist()
    mean_marl = df["mean_marl"].tolist()

    # Plot and save figures
    plt.figure()
    plot_rewards(total_rewards_masked, total_rewards_marl)
    plt.savefig(os.path.join(output_dir, 'rewards_plot.png'))
    plt.close()

    plt.figure()
    plot_rewards_moving_average(total_rewards_masked, total_rewards_marl)
    plt.savefig(os.path.join(output_dir, 'rewards_moving_average_plot_dif.png'))
    plt.close()

    plt.figure()
    plot_rewards_with_std(mean_masked, masked_rewards_std, mean_marl, marl_rewards_std)
    plt.savefig(os.path.join(output_dir, 'rewards_with_std_plot_dif.png'))
    plt.close()

def regenerate_all_seeds(base_dir):
    for seed in range(1, 20):  # from 1 to 19 inclusive
        folder_name = f"node_info_tfm_seed{seed}"
        folder_path = os.path.join(base_dir, folder_name)
        csv_path = os.path.join(folder_path, "results_backup10agents_mix.csv")
        
        if os.path.exists(csv_path):
            print(f"Generating plots for seed {seed}...")
            regenerate_plots_from_csv(csv_path, folder_path)
        else:
            print(f"CSV not found for seed {seed}: {csv_path}")
# Run the function
#compute_average_improvement(base_path="results_tfm/4agents")
#regenerate_plots_from_csv("results_tfm/10agents_mixed_objectives/node_info_tfm_seed9/results_backup10agents_mix.csv", #                          "results_tfm/10agents_mixed_objectives/node_info_tfm_seed9")

# Uncomment to visualize plots with smaller font sizes, done for Thesis!
plt.rcParams.update({
    'axes.titlesize': 20,    
    'axes.labelsize': 16,  
    'xtick.labelsize': 12,   
    'ytick.labelsize': 12, 
    'legend.fontsize': 12,  
    'lines.markersize': 8, 
})
for i in range(4, 11):
    if i == 4 or i == 10:
        path = f"results_tfm/{i}agents"
        regenerate_all_seeds(path)

                          