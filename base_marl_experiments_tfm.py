from __future__ import annotations

import copy
import os
import kubernetes_emmulator as ke
import codeco_rl_env_parallel as cr_par
import data_preprocess as dp
import torch as th
import random
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from torch.serialization import add_safe_globals
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import seaborn as sns


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

def list_agent_folders(base_path, max_agent_number):
    """
    List the paths of the model.pt files of the agents in a base directory.

    Parameters
    ----------
    base_path : str
        The base directory where the agent folders are located.
    max_agent_number : int
        The maximum number of agents to look for.

    Returns
    -------
    agent_folders : list
        A list of the paths of the model.pt files of the agents.
    """
    agent_folders = []
    for i in range(0, max_agent_number):
        agent_folder = os.path.join(base_path, f'agent{i}')
        if os.path.isdir(agent_folder):
            path = os.path.join(agent_folder, 'model/model.pt')
            agent_folders.append(path)
    return agent_folders

def load_MARL_betting_models(nagents):
    """
    Load models for multiple agents from specified directories.

    This function loads models for a specified number of agents 
    from directories in a predefined base path. It uses the function 
    `list_agent_folders` to obtain paths to model files, loads the models 
    using PyTorch, and stores them in a dictionary keyed by agent identifiers.

    Parameters
    ----------
    nagents : int
        The number of agents to load models for.

    Returns
    -------
    models : dict
        A dictionary where keys are agent identifiers (e.g., 'agent0', 'agent1', ...)
        and values are the loaded PyTorch models.
    """

    base_path = 'models_tfm/10_agents_code_carbon/policies'
    path_models = list_agent_folders(base_path, nagents)
    models = {f'agent{i}': None for i in range(nagents)}
    
    for i, path in enumerate(path_models):
        add_safe_globals({'FullyConnectedNetwork': FullyConnectedNetwork})

        model = th.load(path, weights_only=False)
        print(model)
        models[f'agent{i}'] = model
        print(f"Loaded model for agent{i} from {path}")
    return models


def setup_environment(node_path, data_path, nagents, masked, device):
    """
    Set up the environment for a MARL experiment.

    Parameters
    ----------
    node_path : str
        Path to the node configuration data.
    data_path : str
        Path to the pod data.
    nagents : int
        The number of agents.
    masked : bool
        Whether to use a masked model.
    device : str or torch.device
        The device to load the models on.

    Returns
    -------
    env : CodecoEnv
        The Codeco environment.
    cluster_level_model : PPO or MaskablePPO
        The cluster-level model.
    critic : nn.Module
        The critic for the cluster-level model.
    critic_out : nn.Module
        The critic output for the cluster-level model.
    df_pods : pd.DataFrame
        The pod data.
    """
    agent_dfs, df_pods, nodes_info = dp.read_and_create_multiagent(node_path, data_path, nagents)

    env_config = {
        "df_node": [copy.deepcopy(value) for key, value in agent_dfs.items()],
        "df_data": df_pods,
        "nodes_info": [copy.deepcopy(nodes_info) for _ in range(nagents)],
        "emmulators": [ke.KubernetesEmulator({"df_node": agent_dfs[f"agent{a}"],
                                              "df_data": df_pods,
                                              "nodes_info": nodes_info[f"agent{a}"],
                                              "w_ld_cpu": 0,
                                              "w_ld_ram": 0,
                                              "w_eo": 3,
                                              "w_tc": 0,
                                              "w_ec": 5
                                            }) for a in range(nagents)],
        "num_agents": nagents,
        "padding": True
    }

    env = cr_par.CodecoEnv(env_config, render_mode="human")

    # Load the two models
    if masked:
        cluster_level_model = MaskablePPO.load("models_masking/masked_ppo20250410-151354.zip", device=device)
    else:
        cluster_level_model = PPO.load("ppo_4nodegreeness.zip", device=device)

    # Extract critics for both models
    critic = cluster_level_model.policy.mlp_extractor.value_net
    critic_out = cluster_level_model.policy.value_net
    critic.to(device)
    critic_out.to(device)
    critic.eval()
    critic_out.eval()

    # Return the environment, lists of the components for both models, and df_pods
    return env, cluster_level_model, critic, critic_out, df_pods


def run_experiment_base_marl(node_path, data_path, num_agents, device="cpu"):
    """
    Runs a single experiment for the base MARL environment.

    Parameters
    ----------
    node_path : str
        Path to the CSV file containing node information.
    data_path : str
        Path to the CSV file containing pod data.
    num_agents : int
        Number of agents to use in the experiment.
    device : str
        Device to use for the experiment (e.g. "cpu" or "cuda").

    Returns
    -------
    accumulated_rewards : list
        List of accumulated rewards for each pod.
    timestep_records : list
        List of dictionaries containing information about each pod, including the agent, action, and critic value.
    rewards_agent : dict
        Dictionary containing the rewards for each agent.
    critic_values_global : dict
        Dictionary containing the critic values for each agent.
    """
    env, model, critic, critic_out, pod_data = setup_environment(
        node_path, data_path, num_agents, masked=True, device=device
    )

    observations, _ = env.reset()
    accumulated_rewards = []
    timestep_records = []
    rewards_agent = {f"agent{i}": [] for i in range(num_agents)}
    critic_values_global = {f"agent{i}": [] for i in range(num_agents)}
    for pod in pod_data:
        pod_info = pod[1:3]
        actions = {f"agent{i}": 0 for i in range(num_agents)}
        critic_values = {f"agent{i}": 0 for i in range(num_agents)}

        for agent_index in range(num_agents):
            agent_obs = observations[f"agent{agent_index}"].copy()
            agent_obs[:2] = pod_info

            obs_tensor = th.tensor(agent_obs, dtype=th.float32, device=device).reshape(1, -1)
            action_mask = env.get_action_masks()[agent_index]
            action = model.predict(obs_tensor, deterministic=True, action_masks=action_mask)[0]
            actions[f"agent{agent_index}"] = int(action)

            with th.no_grad():
                critic_value = critic(obs_tensor)
                critic_output = critic_out(critic_value)
            critic_values[f"agent{agent_index}"] = critic_output.item()

            timestep_records.append({
                "timestep": pod_info,
                "agent": f"agent{agent_index}",
                "action": actions[f"agent{agent_index}"],
                "critic_value": round(critic_values[f"agent{agent_index}"], 3)
            })

        observations, rewards, terminations, truncations, infos = env.step(actions)
        max_reward = max(rewards.values())
        for agent_index in range(num_agents):
            rewards_agent[f"agent{agent_index}"].append(rewards[f"agent{agent_index}"])
            critic_values_global[f"agent{agent_index}"].append(critic_values[f"agent{agent_index}"])
        accumulated_rewards.append(max_reward)

    return accumulated_rewards, timestep_records, rewards_agent, critic_values_global

def clean_agent_data(rewards_agent, critic_values_global):
    """
    Clean agent data by removing rewards and critic values that are less than -20.
    This is done to remove any outliers that may have been generated by the environment.
    
    Parameters
    ----------
    rewards_agent : dict
        Dictionary containing the rewards for each agent.
    critic_values_global : dict
        Dictionary containing the critic values for each agent.
    """
    for agent in rewards_agent:
        rewards = rewards_agent[agent]
        critic_values = critic_values_global[agent]
        filtered = [(r, c) for r, c in zip(rewards, critic_values) if r > -20 and c > -20]
        if filtered:
            rewards_agent[agent], critic_values_global[agent] = map(list, zip(*filtered))
        else:
            rewards_agent[agent], critic_values_global[agent] = [], []

def compute_and_plot_correlations(rewards_agent, critic_values_global, output_dir):
    """
    Compute the correlation between the rewards and critic values for each agent.
    Plot these correlations, along with the mean rewards and mean critic values.
    The correlation is computed as the Pearson correlation coefficient.
    The loss is computed as the mean squared error between the rewards and critic values.
    The plot also includes a dashed line representing the best possible correlation.
    The plot is saved to a file named 'correlation_plot.png' in the output directory.

    Parameters
    ----------
    rewards_agent : dict
        Dictionary containing the rewards for each agent.
    critic_values_global : dict
        Dictionary containing the critic values for each agent.
    output_dir : str
        The directory where the plot will be saved.

    Returns
    -------
    correlation_coefficients : dict
        Dictionary containing the correlation coefficients for each agent.
    loss_values : dict
        Dictionary containing the loss values for each agent.
    plot_rewards : list
        List containing the mean rewards for each agent.
    plot_critic_values : list
        List containing the mean critic values for each agent.
    loss_values_noagent : list
        List containing the loss values for each agent, without the agent name.
    """
    correlation_coefficients = {}
    loss_values = {}

    plt.figure(figsize=(16, 10))

    # Use a colormap to assign a unique color to each agent
    colormap = plt.cm.get_cmap('tab10', len(rewards_agent))
    plot_rewards = []
    plot_critic_values = []
    loss_values_noagent = []
    for i, agent in enumerate(rewards_agent):
        rewards = np.array(rewards_agent[agent])
        critic_values = np.array(critic_values_global[agent])

        if len(rewards) == 0:
            continue

        mean_reward = np.mean(rewards)
        mean_critic_value = np.mean(critic_values)
        corr = np.corrcoef(rewards, critic_values)[0, 1]
        loss = np.mean((rewards - critic_values) ** 2) / np.sqrt(len(rewards))

        correlation_coefficients[agent] = corr
        loss_values[agent] = loss

        # Assign a unique color to each agent
        color = colormap(i)

        # Plot with vertical error bars
        plt.errorbar(mean_reward, mean_critic_value, yerr=loss, fmt='o', markersize=10, color=color, ecolor='blue', capsize=10, label=f'{agent} (Corr: {corr:.2f}, Loss: {loss:.2f})')

        plot_rewards.append(mean_reward)
        plot_critic_values.append(mean_critic_value)
        loss_values_noagent.append(loss)
        
    all_rewards = np.concatenate(list(rewards_agent.values()))
    all_critic_vals = np.concatenate(list(critic_values_global.values()))
    plt.plot([min(all_rewards), max(all_rewards)],
             [min(all_critic_vals), max(all_critic_vals)],
             color='red', linestyle='--', label='Good performance frontier')

    plt.xlabel('Mean Rewards')
    plt.ylabel('Mean Critic Values')
    plt.title('Correlation Plot for All Agents')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_plot.png"))
    plt.close()

    return correlation_coefficients, loss_values, plot_rewards, plot_critic_values, loss_values_noagent

def scatter_plot_rewards_critic(rewards_agent, critic_values_global, output_dir):
    """
    Creates a scatter plot for the rewards and critic values for all agents.

    Parameters:
    rewards_agent (dict): A dictionary containing rewards for each agent.
    critic_values_global (dict): A dictionary containing critic values for each agent.
    output_dir (str): The directory to save the plot in.

    Returns:
    None
    """
    plt.figure(figsize=(10, 8))

    for i, agent in enumerate(rewards_agent):
        rewards = np.array(rewards_agent[agent])
        critic_values = np.array(critic_values_global[agent])

        plt.scatter(rewards, critic_values, alpha=0.5, label=f'Agent {agent}')

    plt.title('Scatter Plot for All Agents')
    plt.xlabel('Rewards')
    plt.ylabel('Critic Values')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plot_filename = "scatter_plot_all_agents.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()

def plot_reward_boxplot(rewards_agent, output_dir, critic_values):
    """
    Plots a box plot of rewards and critic values for all agents side by side.

    Parameters:
    - rewards_agent (dict): A dictionary where keys are agent names and values are lists of rewards.
    - output_dir (str): The directory where the plot will be saved.
    - critic_values (dict): A dictionary where keys are agent names and values are lists of critic values.
    
    Returns:
    None
    """
    plt.figure(figsize=(16, 10))

    # Prepare data for box plot
    agents = list(rewards_agent.keys())
    rewards_data = list(rewards_agent.values())
    critic_data = list(critic_values.values())

    # Create the box plot for rewards and critic values
    positions_rewards = np.arange(len(agents)) * 2
    positions_critic = positions_rewards + 1

    plt.boxplot(rewards_data, positions=positions_rewards, widths=0.6, patch_artist=True, boxprops=dict(facecolor="lightblue"), label='Rewards')
    plt.boxplot(critic_data, positions=positions_critic, widths=0.6, patch_artist=True, boxprops=dict(facecolor="lightgreen"), label='Critic Values')

    # Add titles and labels
    plt.xlabel('Agents')
    plt.ylabel('Values')
    plt.title('Rewards and Critic Values Box Plot for All Agents')
    plt.ylim(0, 3)

    # Set x-ticks to the middle of each pair of box plots
    plt.xticks(positions_rewards + 0.5, agents)

    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_box_plot_with_critic.png"))
    plt.close()

def plot_correlation_heatmap(correlation_coefficients, output_dir):
    """
    Plot a heatmap of correlation coefficients for all agents.

    Parameters:
    - correlation_coefficients (dict): A dictionary where keys are agent names and values are correlation coefficients.
    - output_dir (str): The directory where the plot will be saved.

    Returns:
    None
    """
    
    df = pd.DataFrame(correlation_coefficients, index=['Correlation'])

    plt.figure(figsize=(10, 1.5))
    ax = sns.heatmap(df, annot=True, cmap='flare', cbar=True, vmin=-1, vmax=1, 
                     linewidths=0.5, linecolor='white', annot_kws={"size": 10})

    plt.title('Correlation Coefficients Heatmap', pad=10)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), bbox_inches='tight', dpi=300)
    plt.close()

def plot_experiment_data(data, title, ylabel, filename, summary_dir):
    # Regroup data by experiment set index
    # Instead of 10 it can be nagents
    agent_counts = sorted(data.keys(), key=lambda x: int(x))
    experiment_sets = [data[agents] for agents in agent_counts]

    plt.figure(figsize=(10, 6))
    plt.boxplot(experiment_sets, patch_artist=True)
    plt.xticks(ticks=range(1, len(agent_counts)+1), labels=agent_counts)
    plt.xlabel('Number of Agents')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, filename))
    plt.close()


def main():
    node_base_path = "data/node_data10/"
    data_base_path = "data/pod_data/"

    csv_files = glob.glob(os.path.join(data_base_path, "*.csv"))
    csv_node_files = glob.glob(os.path.join(node_base_path, "*.csv"))

    data_node_path = csv_node_files[0]
    file_node_name = os.path.basename(data_node_path).split('.')[0]
    print(file_node_name)

    avg_correlation_by_nagents = []
    avg_loss_by_nagents = []
    experiments_rewards = {}
    experiments_critic = {}
    experiments_loss = {}

    for nagents in range(2, 11):
        print(f"\nRunning for {nagents} agents...")

        output_dir = f"results_betting_tests_codeco/{nagents}_agents"
        os.makedirs(output_dir, exist_ok=True)

        data_path = csv_files[-1]

        # Rewards and timestep data not used in this script
        rewards, timestep_data, rewards_agent, critic_values_global = run_experiment_base_marl(
            data_node_path, data_path, nagents)

        clean_agent_data(rewards_agent, critic_values_global)

        correlation_coefficients, loss_values, plot_rewards, plot_critic_values, loss_values_noagent = compute_and_plot_correlations(
            rewards_agent, critic_values_global, output_dir)

        experiments_rewards[str(nagents)] = plot_rewards
        experiments_critic[str(nagents)] = plot_critic_values
        experiments_loss[str(nagents)] = loss_values_noagent
        
        plot_reward_boxplot(rewards_agent, output_dir, critic_values_global)
        plot_correlation_heatmap(correlation_coefficients, output_dir)
        scatter_plot_rewards_critic(rewards_agent, critic_values_global, output_dir)

        for agent, corr in correlation_coefficients.items():
            print(f"Agent {agent}: Correlation = {corr:.4f}")
        for agent, loss in loss_values.items():
            print(f"Agent {agent}: Loss = {loss:.4f}")

        avg_corr = np.mean(list(correlation_coefficients.values())) if correlation_coefficients else 0
        avg_loss = np.mean(list(loss_values.values())) if loss_values else 0
        avg_correlation_by_nagents.append((nagents, avg_corr))
        avg_loss_by_nagents.append((nagents, avg_loss))

    # Plotting results, summarizing results for agents
    
    # Plot average correlation vs number of agents
    summary_dir = "results_betting_tests_codeco"
    os.makedirs(summary_dir, exist_ok=True)

    agent_counts, avg_corrs = zip(*avg_correlation_by_nagents)
    plt.figure(figsize=(10, 6))
    plt.plot(agent_counts, avg_corrs, marker='o', linestyle='-', label='Average Correlation')
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Correlation")
    plt.title("Average Correlation vs Number of Agents")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "average_correlation_vs_agents.png"))
    plt.close()

    # Plot average loss vs number of agents
    agent_counts, avg_losses = zip(*avg_loss_by_nagents)
    plt.figure(figsize=(10, 6))
    plt.plot(agent_counts, avg_losses, marker='o', linestyle='-', color='red', label='Average Loss')
    plt.xlabel("Number of Agents")
    plt.ylabel("Average Loss")
    plt.title("Average Loss vs Number of Agents")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "average_loss_vs_agents.png"))
    plt.close()

    plot_experiment_data(experiments_rewards, 'Rewards Over Experiment Sets', 'Rewards', 'rewards_vs_experiments.png', summary_dir)

    # Plot critic values
    plot_experiment_data(experiments_critic, 'Critic Values Over Experiment Sets', 'Critic Values', 'critic_values_vs_experiments.png', summary_dir)

    # Plot loss values
    plot_experiment_data(experiments_loss, 'Loss Over Experiment Sets', 'Loss', 'loss_vs_experiments.png', summary_dir)

if __name__ == "__main__":
    main()