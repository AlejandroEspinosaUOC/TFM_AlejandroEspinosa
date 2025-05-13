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
import numpy as np
import glob
import matplotlib.pyplot as plt

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from torch.serialization import add_safe_globals
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import warnings

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)
th.cuda.manual_seed_all(SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

def list_agent_folders(base_path, max_agent_number):
    agent_folders = []
    for i in range(0, max_agent_number):
        agent_folder = os.path.join(base_path, f'agent{i}')
        if os.path.isdir(agent_folder):
            path = os.path.join(agent_folder, 'model/model.pt')
            agent_folders.append(path)
    return agent_folders

def load_MARL_betting_models(nagents):
    base_path = 'models_tfm/4_agents_code_carbon2/policies'
    path_models = list_agent_folders(base_path, nagents)
    models = {f'agent{i}': None for i in range(nagents)}
    for i, path in enumerate(path_models):
        add_safe_globals({'FullyConnectedNetwork': FullyConnectedNetwork})

        model = th.load(path, weights_only=False)
        models[f'agent{i}'] = model
    return models


def setup_environment(node_path, data_path, nagents, masked, device, alternate_behavior=True):
    agent_dfs, df_pods, nodes_info = dp.read_and_create_multiagent(node_path, data_path, nagents)

    env_config = {
        "df_node": [copy.deepcopy(value) for key, value in agent_dfs.items()],
        "df_data": df_pods,
        "nodes_info": [copy.deepcopy(nodes_info) for _ in range(nagents)],
        "emmulators": [
            ke.KubernetesEmulator({
                "df_node": agent_dfs[f"agent{a}"],
                "df_data": df_pods,
                "nodes_info": nodes_info[f"agent{a}"],
                "w_ld_cpu": 5 if alternate_behavior and a % 2 == 0 else 0,
                "w_ld_ram": 5 if alternate_behavior and a % 2 == 0 else 0,
                "w_eo": 0 if alternate_behavior and a % 2 == 0 else 3,
                "w_tc": 0,
                "w_ec": 5
            }) for a in range(nagents)
        ],
        "num_agents": nagents,
        "padding": True
    }

    env = cr_par.CodecoEnv(env_config, render_mode="human")
    obs, _ = env.reset()
    if masked:
        model1 = MaskablePPO.load("models_masking/masked_ppo20250410-151354.zip", device=device)
        model2 = MaskablePPO.load("models_masking/masked_ppo_balancing20250422-181513.zip", device=device)
    else:
        model1 = PPO.load("ppo_4nodegreeness.zip", device=device)
        model2 = PPO.load("another_ppo_model.zip", device=device)

    # Critic components
    models = [model1, model2]
    critics = [m.policy.mlp_extractor.value_net.to(device).eval() for m in models]
    critic_outs = [m.policy.value_net.to(device).eval() for m in models]

    return env, models, critics, critic_outs, df_pods


def run_experiment_base_marl(node_path, data_path, num_agents, device="cpu", alternate_behavior=True):
    env, models, critics, critic_outs, pod_data = setup_environment(
        node_path, data_path, num_agents, masked=True, device=device, alternate_behavior=alternate_behavior
    )

    observations, _ = env.reset()
    accumulated_rewards = []
    timestep_records = []

    for pod in pod_data:
        pod_info = pod[1:3]
        actions = {}
        critic_values = {}

        for agent_index in range(num_agents):
            agent_obs = observations[f"agent{agent_index}"].copy()
            agent_obs[:2] = pod_info

            if alternate_behavior and agent_index % 2 != 0:
                agent_obs = list(agent_obs)
                index = 5
                while index < len(agent_obs):
                    agent_obs.pop(index)
                    index += 3
       
                model = models[1]
                critic = critics[1]
                critic_out = critic_outs[1]
            else:
                model = models[0]
                critic = critics[0]
                critic_out = critic_outs[0]

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

        observations, rewards, _, _, _ = env.step(actions)
        accumulated_rewards.append(max(rewards.values()))

    return accumulated_rewards, timestep_records


def run_experiment_marl(node_path, data_path, nagents, device="cpu", alternate_behavior=True):
    env, cluster_models, critics, critic_outs, df_pods = setup_environment(
        node_path, data_path, nagents, masked=True, device=device, alternate_behavior=alternate_behavior
    )
    models = load_MARL_betting_models(nagents)

    marl_observations = {f"agent{a}": [] for a in range(nagents)}
    marl_bettings = {f"agent{a}": [0] for a in range(nagents)}
    marl_perct_available = {f"agent{a}": [] for a in range(nagents)}
    timestep_data = []
    rewards_accum = []

    observations, _ = env.reset()
    for pod in df_pods:
        pod_id = pod[1:3]
        actor_values = {}
        critic_outputs = {}

        for agent_id in range(nagents):
            agent_obs = observations[f"agent{agent_id}"].copy()
            agent_obs[:2] = pod_id

            if alternate_behavior and agent_id % 2 != 0:
                agent_obs = list(agent_obs)
                index = 5
                while index < len(agent_obs):
                    agent_obs.pop(index)
                    index += 3
                model = cluster_models[1]
                critic = critics[1]
                critic_out = critic_outs[1]
            else:
                model = cluster_models[0]
                critic = critics[0]
                critic_out = critic_outs[0]

            tensor = th.tensor(agent_obs, dtype=th.float32, device=device).reshape(1, -1)
            mask = env.get_action_masks()[agent_id]
            action = model.predict(tensor, deterministic=True, action_masks=mask)[0]
            actor_values[f"agent{agent_id}"] = int(action)

            with th.no_grad():
                critic_value = critic(tensor)
                critic_output = critic_out(critic_value)
            critic_outputs[f"agent{agent_id}"] = critic_output.item()

            perct_available = env.get_percnt_available(f"agent{agent_id}")
            perct_available = [round(p, 3) for p in perct_available]
            marl_perct_available[f"agent{agent_id}"].append(perct_available)

            betting_averages = [
                sum(marl_bettings[f"agent{a}"][-3:]) / len(marl_bettings[f"agent{a}"][-5:])
                if len(marl_bettings[f"agent{a}"]) >= 5 else marl_bettings[f"agent{a}"][-1]
                for a in range(nagents)
                if a != agent_id
            ]

            obs_input = [critic_output.item()] + perct_available + betting_averages
            
            marl_observations[f"agent{agent_id}"] = obs_input

            obs_tensor = th.tensor(obs_input, dtype=th.float32, device=device).reshape(1, -1)
            with th.no_grad():
                input_dict = {"obs": obs_tensor}
                betting = models[f"agent{agent_id}"](input_dict)
                betting = np.argmax(betting[0].cpu().numpy())

            marl_bettings[f"agent{agent_id}"].append(int(betting))

            timestep_data.append({
                "pod_id": pod_id,
                "agent": f"agent{agent_id}",
                "actor_value": actor_values[f"agent{agent_id}"],
                "critic_output": round(critic_outputs[f"agent{agent_id}"], 3),
                "betting": betting,
                "perct_available": perct_available
            })

        winner_id = max(marl_bettings, key=lambda a: marl_bettings[a][-1])
        marl_winner = {f"agent{a}": 1 if f"agent{a}" == winner_id else 0 for a in range(nagents)}
        observations, rewards, _, _, _ = env.step(actor_values, marl=marl_winner)
        rewards_accum.append(max(rewards.values()))

    return rewards_accum, timestep_data


def plot_rewards(total_rewards_masked, total_rewards_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(total_rewards_masked))

    # Plot the total rewards for masked model and MARL
    ax.plot(x_values, total_rewards_masked, label='Base MARL', marker='o', linestyle='-', color='b')
    ax.plot(x_values, total_rewards_marl, label='Complex MARL', marker='x', linestyle='-', color='orange')

    # Add labels and title
    ax.set_xlabel('Number of Pods in Experiment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Rewards: Base MARL vs Complex MARL')
    ax.legend()

    # Set x-axis ticks to display every 25 values
    ax.set_xticks(np.arange(0, len(x_values), 25))

    # Store plot (or uncomment to show it too)
    plt.tight_layout()
    #plt.show()

def plot_rewards_moving_average(total_rewards_masked, total_rewards_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(total_rewards_masked))

    # Calculate and plot the moving average for the masked model
    window_size = 25  # Adust window size if needed
    moving_avg_masked = pd.Series(total_rewards_masked).rolling(window=window_size).mean()
    ax.plot(x_values, moving_avg_masked, label='Base MARL (Moving Avg)', linestyle='-', color='b')

    # Calculate and plot the moving average for the MARL
    moving_avg_marl = pd.Series(total_rewards_marl).rolling(window=window_size).mean()
    ax.plot(x_values, moving_avg_marl, label='Complex MARL (Moving Avg)', linestyle='-', color='orange')

    # Add labels and title
    ax.set_xlabel('Number of Pods in Experiment')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Rewards: Base MARL vs Complex MARL')
    ax.legend()

    # Set x-axis ticks to display every 25 values
    ax.set_xticks(np.arange(0, len(x_values), 25))

    # Store plot (or uncomment to show it too)
    plt.tight_layout()
    #plt.show()
def plot_rewards_with_std(mean_masked, std_masked, mean_marl, std_marl):
    fig, ax = plt.subplots(figsize=(12, 6))
    x_values = range(len(mean_masked))

    # Plot mean reward per episode
    ax.plot(x_values, mean_masked, label='Base MARL (Mean)', color='blue')
    ax.plot(x_values, mean_marl, label='Complex MARL (Mean)', color='orange')

    # Fill between ± std deviation
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
  
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    node_base_path = "data/node_data10/"
    data_base_path = "data/pod_data/"
    nagents = 4

    csv_files = glob.glob(os.path.join(data_base_path, "*.csv"))
    csv_node_files = glob.glob(os.path.join(node_base_path, "*.csv"))

    for file_node in csv_node_files:
        data_node_path = file_node
        # Get the base name of the file_node
        file_node_name = os.path.basename(file_node).split('.')[0]
        output_dir = f"tests/{file_node_name}"
        # Create if directory does not exist
        os.makedirs(output_dir, exist_ok=True)

        masked_rewards_std = []
        mean_masked = []
        marl_rewards_std = []
        mean_marl = []
        total_rewards_masked = []
        total_rewards_marl = []

        results = []

        for file in csv_files:
            data_path = file
            # Run the experiment using the base model
            rewards_masked, timestep_data_masked = run_experiment_base_marl(data_node_path, data_path, nagents, device="cpu")
            total_rewards_masked.append(np.average(rewards_masked))
            masked_rewards_std.append(np.std(rewards_masked))
            mean_masked.append(np.mean(rewards_masked))

            # Run the experiment using the MARL component
            rewards_marl, timestep_data_marl = run_experiment_marl(data_node_path, data_path, nagents, device="cpu")
            total_rewards_marl.append(np.average(rewards_marl))
            marl_rewards_std.append(np.std(rewards_marl))
            mean_marl.append(np.mean(rewards_marl))

            # Store results in a list of dictionaries
            results.append({
                'data_file': os.path.basename(file),
                'total_rewards_masked': np.average(rewards_masked),
                'masked_rewards_std': np.std(rewards_masked),
                'mean_masked': np.mean(rewards_masked),
                'total_rewards_marl': np.average(rewards_marl),
                'marl_rewards_std': np.std(rewards_marl),
                'mean_marl': np.mean(rewards_marl)
            })

            #print("Reward base list (masked model alone):", np.average(rewards_masked))
            #print("Reward base list (MARL):", np.average(rewards_marl))

        # Convert results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'results_backup10agents_mix.csv'), index=False)

        # Plot and save the rewards
        plt.figure()
        plot_rewards(total_rewards_masked, total_rewards_marl)
        plt.savefig(os.path.join(output_dir, 'rewards_plot.png'))
        plt.close()

        # Plot and save the rewards moving average
        plt.figure()
        plot_rewards_moving_average(total_rewards_masked, total_rewards_marl)
        plt.savefig(os.path.join(output_dir, 'rewards_moving_average_plot.png'))
        plt.close()

        plt.figure()
        plot_rewards_with_std(mean_masked, masked_rewards_std, mean_marl, marl_rewards_std)
        plt.savefig(os.path.join(output_dir, 'rewards_with_std_plot.png'))
        plt.close()

        print("Total rewards (base model):", np.average(total_rewards_masked))
        print("Total rewards (complex MARL):", np.average(total_rewards_marl))