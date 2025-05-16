import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

import ray
import wandb
from codecarbon import track_emissions

from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import betting_training_env as tfm

wandb.init(project="tfm_tests2", group="experiment")

parser = add_rllib_example_script_args(
    default_iters=200,
    default_timesteps=15000,
    default_reward=0.0,
)

@track_emissions()
def training_marl(base_config, nagents, csv_file):
    """
    Train a MARL algorithm with the given configuration.

    Args:
        base_config: A `PPOConfig` object that specifies the configuration for the algorithm.
        nagents: The number of agents in the environment.
        csv_file: The path to a CSV file where the mean rewards for each agent will be logged.
    """
    algo = base_config.build_algo()
    checkpoint_dir = "tests"
    # Write header to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Iteration"] + [f"Agent{i}_Mean_Reward" for i in range(nagents)]
        writer.writerow(header)

    for i in range(20):
        result = algo.train()
        print(f"Iteration {i}: {result}")

        mean_rewards = [
            result["env_runners"]["policy_reward_mean"][f"agent{agent_id}"]
            for agent_id in range(nagents)
        ]

        # Log metrics to wandb
        log_metrics = {"Iteration": i}
        for agent_id, reward in enumerate(mean_rewards):
            log_metrics[f"Agent{agent_id}_Mean_Reward"] = reward
        wandb.log(log_metrics)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i] + mean_rewards)

        checkpoint_path = algo.save(checkpoint_dir)
        print(f"Checkpoint saved at {checkpoint_path}")

    # Stop Ray
    ray.shutdown()

    plt.figure(figsize=(10, 6))
    for agent_id in range(nagents):
        plt.plot(mean_rewards[agent_id], label=f"Agent {agent_id}")

    plt.title("Mean Rewards per Agent Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.grid(True)
    plt.show()    


if __name__ == "__main__":
    args = parser.parse_args()

    assert args.num_agents > 0, "Must set --num-agents > 0 when running this script!"
    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    env_config = {"num_agents": args.num_agents, "padding": 20, "verbose": 1}
    nagents = env_config["num_agents"]

    data_agents = []

    for root, dirs, files in os.walk("synthetic_data/"):
        for file in files:
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(root, file))
                data_agents.append(df)

    env_config["data_agents"] = data_agents
        
    # Register parallel PettingZoo environment
    register_env("env", lambda _: ParallelPettingZooEnv(tfm.TFMCodecoEnv(env_config)))

    # Define policies
    policies = {
        f"agent{i}": PolicySpec() for i in range(nagents)
    }
    ray.init()

    # Define policy mapping function
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return f"agent{int(agent_id[-1]) % nagents}"

    # RLlib Configuration
    base_config = (
        PPOConfig()
        .environment("env",
                     disable_env_checking=True)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model={"vf_share_layers": True},
            vf_loss_coeff=0.005,
            use_critic=True,
            use_gae=True,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={f"agent{i}": RLModuleSpec() for i in range(nagents)}
            ),
        )
        .env_runners(
            num_env_runners=4,  
            rollout_fragment_length='auto',
            batch_mode="complete_episodes" 
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
    )
    base_config.export_native_model_files = True
    
    csv_file = "csv_files/marltfm_codecarbon.csv"
    
    
    training_marl(base_config, nagents, csv_file)
