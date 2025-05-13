import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import wandb

def normalize(value, min_val, max_val):
    """
    Normalizes values in a range of 0-1
    Parameters:
        value: the value to be normalized
        min_val: the minimum value of the range
        max_val: the maximum value of the range
    Returns:
        the normalized value
    """
    return (value - min_val) / (max_val - min_val)


class CodecoEnvV2(gym.Env):
    """_summary_
    Gymnasium environment that inherits from the base Environment from Gymnasium,
    this class models the problem to be solved as well as the training process to be conducted by agents
    Args:
        gym (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Metadata needed in case we want to create a render function for our environment
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 2}

    def __init__(self, env_config):
        self.k8semmulator = env_config["emmulator"]
        
        self.pod_pending = copy.deepcopy(env_config["df_data"])
        self.pod_pending_reset = copy.deepcopy(env_config["df_data"])
        
        self.nodes_names = list(env_config["df_node"]["name"])
        allocation = len(self.nodes_names)
        allocation_aux = list(range(allocation))
        self.numeric_to_name = dict(zip(allocation_aux, self.nodes_names))
        self.name_to_numeric = dict(zip(self.nodes_names, allocation_aux))
        self.n_nodes = len(env_config["df_node"]["name"].unique())
        # Initialize first state
        self.state = np.float32(
            [self.pod_pending[0][1], self.pod_pending[0][2]] + self.k8semmulator.info_nodes
        )
        self.padding = env_config["padding"]
        if self.padding:
            self.max_size = 20
            assert len(self.state) < self.max_size, "The state is too big, maximum supported size is 20 nodes"
        else:
            self.max_size = self.n_nodes +1
        
        
        self.action_space = spaces.Discrete(self.max_size)
        self.possible_actions = list(range(self.max_size))
        if self.padding:
            self.invalid_actions = self.possible_actions[(self.n_nodes):]
        # 4 per node now
        low = np.array([0,0] + ([0,0,0,0] * self.max_size), dtype=np.float32)
        high = np.array(
            [np.float32("inf"), np.float32("inf")]
            + (
                [
                    np.float32("inf"),
                    np.float32("inf"),
                    np.float32("inf"),
                    np.float32("inf"),
                                        
                ]
                * self.max_size
            ),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.window = None
        self.clock = None
        

    def reset(self, *, seed=None, options=None):
        """
        Resetting the environment and initializing various attributes back to their original values.
        """
        super().reset(seed=seed, options=options)
        # Resetting environment
        self.pod_pending = copy.deepcopy(self.pod_pending_reset)
        self.k8semmulator.reset_emmulator()
        
        # Recreating environment from a copy of original one, to minimize errors
        aux = self.observation_space.sample()
        aux[0] = self.pod_pending[0][1]
        aux[1] = self.pod_pending[0][2]
        i = 2
        for j in self.k8semmulator.cluster_state:
            if j.kind_of_node != "Fake":
                aux[i] = np.float32(round(j.Resources["Max_cpu"] - j.Used["Cpu"], 5))
                aux[i + 1] = np.float32(
                    round(j.Resources["Max_ram"] - j.Used["Ram"], 5)
                )
                
                node_energy = (j.Used["Cpu"] * j.Resources["Cpu_potency"] + j.Used["Ram"] * j.Resources["Ram_potency"])
                links_energy = j.Resources["energy_links"] * j.Used["Node_degree"]
                aux[i+2] = (node_energy*links_energy)
                aux[i+3] = 0.0

            i += 4
        
        # Padding
        for j in range(len(self.k8semmulator.cluster_state)*3 + 2, len(aux)):
            aux[j] = -1  
        
        estat = aux
        return (estat, {})

    def action_masks(self) -> list[bool]:
        return [action not in self.invalid_actions for action in self.possible_actions]
    
    def step(self, action):
        """
        A function to take a step in the environment based on the given action.
        Updates the environment state and returns the new state, reward, and whether the episode is done.
        """
        action = self.numeric_to_name[int(action)]
        truncated = False
        done = False
        # Return info on the cluster and reward
        info_cluster, reward, correct = self.k8semmulator.new_pod_enters(self.pod_pending[0], action, "rl")
        
        # If not correct pod is the same, if correct we advance
        if correct:
            # Advance
            if len(self.pod_pending) > 1:
                self.pod_pending.pop(0)
            else:
                done = True
                
        
        if not done:
            estat = np.array(
                        [self.pod_pending[0][1], self.pod_pending[0][2]] + info_cluster, dtype=np.float32
                    )
        else:
            estat = np.array(
                        [0.0,0.0] + info_cluster, dtype=np.float32
                    )
        info = {}
        while len(estat) < (4 * self.max_size + 2):
            estat = np.append(estat, -1)
        wandb.log({"Reward":reward})
        
        return estat, reward, done, truncated, info
