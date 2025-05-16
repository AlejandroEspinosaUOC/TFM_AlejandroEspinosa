import copy
import functools
from collections import deque

import gymnasium as gym
import numpy as np
import wandb
from pettingzoo import ParallelEnv

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


class TFMCodecoEnv(ParallelEnv, gym.Env):
    metadata = {"name":"tfmcodeco_env","render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config, render_mode="human"):
        # Configuring number of agents
        self.n_agents = config["num_agents"]
        
        # Verbose
        self.verbose = config["verbose"]
        
        # Establish possible agents 
        self.possible_agents = [f'agent{i}' for i in range(self.n_agents)]
        
        # Setting up data
        self.agent_data = {a: [] for a in self.possible_agents}
        self.data_reset = {a: [] for a in self.possible_agents}
        
        for i, agent in zip(config["data_agents"], self.possible_agents):
            self.agent_data[agent] = copy.deepcopy(i)
            self.data_reset[agent] = copy.deepcopy(i)
        
        # Setup deque for agent bettings
        self.agent_bettings = {a: deque([0], maxlen=3) for a in self.possible_agents}
        
        # Setting padding, by default is 20
        self.padding_size = config["padding"]
        self.padding = self.padding_size > 0
        
        self.count = 0
        assert self.padding_size >= self.n_agents, "Padding size must be greater or equal than the number of agents"
        
            
        # Auxiliar class variables
        self.biddings = {a: 0 for a in self.possible_agents}
        self.last_biddings = {a: -1 for a in self.possible_agents}
        self.bidding_ended = False
        
        # Setting first state
        self.states = self.update_state()
            
        # Setting last state
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        rewards = {a: 0 for a in self.possible_agents}
        infos = {a: {} for a in self.possible_agents}
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode   
        
        # Rendering utilities
        self.window = None
        self.clock = None 
        self.count = 0
        
        # Initialize wandb in here in case it is not already initialized
        # This is done as parallelized agents may not have access to the same wandb instance
        if not wandb.run:
            wandb.init(project="tfm_tests2", group="experiment")
            
            
    def reset(self, seed=None, options=None):

        # Setting first state
        self.agent_data = copy.deepcopy(self.data_reset)
        self.states = self.update_state()
            
        #Establish live agents (all in this case)
        self.agents = self.possible_agents[:]
        
        infos = {a: {} for a in self.possible_agents}
        rewards = {a: 0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        self.count = 0
        self.bidding_ended = False
        
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        self.window = None
        self.clock = None
        return (self.states, infos)   
    
    def update_state(self):
        """
        Updates the state of the environment. This function is called after each time step in the environment.
        It takes the current state of the environment and updates it according to the rules of the environment.
        The state is a dictionary where the keys are the agents and the values are lists of floats.
        The function returns the updated state.
        """
        states = {a: [] for a in self.possible_agents}
        for agent in self.possible_agents:
            # Flatten data_for_state if it contains nested lists
            data_for_state = list(self.agent_data[agent].iloc[0])
            if any(isinstance(i, list) for i in data_for_state):
                data_for_state = [item for sublist in data_for_state for item in sublist]

            avg_bettings = []
            for i in self.possible_agents:
                if i != agent:
                    elem = self.agent_bettings[i]
                    avg = sum(elem) / len(elem) if len(elem) > 0 else 0
                    avg_bettings.append(avg)

            # Ensure avg_bettings is a flat list
            if any(isinstance(i, list) for i in avg_bettings):
                avg_bettings = [item for sublist in avg_bettings for item in sublist]

            data_for_state.extend(avg_bettings)

            # Convert single-element arrays to scalars
            data_for_state = [x if not isinstance(x, np.ndarray) else x.item() for x in data_for_state]

            states[agent] = data_for_state

        # Final check and correction: Ensure each state is a flat list and has the correct length
        expected_length = 3 + (self.n_agents - 1)
        for agent, state in states.items():
            if len(state) != expected_length:
                # Fix the length by truncating or padding with zeros
                if len(state) > expected_length:
                    state = state[:expected_length]
                else:
                    state.extend([0] * (expected_length - len(state)))

            # Ensure the state is a flat list
            if any(isinstance(i, list) for i in state):
                state = [item for sublist in state for item in sublist]

            states[agent] = state

        return states
        
    def betting_step(self, actions):
        """
        Perform a step in the betting environment.

        Parameters
        ----------
        actions : dict
            A dictionary where the keys are the agents and the values are the actions taken by each agent.

        Returns
        -------
        norm_rewards : dict
            A dictionary where the keys are the agents and the values are the normalized rewards received by each agent after taking their actions.
        """
       
        for i in self.agents:
            self.agent_bettings[i].append(actions[i])

        rewards = {a: 0 for a in self.agents}
        sorted_list = sorted(actions.items(), key=lambda item: item[1], reverse=True)
        N = self.n_agents // 3

        subset = sorted_list[:N]

        if len(subset) >= 1:
            rewards[subset[0][0]] = 3

        # Adjust rewards based on confidence, CPU, and RAM usage
        for agent, action in actions.items():
            state = self.states[agent]
            confidence = state[0]
            cpu_usage = state[1]
            ram_usage = state[2]

            # Penalize big bets if confidence is low or CPU/RAM usage is high
            if action > 8:
                if confidence < 1 or cpu_usage > 80 or ram_usage > 80:
                    rewards[agent] = -5  # Penalize by reducing the reward

            # 
            if action <= 2:  # Considering a bet <= 50 as a low bet
                if confidence < 1:
                    rewards[agent] += 2  # Reward by increasing the reward

        norm_rewards = {a: (r - -5) / (5 - -5) for a, r in rewards.items()}
        return norm_rewards        
      
    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        log_metrics = {f"{i}_Action": int(action) for i, action in actions.items()}
        wandb.log(log_metrics)

        #if isinstance(actions, dict):
        #    actions = list(actions.values())
            
        # Initializing data sctructures for multiple agents
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        info_states = {a: {} for a in self.agents}
        infos = {a: {} for a in self.agents}
        

        rewards = self.betting_step(actions)
        log_metrics = {f"{i}_Action": float(reward) for i, reward in rewards.items()}        
        wandb.log(log_metrics)
        # TODO
        # Estats, i rewards
        # Update states, itearate data
        self.agent_data = {a: self.agent_data[a].iloc[1: , :] for a in self.agents}
        if any ([len(self.agent_data[a]) <= 0 for a in self.agents]): #len(self.)
            self.bidding_ended = True
            #print("Ending dataset")
            
        if self.bidding_ended:
            terminations = {a: True for a in self.agents}
        else:
            #print([len(self.agent_data[a]) for a in self.agents])
            self.states = self.update_state()
            self.count += 1
        
        # When dataset is finished
            
        

        if any(terminations.values()) or all(truncations.values()):
            # self.agents = []
            pass
        
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        return self.states, rewards, terminations, truncations, infos

    def last(self, env=None):
        return self.last_state
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Configuring observation and action spaces       
        low = np.array(
            [-20, -1, -1] 
            + [-1]*(self.n_agents-1), dtype=np.float32
            )
        high = np.array(
            [np.float32("inf"), np.float32("inf"), np.float32("inf")]
            + [np.float32("inf")]*(self.n_agents-1),dtype=np.float32,
        )
        
        observation_spaces = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        return observation_spaces

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gym.spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32)
    