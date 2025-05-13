import copy
import gymnasium as gym
import numpy as np
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


class CodecoEnv(ParallelEnv, gym.Env):
    metadata = {"name":"codeco_env","render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config, render_mode="human", size=10):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        # Will be configured
        self.pod_pending = copy.deepcopy(config["df_data"])
        self.pod_pending_reset = copy.deepcopy(config["df_data"])
        self.n_agents = config["num_agents"]
        #self.agents = [f'agent{i}' for i in range(self.n_agents)]
        self.agents = [f'agent{i}' for i in range(self.n_agents)]
        self.possible_agents = self.agents[:]
        self.k8semmulators = {a: None for a in self.possible_agents}
        self.npods = [0 for i in range(self.n_agents)]
        self.numeric_to_names = {a: None for a in self.possible_agents}
        self.name_to_numerics = {a: None for a in self.possible_agents}
        self.states = {a: None for a in self.possible_agents}
        self.possible_actions = {a: None for a in self.possible_agents}
        self.invalid_actions = {a: None for a in self.possible_agents}
        self.render_mode = "human"
        self.padding = config["padding"]
        # Hardcoded, should be changed, only used if padding is True
        self.max_size = 20        
        self.agent_selection = self.agents[0]
                
        self.observation_spaces = {}
        self.action_spaces = {}
        for count, i in enumerate(self.possible_agents):
            self.k8semmulators[i] = config["emmulators"][count]
            nodes_names = list(config["df_node"][count]["name"])
            nodes_names.append("Fake")
            n_nodes = len(config["df_node"][count]["name"].unique())

            allocation = len(nodes_names)
            allocation_aux = list(range(allocation))
            numeric_to_name = dict(zip(allocation_aux, nodes_names))
            name_to_numeric = dict(zip(nodes_names, allocation_aux))
            
            self.numeric_to_names[str(i)] = copy.deepcopy(numeric_to_name)
            self.name_to_numerics[str(i)] = copy.deepcopy(name_to_numeric)
            
            if self.padding:
                low = np.array([-1, -1] + ([-1, -1, -1, -1] * self.max_size), dtype=np.float32)
                high = np.array(
                    [np.float32("inf"), np.float32("inf")]
                    + (
                        [
                            np.float32("inf"),
                            np.float32("inf"),
                            np.float32("inf"),
                            np.float32("inf")
                                                
                        ]
                        * self.max_size
                    ),
                    dtype=np.float32,
                )
            else:
                low = np.array([-1, -1] + ([-1, -1, -1, -1] * n_nodes), dtype=np.float32)
                high = np.array(
                    [np.float32("inf"), np.float32("inf")]
                    + (
                        [
                            np.float32("inf"),
                            np.float32("inf"),
                            np.float32("inf"),
                            np.float32("inf")
                                                
                        ]
                        * n_nodes
                    ),
                    dtype=np.float32,
                )

            aux_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

            self.observation_spaces[str(i)] = copy.deepcopy(aux_space)
            count += 1
            self.action_spaces[i] = gym.spaces.Discrete(allocation)
            self.possible_actions[i] = list(range(self.max_size))
            if self.padding:
                self.invalid_actions[i] = list(self.possible_actions[i])[(n_nodes):]
            
            state_aux = np.float32(
            [self.pod_pending[0][1], self.pod_pending[0][2]] + self.k8semmulators[i].info_nodes
            )
            
            self.states[i] = state_aux
        
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        infos = {a: {} for a in self.agents}
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        
        self.window = None
        self.clock = None
        self.inner_rect_count = {a: 0 for a in self.possible_agents}
        for agent, value in self.k8semmulators.items():
            # Substract Fake node
            self.inner_rect_count[agent] = len(value.cluster_state) - 1
        self.inner_rect_count['circle'] = 0
        #print(self.observation_spaces)
        #print(self.action_spaces)

    def reset(self, seed=None, options=None):
        self.pod_pending = copy.deepcopy(self.pod_pending_reset)
        map(lambda i: i.reset_emmulator(), self.k8semmulators.values())
        
        for agent in self.possible_agents:
                #aux = list(self.observation_space(agent).sample())
                aux = [0.0] * 82
                aux[0] = self.pod_pending[0][1]
                aux[1] = self.pod_pending[0][2]
                i = 2
                for j in self.k8semmulators[agent].cluster_state:
                    available_cpu = np.float32(round(j.Resources["Max_cpu"] - j.Used["Cpu"], 5))
                    available_ram = np.float32(round(j.Resources["Max_ram"] - j.Used["Ram"], 5))

                    # Calculate the total CPU and RAM
                    total_cpu = np.float32(j.Resources["Max_cpu"])
                    total_ram = np.float32(j.Resources["Max_ram"])

                    # Compute the available percentage
                    available_cpu_percentage = available_cpu / total_cpu
                    available_ram_percentage = available_ram / total_ram

                    # Ensure the percentages are between 0 and 1
                    available_cpu_percentage = np.clip(available_cpu_percentage, 0, 1)
                    available_ram_percentage = np.clip(available_ram_percentage, 0, 1)

                    # Store the percentages in the aux list
                    aux[i] = available_cpu_percentage
                    aux[i+1] = available_ram_percentage
                    
                    # Init energy and resilience with 0
                    i += 4

                
                if self.padding:
                    for j in range(len(self.k8semmulators[agent].cluster_state)*4 + 2, len(aux)):
                        aux[j] = -1 
                
                estat = aux

                self.states[agent] = estat
        infos = {a: {} for a in self.possible_agents}
        rewards = {a: 0 for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        self.window = None
        self.clock = None
        return (self.states, infos)   

    def solve_betting(self, rewards, corrects, actions, info_states_ini):
        best = -1
        best_agent = None
        for i, agent in enumerate(self.agents):
            action_num = actions[agent]
            action = self.numeric_to_names[agent][action_num]
            #print(action)
            if action != "Fake" and rewards[agent] > best and corrects[agent]:
                best = rewards[agent]
                best_agent = agent
        
        if best_agent is None:
            best_agent = self.agents[0]
            best = rewards[self.agents[0]]
            print("All agents were incorrect")
                
        for i, agent in enumerate(self.agents):
            if agent != best_agent and corrects[agent]:
                #print("Before", self.k8semmulators[agent].npods)
                action_num = actions[agent]
                action = self.numeric_to_names[agent][action_num]
                if action != "Fake":
                    nodes_info = self.k8semmulators[agent].remove_allocation(action, self.pod_pending[0][0])
                    info_states_ini[agent] = nodes_info
                    # Lost the betting, enchance criteria!
                    rewards[agent] -= 0
                #print("After", self.k8semmulators[agent].npods)
                    
        return best_agent, rewards[best_agent], info_states_ini        
        #print(best_agent, self.pod_pending[0][0])
    
    def get_action_masks(self) -> list[list[bool]]:
        mask = []
        for agent in self.agents:
            mask.append([action not in self.invalid_actions[agent] for action in self.possible_actions[agent]])
        return mask
    
    def get_percnt_available(self, agent):
        return self.k8semmulators[agent].get_percnt_available()
    
    def step(self, actions, marl={}):
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        corrects = {a: True for a in self.agents}
        info_states = {a: {} for a in self.agents}
        infos = {a: {} for a in self.agents}
        #print(actions)
        
        for agent in self.agents:
            action_num = actions[agent]
            action = self.numeric_to_names[agent][action_num]
            info_nodes, reward_ret, correct = self.k8semmulators[agent].new_pod_enters(self.pod_pending[0], action, mode="rl")
                
            if self.padding:
                 while len(info_nodes) < (4 * self.max_size):
                    info_nodes.append(-1)
                
            rewards[agent] = reward_ret
            corrects[agent] = correct
            infos[agent]["correct"] = correct
            info_states[agent] = info_nodes
            
        rigging_betting = copy.deepcopy(rewards)
        if any(marl):
            for agent in self.agents:
                if marl[agent] == 1:
                    rigging_betting[agent] = 1
                else:
                    rigging_betting[agent] = 0
        best_agent, reward_best, updated_info_states = self.solve_betting(rigging_betting, corrects, actions, info_states)
        if any(corrects.values()):
            if len(self.pod_pending) > 2:
                self.pod_pending.pop(0)
            else:
                terminations = {a: True for a in self.agents}
         
        for i in self.agents:
            if not terminations[i]:
                state = np.array(
                    [self.pod_pending[0][1], self.pod_pending[0][2]] + updated_info_states[i], dtype=np.float32
                )
            else:
                state = np.array(
                    [0.0, 0.0] + updated_info_states[i], dtype=np.float32
                )
                
            while len(state) < (4 * self.max_size + 2):
                state = np.append(state, -1)
            self.states[i] = state
        
        # Computing pods in each agent        

        #print(terminations)
        #print(self.states.keys())
        self.last_state = [self.states, rewards, terminations, truncations, infos]
        return self.states, rewards, terminations, truncations, infos

    def last(self, env=None):
        return self.last_state
            
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()