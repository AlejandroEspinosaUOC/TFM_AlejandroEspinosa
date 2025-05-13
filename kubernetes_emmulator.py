import copy
import numpy as np

# Pot anar a clase útil o algo de l'estil
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

class KubernetesEmulator():
    
    def get_pods_running(self):
        pods_per_node = []
        for i in self.cluster_state:
            if i.kind_of_node != "Fake":
                pods_per_node.append(len(i.Pods))
                
        return pods_per_node
    
    def get_percnt_available(self):
        """
        Function that returns the percentage of available resources in the cluster
        """
        total_cpu = 0
        total_ram = 0
        for i in self.cluster_state:
            if i.kind_of_node != "Fake":
                total_cpu += i.Resources["Max_cpu"]
                total_ram += i.Resources["Max_ram"]

        used_cpu = 0
        used_ram = 0
        for i in self.cluster_state:
            if i.kind_of_node != "Fake":
                used_cpu += i.Used["Cpu"]
                used_ram += i.Used["Ram"]

        return [used_cpu / total_cpu, used_ram / total_ram]
        
    
    def update_info(self, mode="normal"):
        self.info_nodes = []
        self.info_nodes_dict = {}

        for i in self.cluster_state:
            if i.kind_of_node != "Fake":
                if mode == "init":
                    if i.name == "codeco_master":
                        self.master = i

                self.info_nodes_dict[i.name] = {}

                # Calculate available CPU and RAM
                available_cpu = np.float32(round(i.Resources["Max_cpu"] - i.Used["Cpu"], 5))
                available_ram = np.float32(round(i.Resources["Max_ram"] - i.Used["Ram"], 5))

                # Calculate the total CPU and RAM
                total_cpu = np.float32(i.Resources["Max_cpu"])
                total_ram = np.float32(i.Resources["Max_ram"])

                # Compute the available percentage
                available_cpu_percentage = available_cpu / total_cpu
                available_ram_percentage = available_ram / total_ram

                # Ensure the percentages are between 0 and 1
                available_cpu_percentage = np.clip(available_cpu_percentage, 0, 1)
                available_ram_percentage = np.clip(available_ram_percentage, 0, 1)

                # Update node info with available percentages
                self.info_nodes.append(available_cpu_percentage)
                self.info_nodes.append(available_ram_percentage)
                self.info_nodes_dict[i.name]["Remaining_CPU"] = available_cpu_percentage
                self.info_nodes_dict[i.name]["Remaining_RAM"] = available_ram_percentage

                # Calculate energy consumption
                node_energy = (i.Used["Cpu"] * i.Resources["Cpu_potency"] + i.Used["Ram"] * i.Resources["Ram_potency"])
                links_energy = i.Resources["energy_links"] * i.Used["Node_degree"]
                energy_consumed = node_energy * links_energy

                # Update node info with energy consumption
                self.info_nodes.append(energy_consumed)
                self.info_nodes_dict[i.name]["Energy_Consumed"] = energy_consumed

                if mode == "init":
                    self.pods_in_execution = []
                    
    def __init__(self, emmulator_config):
        self.cluster_state = copy.deepcopy(emmulator_config["nodes_info"])
        self.cluster_state_reset = copy.deepcopy(emmulator_config["nodes_info"])

        self.time_passed = 0 
        self.exec_queue = []
        self.nodes_metadata = []
        
        self.w_ld_cpu = emmulator_config["w_ld_cpu"] #5 #0
        self.w_ld_ram = emmulator_config["w_ld_ram"] #5 #0
        self.w_eo = emmulator_config["w_eo"] #0 #3
        self.w_tc = emmulator_config["w_tc"] #0 #0
        self.w_ec = emmulator_config["w_ec"] #5 #5
        self.actual_time = -1
        self.npods = 0

        self.reward_max_value = (
            self.w_ld_cpu + self.w_ld_ram + (self.w_eo * 2) + self.w_tc + self.w_ec
        )
        
        self.update_info("init")
        
    def reset_emmulator(self):
        self.cluster_state = copy.deepcopy(self.cluster_state_reset)
        
        self.time_passed = 0 
        self.exec_queue = []
        self.nodes_metadata = []
        self.npods = 0

        self.update_info("init")
    
    def compute_best_action(self, pod):
        """
        A function to compute the best action based on CPU and RAM values of a pod.
        
        Parameters:
            self: the object instance
            pod: a list containing CPU and RAM values of a pod
        
        Returns:
            The best action to take
        """
        cpu = pod[1]
        ram = pod[2]

        best_action = {}
        #print(pod)
        for i in self.cluster_state:
            aux_action = i.name
            correct = self.check_correct_action_reassign(aux_action, pod[1], pod[2])

            if correct and i.kind_of_node != "Fake":
                i.update_resources(cpu, ram)
            best_action[i.name] = self.compute_reward_rl(cpu, ram, aux_action, correct)

            if correct and i.kind_of_node != "Fake":
                i.remove_resources_not_pod(cpu, ram)

        #print(best_action)
        best_one = max(best_action, key=best_action.get)
        max_reward = best_action[best_one]
        #print("Best one: ", best_one, max_reward)
        return best_one
    
    def compute_workload(self):
        """_summary_
        Function that computes the workload for all nodes

        Args:
            current_allocation (list[int]): Allocation of pods to nodes
            correct_checked (Boolean): Bool that changes the behaviour of the function for the case in which a pod is allocated in a full node

        Returns:
            _type_: (list containing all workloads, bool indicating if the action is correct)
        """
        workload_vect_cpu = []
        workload_vect_ram = []
        for node in self.cluster_state:
            if node.Used["Cpu"] < 0 or node.Used["Ram"] < 0:
                print(node.Used["Cpu"])
                print(node.Used["Ram"])
                print("NEGATIVE RESOURCES")
                print(node.name, node.Used["Cpu"], node.Used["Ram"])
            if node.kind_of_node != "Fake":
                # Value between 0 and 1
                value_cpu = node.Used["Cpu"] / node.Resources["Max_cpu"]
                value_ram = node.Used["Ram"] / node.Resources["Max_ram"]
                workload_vect_cpu.append(value_cpu)
                workload_vect_ram.append(value_ram)

        return [workload_vect_cpu, workload_vect_ram]

    def greenness_module(self, cpu, ram, action):
        """
        Calculate the energy cost based on the CPU and RAM usage for a specific action.

        Parameters:
            cpu (int): The CPU usage.
            ram (int): The RAM usage.
            action (str): The specific action to calculate energy cost for.

        Returns:
            int: The calculated energy cost.
        """
        energy_cost = 0
        for i in self.cluster_state:
            if i.name == action and i.kind_of_node != "Fake":
                energy_cost = (
                    cpu * i.Resources["Cpu_potency"] + ram * i.Resources["Ram_potency"]
                )
                break

        return energy_cost

    def resilience_module(self, action):
        """
        Calculate the resilience of a given action based on the nodes information.

        :param action: The action to calculate resilience for.
        :return: The calculated resilience value.
        """
        resilience = 0
        for i in self.cluster_state:
            if i.name == action:
                resilience = (
                    i.Used["Node_degree"]
                    / i.Used["Node_failures"]
                    / i.Used["Link_failures"]
                )
        return resilience

    def compute_task_collocation(self, action, ram):
        """
        A function to compute the task collocation based on the given action and RAM.
        Parameters:
            - action: a string representing the action taken
            - ram: a float representing the amount of RAM
        Returns:
            - float: the task_cost divided by the maximum cost
        """
        task_cost = 0

        # Assuming that all pods are collocated from the master node, even reallocated ones
        if action not in ("codeco_master", "Fake"):
            latency = float(self.master.Connections[str(action)])
            task_cost = latency * float(ram)
            
        # Maximum values for normalization, hardcoded for now
        max_latency = 5.12
        max_cost = 32 * max_latency
        return task_cost / max_cost
    
    def compute_reward_rl(self, cpu, ram, action, correct):
        """
        A function to calculate the reward based on CPU, RAM, action, and correctness of the action.
        Parameters:
        - cpu: the CPU information
        - ram: the RAM information
        - action: the action taken
        - correct: a boolean indicating if the action was correct
        Returns:
        - reward: a calculated reward value
        """
        if action == "Fake":
            self.last_reward = -1.0
            self.energy_optimization_out = 0.0
            return -1.0

        if correct:
            load = self.compute_workload()
            # Value in range 0 and 1, splitted in cpu and ram
            load_distribution_cpu = 1 - np.std(load[0])
            load_distribution_ram = 1 - np.std(load[1])
            # Value in range 0 and 1
            aux_energy = self.greenness_module(cpu, ram, action)
            energy_optimization = max(1 - aux_energy * 2, 0)
            # Value in range 0 and 1
            task_collocation_cost = self.compute_task_collocation(action, ram)
            encourage_collocation = 1.0

            self.load_cpu_out = load_distribution_cpu
            self.load_ram_out = load_distribution_ram
            self.mean_load_cpu = np.mean(load[0])
            self.mean_load_ram = np.mean(load[1])
            self.load_distribution_cpu = load_distribution_cpu
            self.load_distribution_ram = load_distribution_ram
            self.energy_optimization_out = aux_energy
            self.task_collocation_out = task_collocation_cost
            self.encourage_collocation_out = encourage_collocation

            act_reward = (
                load_distribution_cpu * self.w_ld_cpu
                + load_distribution_ram * self.w_ld_ram
                + energy_optimization * self.w_eo
                + task_collocation_cost * self.w_tc
                + encourage_collocation * self.w_ec
            )

            reward = normalize(act_reward, 0, self.reward_max_value)
            self.last_reward = normalize(act_reward, 0, self.reward_max_value)
        else:
            reward = -20
            # reward = -1 - self.last_reward
            self.energy_optimization_out = 0.0
            self.last_reward = -5
        return reward
        
    def check_correct_action_reassign(self, action, cpu, ram):
        """
        Check if the provided action can be applied based on the state of the system after placing it.

        Parameters:
        - action: The action
        - cpu: The CPU resources
        - ram: The RAM resources

        Returns:
        - correct_action: A boolean indicating if the action is correct
        """
        correct_action = True
        for node in self.cluster_state:
            if (str(node.name) == str(action)) and node.kind_of_node != "Fake":
                if ((node.Used["Cpu"] + cpu) > node.Resources["Max_cpu"]) or (
                    node.Used["Ram"] + ram
                ) > node.Resources["Max_ram"]:
                    correct_action = False

        return correct_action
    
    def update_task_timers(self, time):
        """
        Update task timers based on the given time. If time is -1, set the time_passed to 2, otherwise set it to the given time.
        Update the actual_time by adding the time_passed.
        Remove elements from the exec_queue if their remaining time is less than or equal to 0, and remove corresponding resources from the nodes_information.
        Update the info_nodes with the CPU and RAM usage information for each non-Fake node.
        """
        if time == -1:
            self.time_passed = 1
        else:
            self.time_passed = time

        self.actual_time += self.time_passed

        elements_to_remove = []
        for i in self.exec_queue:
            i[4] -= self.time_passed
            if i[4] <= 0:
                elements_to_remove.append(i)

        for element in elements_to_remove:
            for j in self.cluster_state:
                if j.name == element[0]:
                    j.remove_resources(element[2], element[3], element[1])
                    break
            self.exec_queue.remove(element)


        self.update_info()


    def calculate_time_to_next_resource_release(self):
        remaining_times = []
        for node in self.cluster_state:
            for pod in node.Pods:
                updated_remainin_time = pod[1]
                remaining_times.append(updated_remainin_time) 
            
        if len(remaining_times) == 0:
            return -1
        else:
            return min(remaining_times)
    
    def update_allocation(self, new_to, pod):
        """_summary_
        Function that redisttributes resources and pods between nodes after an action is taken
        Args:
            df (Pandas dataframe): dataset
            new_to (_type_): action
        """
        name = pod[0]
        cpu = pod[5]
        ram = pod[6]
        exec_time = pod[4]

        for i in self.cluster_state:
            if str(i.name) == str(new_to):
                #print("ASSIGNING POD", str(new_to))
                pod_resources = {
                    "cpu": cpu,
                    "ram": ram,
                }
                i.assign_pod(name, exec_time, pod_resources)
                i.update_resources(cpu, ram)

        self.update_info()

    def remove_allocation(self, from_node, pod_name):
        #print("HEY", from_node, pod_name)
        success = 0
        for node in self.cluster_state:
            #print(node)
            if str(node.name) == str(from_node):
                for node_pod in node.Pods:
                    node_pod_name = node_pod[0]
                    if node_pod_name == pod_name:
                        node_pod_cpu = node_pod[2]["cpu"]
                        node_pod_ram = node_pod[2]["ram"]
                        node.remove_resources(node_pod_cpu,node_pod_ram,node_pod_name)
                        for i in node.Pods:
                            if i[0] == pod_name:
                                node.Pods.remove(i)
                        to_rem = None
                        for i in self.exec_queue:
                            if i[0] == from_node and i[1] == pod_name:
                                to_rem = i
                        if to_rem != None:
                            self.exec_queue.remove(to_rem)
                        success += 1
                        self.npods -= 1
                        break
                    
        self.update_info()
        return self.info_nodes
        assert success == 1, "Allocation was not removed"

    def reallocate_pod(self, from_node, to_node):
        # find current pod executed by node (ensure there is one on execution)
        for node in self.cluster_state:
            if str(node.name) == str(from_node):
                pod_in_execution = node.get_pod_in_execution()
                assert pod_in_execution["in_execution"] == 1 # SHOULD THIS BE CHANGED BY THE USE OF A VARIABLE CALLED SUCCESS THAT IS RETURNED?
                pod = pod_in_execution["pod"]

        # free resources of the node
        pod_name = pod[0]
        self.remove_allocation(from_node, pod_name)

        # allocate resources to the new node

        # this is quite a "guarrada" in my opinion, because it should not be needed
        name = pod[0]
        cpu = pod[2]["cpu"]
        ram = pod[2]["ram"]
        exec_time = pod[1]
        _pod = [name, '', '', '', exec_time, cpu, ram]
        #
        self.update_allocation(to_node, _pod)

        # increase time by step
        self.update_task_timers(-1)



    # Idea, fer que es puguin passar els pods manualment, tenint prioritat per sobre dels del dataset?¿
    # new_pod_enters(self, pod, node_to)
    def new_pod_enters(self, pod, node_to, mode="default"):
        """
        A function to emulate an action within the system. Takes an action as input and returns various state information and rewards.
        """
        correct = self.check_correct_action_reassign(node_to, pod[1], pod[2])

        if node_to == "Fake":
            reward_ret = self.compute_reward_rl(pod[1], pod[2], node_to, correct)
            self.pointer_pods = 0
            
            # Mirar com funca aquesta funció amb el nou mode¿? -> Sembla que bé
            self.update_task_timers(self.calculate_time_to_next_resource_release())
        elif not correct:
            reward_ret = self.compute_reward_rl(pod[1], pod[2], node_to, correct)
            self.update_task_timers(self.calculate_time_to_next_resource_release()) # update time until more resources are available
        else:
            if pod[3] > self.actual_time:
                self.update_task_timers(pod[3] - self.actual_time)
            self.exec_queue.append([node_to, pod[0], pod[5], pod[6], pod[4]])
            self.update_allocation(node_to, pod)
            reward_ret = self.compute_reward_rl(pod[1], pod[2], node_to, correct)
            self.update_task_timers(-1)
            
            self.npods += 1
            
        #print(self.npods)
        # To see, depenent del mode?¿
        
        self.nodes_metadata = []
        for i in self.cluster_state:
            used = [
                i.name,
                i.Resources["Cpu_potency"],
                i.Resources["Max_cpu"],
                i.Used["Cpu"],
                i.Used["Cpu"] / i.Resources["Max_cpu"],
                i.Resources["Ram_potency"],
                i.Resources["Max_ram"],
                i.Used["Ram"],
                i.Used["Ram"] / i.Resources["Max_ram"],
                i.Used["Energy"]
            ]
            self.nodes_metadata.append(used)
        if mode == "rl":
            return self.info_nodes, reward_ret, correct
        else:
            return reward_ret