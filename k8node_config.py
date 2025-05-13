from decimal import Decimal
import random
import numpy as np

class K8Node():
    """_summary_
        Class that represents a kubernetes node that runs applications (allocated in pods)
        A node has the following attributes:

        Name = numerical id for the node
        Pods_assigned = List of pods that are currently assigned and running in the node
        Used = Resources being used, represented in CPU units and ram (in MB). For more information in the CPU units explanation and usage check the D11 deliverable
        Resources = Maximum resources available in node, represented as in Used
        Kind of node = String that identifies the kind of node, currently we have 4 types:
                                                                                - Small
                                                                                - Medium
                                                                                - Large
                                                                                - Fake (represents non assignation of nodes)
    """
    def __init__(self, name, parameters, connect_nodes, latency):
        self.name = name
        self.Pods = []
        #print(connect_nodes, latency)
        result_dict = {k: v for k, v in zip(connect_nodes.split(';'), latency.split(';'))}
        self.Connections = result_dict
        
        alpha = 0.5
        beta = 5.0

        # Function to initialize CPU and RAM
        def initialize_resource(max_value, alpha, beta):
            # Generate a random value between 0 and 1 with the specified beta distribution
            random_value = np.random.beta(alpha, beta)
            # Scale the random value to the range [0, max_value]
            return int(random_value * max_value)
        #'Cpu': initialize_resource(parameters['max_cpu'], alpha, beta),
        #'Ram': initialize_resource(parameters['max_ram'], alpha, beta),
        # Initialize Used resources
        self.Used = {
            'Cpu': 0,
            'Ram': 0,
            'Energy': 0,
            'Node_failures': 1,
            'Link_failures': 1,
            'Node_degree': parameters['degree'],
            'Greenness': -1,
            'Resiliance': -1,
            'Available_bandwith': 10
        }
        self.Resources = {
            'Max_cpu' : parameters['max_cpu'],
            'Max_ram' : parameters['max_ram'],
            'Cpu_potency': 0.01 + random.uniform(0.01, 0.03),
            'Ram_potency': 0.0001 + random.uniform(0.0001, 0.0003),
            'energy_links': parameters['energy_links']
        }
        
        if self.Resources['Max_cpu'] == -1 or self.Resources['Max_ram'] == -1:
            self.kind_of_node = 'Fake'
        elif self.Resources['Max_cpu'] < 4 or self.Resources['Max_ram'] < 8:
            self.kind_of_node = 'Small'
        elif self.Resources['Max_cpu'] < 16 or self.Resources['Max_ram'] < 64:
            self.kind_of_node = 'Medium'
        else:
            self.kind_of_node = 'Large'

    def assign_pod(self, pod, exec_time, pod_resources):
        """_summary_
        Function that assigns a pod to the node
        Args:
            pod (int): id of pod to be assigned
        """
        ##### CHANGE AFTER DEMO! #####
        self.Pods.append([pod, exec_time, pod_resources])

        
    def assign_pods(self, pods):
        """_summary_
        Function that assigns pods to the node
        Args:
            pods (list of int): list of pod if to be assigned
        """
        for pod in pods:
            self.assign_pod(pod)

    def get_pod_in_execution(self):
        """
        Gets the pod in execution if any.

        Returns:
            dict(): Returns a dictionary with keys "in_execution" and "pod". If "in_execution" is 1, "pod" will contain the pod in execution. Otherwise it will be empty.
        """
        in_execution = len(self.Pods) > 0
        pod = self.Pods[0] if in_execution else []
        return {"in_execution" : in_execution, "pod" : pod}


    def remove_resources_not_pod(self, cpu, ram):
        self.Used['Cpu'] = round(float(Decimal(self.Used['Cpu']) - Decimal(cpu)), 8)
        self.Used['Ram'] = round(float(Decimal(self.Used['Ram']) - Decimal(ram)), 8)
        self.Used['Energy'] = self.Used['Cpu']*self.Resources['Cpu_potency'] + self.Used['Ram']*self.Resources['Ram_potency']
        
    def remove_resources(self, cpu, ram, element):
        """_summary_
        Function that removes resources being used in the node
        Args:
            cpu (float): cpu in cpu units
            ram (float): ram in MB
        """

        self.Used['Cpu'] = round(float(Decimal(self.Used['Cpu']) - Decimal(cpu)), 8)
        self.Used['Ram'] = round(float(Decimal(self.Used['Ram']) - Decimal(ram)), 8)
        self.Used['Energy'] = self.Used['Cpu']*self.Resources['Cpu_potency'] + self.Used['Ram']*self.Resources['Ram_potency']

        elem_to_remove = []
        for i in self.Pods:
            if i[0] == element:
                elem_to_remove = i
                break
        self.Pods.remove(elem_to_remove)
        
    def update_resources(self, cpu, ram):
    
        """_summary_
        Function that updates resources being used in the node
        Args:
            cpu (float): cpu in cpu units
            ram (float): ram in MB
        """
        self.Used['Cpu'] = round(float(Decimal(self.Used['Cpu']) + Decimal(cpu)),8)
        self.Used['Ram'] = round(float(Decimal(self.Used['Ram']) + Decimal(ram)),8)
        self.Used['Energy'] = self.Used['Cpu']*self.Resources['Cpu_potency'] + self.Used['Ram']*self.Resources['Ram_potency']


    def update_timers(self, time_passed):
        elements_to_remove = []

        for i in self.Pods:
            i[1] -= time_passed
            if i[1] <= 0:
                elements_to_remove.append(i)

        for element in elements_to_remove:
            self.remove_resources(element[2]['cpu'], element[2]['ram'], element)
        
        return len(elements_to_remove)
                
                
    def __str__(self):
        pods = ', '.join(map(str, [i[0] for i in self.Pods]))
        #print(pods)
        return 'K8Node representation: \n Name: ' + str(self.name) + '\n Type of node: ' + self.kind_of_node + '\n Resources of node: \n ---CPU: ' + str(self.Resources['Max_cpu']) +  '\n ---RAM: ' + str(self.Resources['Max_ram']) + '\n Used resources: \n ---CPU: ' + str(self.Used['Cpu']) + '\n ---RAM: ' + str(self.Used['Ram']) + '\n Pods_assigned to Node: ' + pods