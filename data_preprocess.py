import pandas as pd
from k8node_config import K8Node
import random
import time
import numpy as np

def fix_node_names(df_data):
    """Auxiliar function that fixes problems of data containing unordered pods

    Args:
        df_data (pandas dataframe): dataframe that contains the monitorization data

    Returns:
        pandas dataframe: fixed dataframe
    """
    # Create a dictionary to map unique 'name' values to new values
    name_mapping = {}
    current_value = 0

    # Iterate through the 'name' column
    for name in df_data['name']:
        if name not in name_mapping:
            name_mapping[name] = current_value
            current_value += 1

    # Use the map method to replace the 'name' values with the new values
    df_data.loc[:, 'name'] = df_data['name'].map(name_mapping)

    return df_data


def generate_random_energy_values():
    energy_value = random.uniform(0, 1000)
    return energy_value

def calculate_energy(cpu, mem):
    cpu_power_coefficient = 0.01 + random.uniform(0.01, 0.03)  # watts per core
    memory_power_coefficient = 0.0001 + random.uniform(0.0001, 0.0003)  # watts per byte

    cpu_energy = cpu * cpu_power_coefficient
    memory_energy = mem * memory_power_coefficient

    return cpu_energy + memory_energy

def normalize(value, min_val, max_val):
    #Normalizes vlaues in a range of 0-1
    return (value - min_val) / (max_val - min_val)

def read_and_create(path_node, data_path):
    """Auxiliar function that reads all the required data for the program and initializies all k8Node structures

    Args:
        path_node (string): path to node configuration data
    Returns:
        dataframe, dataframe, list(K8Node): all data and auxiliar information ready to use
    """
    df_node = pd.read_csv(path_node)
    columns_to_read = ['name', 'cpu', 'ram', 'creation_time', 'exec_time', 'real_cpu', 'real_ram']
    df_pods = pd.read_csv(data_path, usecols=columns_to_read)

    name_to_id = {name: idx for idx, name in enumerate(df_pods['name'].unique())}

    # Replace the values in the 'names' column with numerical IDs
    #df_pods['name'] = df_pods['name'].map(name_to_id)
    #df_pods['name'] = df_pods['name']

    nodes_names = df_node['name']
    # Initiliaze all node infromation
    nodes_info = []
    for i in df_node.values.tolist():
        #print(i)
        parameters = {
            'max_cpu' : i[1],
            'max_ram' : i[2],
            'degree' : i[3],
            'cpu_potency': i[4],
            'ram_potency': i[5],
            'energy_links': i[8],
        }
        nodes_info.append(K8Node(str(i[0]), parameters, i[6], i[7]))
        
    pods = []
    for i in df_pods.values.tolist():
        pods.append([str(i[0]), i[1], i[2], i[3], i[4], i[5], i[6]])
    #for i in nodes_info:
    #    i.update_resources(random.uniform(0,25), random.uniform(0,4))
        

    # We also append our fake node, where non allocated pods will be "stored"
    parameters = {
            'max_cpu' : -1,
            'max_ram' : -1,
            'degree' : 0,
            'cpu_potency': 1,
            'ram_potency': 1,
            'energy_links':1
        }
    #nodes_info.append(K8Node(max(nodes_names)+1, parameters))
    nodes_info.append(K8Node("Fake", parameters, '', ''))
    #df_data.drop(df_data[df_data['type'] == 'node'].index, inplace=True)
    
    
    #print(df_node)
    #print(df_pods.describe())

    return df_node,pods,nodes_info


def read_and_create_multiagent(path_node, data_path, nagents):
    """Auxiliar function that reads all the required data for the program and initializies all k8Node structures

    Args:
        path_node (string): path to node configuration data
    Returns:
        dataframe, dataframe, list(K8Node): all data and auxiliar information ready to use
    """
    np.random.seed(1)
    df_node = pd.read_csv(path_node)
    columns_to_read = ['name', 'cpu', 'ram', 'creation_time', 'exec_time', 'real_cpu', 'real_ram']
    df_pods = pd.read_csv(data_path, usecols=columns_to_read)

    #df_pods = df_pods.rename(columns={'cpu_milli': 'cpu', 'memory_mib': 'ram'})
    #df_pods['cpu_normalised'] = df_pods['cpu'].apply(lambda x: normalize(x, 0, 20))
    #df_pods['ram_normalised'] = df_pods['ram'].apply(lambda x: normalize(x, 0, 32))

    #df_pods['exec_time'] = (df_pods['deletion_time'] - df_pods['creation_time'])

    name_to_id = {name: idx for idx, name in enumerate(df_pods['name'].unique())}

    # Replace the values in the 'names' column with numerical IDs
    #df_pods['name'] = df_pods['name'].map(name_to_id)
    #df_pods['name'] = df_pods['name']

    nodes_names = df_node['name']
    
    agent_dfs = {
        f"agent{a}": df_node.assign(
            max_CPU=df_node['max_CPU'] * np.random.uniform(0.5, 2, size=len(df_node)),
            max_RAM=df_node['max_RAM'] * np.random.uniform(0.5, 2, size=len(df_node))
        ).round({'max_CPU': 1, 'max_RAM': 1})
        for a in range(nagents)
    }
    
    # Initiliaze all node infromation
    nodes_info = {f"agent{a}": [] for a in range(nagents)}
    for a in range(nagents):
        for i in agent_dfs[f"agent{a}"].values.tolist():
            #print(i)
            parameters = {
                'max_cpu' : i[1],
                'max_ram' : i[2],
                'degree' : i[3],
                'cpu_potency': i[4],
                'ram_potency': i[5],
                'energy_links': i[8],
            }
            nodes_info[f"agent{a}"].append(K8Node(str(i[0]), parameters, i[6], i[7]))
        
    pods = []
    for i in df_pods.values.tolist():
        pods.append([str(i[0]), i[1], i[2], i[3], i[4], i[5], i[6]])
    #for i in nodes_info:
    #    i.update_resources(random.uniform(0,25), random.uniform(0,4))
        

    # We also append our fake node, where non allocated pods will be "stored"
    parameters = {
            'max_cpu' : -1,
            'max_ram' : -1,
            'degree' : 0,
            'cpu_potency': 1,
            'ram_potency': 1,
            'energy_links':1
        }
    #nodes_info.append(K8Node(max(nodes_names)+1, parameters))
    #nodes_info.append(K8Node("Fake", parameters, '', ''))
    #df_data.drop(df_data[df_data['type'] == 'node'].index, inplace=True)
    
    
    #print(df_node)
    #print(df_pods.describe())

    return agent_dfs,pods,nodes_info