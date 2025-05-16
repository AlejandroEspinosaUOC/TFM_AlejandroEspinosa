import csv
import random

def generate_data(filename, seed, N):
    # Set the seed for reproducibility
    """
    Generate node data for a Emmulated Kubernetes cluster.

    This function generates a CSV file at the given filename with N nodes, 
    each with a random max_CPU and max_RAM. The connections between nodes
    are also randomly generated. A potency is calculated for each node based
    on the max_CPU and max_RAM, and an energy connection value is calculated
    based on the max_CPU and max_RAM. The seed is used to ensure reproducibility.

    Parameters
    ----------
    filename : str
        The filename to write the CSV data to.
    seed : int
        The seed used for generating random numbers.
    N : int
        The number of nodes to generate.

    Returns
    -------
    No return value.
    """
    random.seed(seed)

    # Generate names dynamically
    names = ["codeco_master"] + [f"codeco_node{i}" for i in range(1, N)]

    # Generate connections dynamically
    connections = {}
    connections = {}
    for name in names:
        if name == "codeco_master":
            # codeco_master connects to all other nodes
            connections[name] = names[1:]
        else:
            # Each other node connects to a random subset of other nodes, including codeco_master
            possible_connections = [other for other in names if other != name]
            connections[name] = random.sample(possible_connections, random.randint(1, N-1))

    # Generate latency ranges dynamically
    latency_ranges = {}
    for name in names:
        if name == "codeco_master":
            # codeco_master has latency values for all other nodes
            latency_ranges[name] = [round(random.uniform(1.0, 5.0), 2) for _ in range(N-1)]
        else:
            num_connections = len(connections[name])
            latency_ranges[name] = [round(random.uniform(1.0, 5.0), 2) for _ in range(num_connections)]

    # Function to generate potency based on max_CPU and max_RAM
    def generate_potency(max_value, base_potency, scale_factor):
        return base_potency + (max_value / 100) * scale_factor

    # Function to generate energy connections based on max_CPU and max_RAM
    def generate_energy_connections(max_cpu, max_ram):
        return 0.6 + (max_cpu / 100) * 0.2 + (max_ram / 256) * 0.2

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "name", "max_CPU", "max_RAM", "degree", "cpu_potency", "ram_potency",
            "connections", "latency", "energy_connections"
        ])

        for name in names:
            max_cpu = random.randint(30, 100)
            max_ram = random.randint(16, 256)
            degree = len(connections[name])
            cpu_potency = generate_potency(max_cpu, 0.6, 0.4)
            ram_potency = generate_potency(max_ram, 0.4, 0.45)
            energy_connections = generate_energy_connections(max_cpu, max_ram)

            conn_str = ";".join(connections[name])
            latency_str = ";".join(map(str, latency_ranges[name]))

            writer.writerow([
                name, max_cpu, max_ram, degree, cpu_potency, ram_potency,
                conn_str, latency_str, energy_connections
            ])
            
# Generate the data and save to a CSV file

nagents = 10
for i in range(1, 20):
    generate_data(f'data/node_data10/node_info_tfm_seed{i}.csv', i,nagents)